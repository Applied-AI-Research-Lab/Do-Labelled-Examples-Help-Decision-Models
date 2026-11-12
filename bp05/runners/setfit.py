from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from ..config import BUDGETS, HF_MODELS, SEED, TASKS, TASKS_DIR, raw_path, records_path
from ..records import budget_pool
from ..utils import append_jsonl, done_ids, get_logger, read_jsonl
from .common import pred_row

log = get_logger("setfit")
ITERATIONS, BATCH, LR, WARMUP, MAX_LEN, HEAD_C = 20, 16, 2e-5, 0.10, 256, 10.0


def make_pairs(labels: list[str], iterations: int, seed: str) -> list[tuple[int, int, float]]:
    rng = random.Random(seed)
    by = {}
    for i, lab in enumerate(labels):
        by.setdefault(lab, []).append(i)
    classes = sorted(by)
    pairs = []
    for _ in range(iterations):
        for i, lab in enumerate(labels):
            same = [j for j in by[lab] if j != i] or [i]
            pairs.append((i, rng.choice(same), 1.0))
            other = [c for c in classes if c != lab]
            if other:
                pairs.append((i, rng.choice(by[rng.choice(other)]), 0.0))
    rng.shuffle(pairs)
    return pairs


def _pool(model, enc):
    h = model(**enc).last_hidden_state[:, -1]
    import torch
    return torch.nn.functional.normalize(h.float(), dim=-1)


def _embed(texts, tok, model, device, bs=64):
    import torch
    out = []
    model.eval()
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(_pool(model, enc).cpu().numpy())
    return np.concatenate(out)


def train_and_predict(task: str, b: int, run_name: str, max_steps: int | None):
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    out = raw_path(run_name, f"setfit_b{b}", task, "k0")
    if done_ids(out):
        log.info("%s b%d: already done (%s)", task, b, out)
        return
    d = TASKS_DIR / task
    pool = read_jsonl(d / "pool_dedup.jsonl")
    idx = budget_pool(pool, b, task)
    texts, labels = [pool[i]["text"] for i in idx], [pool[i]["label"] for i in idx]
    recs = read_jsonl(records_path(task, "k0"))
    pairs = make_pairs(labels, ITERATIONS, f"{SEED}-{task}-setfit-b{b}")
    steps = (len(pairs) + BATCH - 1) // BATCH
    if max_steps:
        steps = min(steps, max_steps)
    log.info("%s b%d: %d labelled examples, %d pairs, %d steps", task, b, len(idx), len(pairs), steps)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(HF_MODELS["embed"], padding_side="left")
    model = AutoModel.from_pretrained(HF_MODELS["embed"], dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    t0 = time.time()
    model.train()
    losses = []
    for s in range(steps):
        chunk = pairs[s * BATCH:(s + 1) * BATCH]
        a = tok([texts[i] for i, _, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        bb = tok([texts[j] for _, j, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        y = torch.tensor([t for _, _, t in chunk], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cos = (_pool(model, a) * _pool(model, bb)).sum(-1)
        loss = torch.nn.functional.mse_loss(cos.float(), y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        losses.append(loss.item())
        if (s + 1) % max(1, steps // 10) == 0 or s + 1 == steps:
            log.info("%s b%d: step %d/%d  mean loss (last chunk) %.4f", task, b, s + 1, steps,
                     float(np.mean(losses[-max(1, steps // 10):])))
    train_s = time.time() - t0

    emb_train = _embed(texts, tok, model, device)
    t1 = time.time()
    emb_test = _embed([r["text"] for r in recs], tok, model, device)
    clf = LogisticRegression(max_iter=3000, C=HEAD_C, random_state=SEED).fit(emb_train, labels)
    P = clf.predict_proba(emb_test)
    lat = (time.time() - t1) / len(recs)
    rows = []
    for r, p in zip(recs, P):
        dd = dict(zip(clf.classes_, p))
        rows.append(pred_row(r, f"setfit_b{b}", {o: float(dd.get(o, 0.0)) for o in r["options"]}, lat,
                             meta={"train_size": len(idx), "pairs": len(pairs), "steps": steps, "train_s": round(train_s, 1),
                                   "loss_first": float(np.mean(losses[:max(1, steps // 10)])),
                                   "loss_last": float(np.mean(losses[-max(1, steps // 10):]))}))
    append_jsonl(out, rows)
    acc = np.mean([row["pred"] == r["gold"] for row, r in zip(rows, recs)])
    log.info("%s b%d: done in %.0f s, accuracy %.4f -> %s", task, b, train_s, acc, out)
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--run-name", default="setfit")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: stop training after N steps")
    args = ap.parse_args()
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    for t in tasks:
        for b in [int(x) for x in args.budgets.split(",")]:
            train_and_predict(t, b, args.run_name, args.max_steps)
    log.info("SETFIT-RUNS-DONE %s", json.dumps({"tasks": tasks, "budgets": args.budgets, "run": args.run_name}))


if __name__ == "__main__":
    main()
