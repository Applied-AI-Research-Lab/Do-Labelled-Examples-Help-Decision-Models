from __future__ import annotations

import json
import random

import numpy as np

from ..config import BUDGETS, HF_MODELS, SEED, TASKS, TASKS_DIR
from ..records import budget_pool
from ..utils import Timer, append_jsonl, get_logger, read_jsonl
from .common import base_parser, jobs, pred_row

log = get_logger("baselines")


def run_mock(task, recs, out, model_key="mock"):
    rng = random.Random(SEED)
    rows = []
    for r in recs:
        w = [rng.random() for _ in r["options"]]
        rows.append(pred_row(r, model_key, {o: x / sum(w) for o, x in zip(r["options"], w)}, 0.001))
    append_jsonl(out, rows)


def run_embedlr(args):
    from sklearn.linear_model import LogisticRegression
    from ..config import raw_path, records_path
    from ..utils import done_ids
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    for task in tasks:
        d = TASKS_DIR / task
        pool, emb_pool = read_jsonl(d / "pool_dedup.jsonl"), np.load(d / "emb_pool_dedup.npy").astype(np.float32)
        test_emb = np.load(d / "emb_test.npy").astype(np.float32)
        test_ids = [r["id"] for r in read_jsonl(d / "test.jsonl")]
        recs = {r["id"]: r for r in read_jsonl(records_path(task, "k0"))}
        budgets = [(f"embedlr_b{b}", np.array(budget_pool(pool, b, task))) for b in BUDGETS]
        budgets.append(("embedlr_full", np.arange(len(pool))))
        for key, idx in budgets:
            out = raw_path(args.run_name, key, task, "k0")
            if done_ids(out):
                continue
            y = [pool[i]["label"] for i in idx]
            clf = LogisticRegression(max_iter=3000, C=10.0, random_state=SEED).fit(emb_pool[idx], y)
            with Timer() as t:
                P = clf.predict_proba(test_emb)
            rows = []
            for tid, p in zip(test_ids, P):
                r = recs[tid]
                d_ = dict(zip(clf.classes_, p))
                rows.append(pred_row(r, key, {o: float(d_.get(o, 0.0)) for o in r["options"]}, t.seconds / len(P),
                                     meta={"train_size": int(len(idx))}))
            append_jsonl(out, rows)
            log.info("%s %s: trained on %d labels", task, key, len(idx))


def run_modernbert(task, recs, out, epochs=3, lr=5e-5, bs=32, max_len=128):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    labels = json.loads((TASKS_DIR / task / "labels.json").read_text())
    train = read_jsonl(TASKS_DIR / task / "pool_dedup.jsonl")
    lab2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(HF_MODELS["modernbert"])
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODELS["modernbert"], num_labels=len(labels)).cuda()
    data = [(r["text"], lab2id[r["label"]]) for r in train]

    def collate(b):
        enc = tok([x for x, _ in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b])
        return enc

    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * epochs), len(dl) * epochs)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        log.info("%s epoch %d loss %.4f", task, ep, tot / len(dl))
    model.eval()
    rows = []
    for i in range(0, len(recs), 128):
        chunk = recs[i:i + 128]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        with Timer() as t, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = model(**enc).logits.float().softmax(-1).cpu().tolist()
        rows += [pred_row(r, "modernbert", {o: p[lab2id[o]] for o in r["options"]}, t.seconds / len(chunk))
                 for r, p in zip(chunk, P)]
    append_jsonl(out, rows)


def main():
    ap = base_parser("Classifier baselines / mock")
    ap.add_argument("--which", choices=["embedlr", "modernbert", "mock"], required=True)
    args = ap.parse_args()
    if args.which == "embedlr":
        return run_embedlr(args)
    if args.which == "modernbert":
        args.conditions = "k0"
    for task, cond, recs, out in jobs(args, args.which):
        (run_modernbert if args.which == "modernbert" else run_mock)(task, recs, out)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
