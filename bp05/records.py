from __future__ import annotations

import bisect
import json
import random

import numpy as np

from .config import BUDGETS, DEDUP_COSINE, K_VALUES, SEED, TASKS, TASKS_DIR, records_path
from .data import TOPIC_NAMES, slug
from .utils import get_logger, read_jsonl, write_jsonl

log = get_logger("records")

STATE_PREFIX = {"banking77": "Customer message: ", "bitext": "Customer message: ",
                "fomc": "Sentence from a U.S. Federal Reserve (FOMC) communication: ",
                "tw_topic": "Financial news tweet: "}
INSTRUCTIONS = {
    "banking77": "Which customer-service intent does this bank customer's message express?",
    "bitext": "Which customer-service intent does this customer's message express?",
    "fomc": "What monetary-policy stance does this sentence express?",
    "tw_topic": "What is the main topic of this financial news tweet?",
}
FOMC_CRITERIA = {
    "dovish": "Supports easier monetary policy: lower interest rates, more stimulus, or concern about weak growth or employment.",
    "hawkish": "Supports tighter monetary policy: higher interest rates, less stimulus, or concern about inflation or an overheating economy.",
    "neutral": "Neither hawkish nor dovish: factual or balanced, with no monetary-policy stance.",
}
LENGTH_CONTROL_SOURCE = {"banking77": ["fomc", "tw_topic"], "bitext": ["fomc", "tw_topic"],
                         "fomc": ["banking77", "bitext"], "tw_topic": ["banking77", "bitext"]}


def humanize(label: str) -> str:
    s = label.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def criteria_for(task: str, labels: list[str]) -> dict[str, str]:
    if task == "fomc":
        return dict(FOMC_CRITERIA)
    if task == "tw_topic":
        names = {slug(n): n for n in TOPIC_NAMES}
        return {l: names[l] for l in labels}
    return {l: humanize(l) for l in labels}


def dedup_pool(pool, emb_pool, emb_test):
    if len(emb_test) == 0:
        return pool, emb_pool
    sims = emb_pool.astype(np.float32) @ emb_test.astype(np.float32).T
    keep = sims.max(1) <= DEDUP_COSINE
    return [p for p, k in zip(pool, keep) if k], emb_pool[keep]


def budget_pool(pool: list[dict], b: int, task: str) -> list[int]:
    rng = random.Random(f"{SEED}-{task}-budget{b}")
    by = {}
    for i, p in enumerate(pool):
        by.setdefault(p["label"], []).append(i)
    out = []
    for lab in sorted(by):
        idx = by[lab]
        out += rng.sample(idx, min(b, len(idx)))
    return sorted(out)


def example_block(examples: list[dict]) -> str:
    lines = [f"{i}. {e['text']} -> {e['label']}" for i, e in enumerate(examples, 1)]
    return "Labelled examples:\n" + "\n".join(lines) + "\n\nItem to decide:\n"


def unlabelled_block(texts: list[str]) -> str:
    lines = [f"{i}. {t}" for i, t in enumerate(texts, 1)]
    return "Other recent messages (not to be classified):\n" + "\n".join(lines) + "\n\nItem to decide:\n"


def build_task(task: str, other_pools: dict) -> dict[str, int]:
    d = TASKS_DIR / task
    labels = json.loads((d / "labels.json").read_text())
    test, pool = read_jsonl(d / "test.jsonl"), read_jsonl(d / "train.jsonl")
    emb_test, emb_pool = np.load(d / "emb_test.npy"), np.load(d / "emb_pool.npy")
    n0 = len(pool)
    pool, emb_pool = dedup_pool(pool, emb_pool, emb_test)
    log.info("%s: pool %d -> %d after near-duplicate removal", task, n0, len(pool))
    crit = criteria_for(task, labels)
    sims = emb_test.astype(np.float32) @ emb_pool.astype(np.float32).T

    def rec(item, cond, block, meta):
        return {"id": item["id"], "task": task, "condition": cond, "text": item["text"],
                "state": block + STATE_PREFIX[task] + item["text"], "gold": item["label"],
                "instructions": INSTRUCTIONS[task], "options": labels, "criteria": crit, "meta": meta}

    out = {c: [] for c in ["k0"] + [f"k{k}_{s}" for k in K_VALUES for s in ("rand1", "rand2", "knn")] +
           ["k8_len"] + [f"k8_knn_b{b}" for b in BUDGETS]}
    budget_idx = {b: np.array(budget_pool(pool, b, task)) for b in BUDGETS}
    other = sorted({s for src in LENGTH_CONTROL_SOURCE[task] for s in other_pools[src]}, key=len)
    other_len = [len(s) for s in other]
    overhead = len(unlabelled_block([""] * 8))
    for ti, item in enumerate(test):
        out["k0"].append(rec(item, "k0", "", {"k": 0, "selection": "none"}))
        for k in K_VALUES:
            for draw in (1, 2):
                rng = random.Random(f"{SEED}-{task}-{item['id']}-rand{draw}-k{k}")
                ex = [pool[i] for i in rng.sample(range(len(pool)), k)]
                out[f"k{k}_rand{draw}"].append(rec(item, f"k{k}_rand{draw}", example_block(ex),
                                                   {"k": k, "selection": "random", "draw": draw}))
            top = np.argsort(-sims[ti])[:k][::-1]
            out[f"k{k}_knn"].append(rec(item, f"k{k}_knn", example_block([pool[i] for i in top]),
                                        {"k": k, "selection": "knn", "draw": 0}))
        target = len(out["k8_rand1"][-1]["state"]) - len(STATE_PREFIX[task]) - len(item["text"])
        per = max((target - overhead) / 8, 1.0)
        lo, hi = bisect.bisect_left(other_len, 0.85 * per), bisect.bisect_right(other_len, 1.15 * per)
        if hi - lo < 50:
            mid = bisect.bisect_left(other_len, per)
            lo, hi = max(0, mid - 25), min(len(other), mid + 25)
        cands = other[lo:hi]
        rng = random.Random(f"{SEED}-{task}-{item['id']}-len")
        best = min((rng.sample(cands, 8) for _ in range(40)),
                   key=lambda s: abs(len(unlabelled_block(s)) - target))
        out["k8_len"].append(rec(item, "k8_len", unlabelled_block(best), {"k": 8, "selection": "length_control"}))
        for b in BUDGETS:
            idx = budget_idx[b]
            top = idx[np.argsort(-sims[ti][idx])[:8]][::-1]
            out[f"k8_knn_b{b}"].append(rec(item, f"k8_knn_b{b}", example_block([pool[i] for i in top]),
                                           {"k": 8, "selection": "knn", "budget_per_class": b}))
    for cond, recs in out.items():
        write_jsonl(records_path(task, cond), recs)
    write_jsonl(d / "pool_dedup.jsonl", pool)
    np.save(d / "emb_pool_dedup.npy", emb_pool)
    return {c: len(v) for c, v in out.items()}


def main():
    other_pools = {t: [r["text"] for r in read_jsonl(TASKS_DIR / t / "train.jsonl")] for t in TASKS}
    for t in TASKS:
        counts = build_task(t, other_pools)
        log.info("%s: %d conditions x %d items", t, len(counts), counts["k0"])


if __name__ == "__main__":
    main()
