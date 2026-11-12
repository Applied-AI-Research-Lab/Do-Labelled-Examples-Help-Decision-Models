from __future__ import annotations

import argparse
import zlib

from ..config import TASKS, conditions_for, raw_path, records_path
from ..utils import done_ids, get_logger, read_jsonl

log = get_logger("runner")


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--tasks", default="all", help="comma list or 'all'")
    ap.add_argument("--conditions", default="all", help="comma list or 'all' (per-model defaults)")
    ap.add_argument("--limit", type=int, default=None, help="first N records per file (pilot/preflight)")
    ap.add_argument("--run-name", default="main", help="results/<run-name>/ (e.g. pilot, preflight)")
    ap.add_argument("--shard", default="0/1",
                    help="k/n or k1,k2,.../n: only ids with crc32(id) %% n in {k...}. Shards of one model can run "
                         "on several GPUs at once and share the output file (appends are single atomic writes)")
    return ap


def in_shard(rec_id: str, shard: str) -> bool:
    ks, n = shard.split("/")
    return zlib.crc32(rec_id.encode()) % int(n) in {int(k) for k in ks.split(",")}


def jobs(args, model: str):
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    for task in tasks:
        conds = conditions_for(task, model)
        if args.conditions != "all":
            conds = [c for c in args.conditions.split(",") if c in conds]
        for cond in conds:
            rp = records_path(task, cond)
            if not rp.exists():
                log.warning("missing records %s (run python -m bp05.records)", rp)
                continue
            recs = read_jsonl(rp)
            if args.limit:
                recs = recs[: args.limit]
            out = raw_path(args.run_name, model, task, cond)
            done = done_ids(out)
            pending = [r for r in recs if r["id"] not in done and in_shard(r["id"], args.shard)]
            log.info("%s %s/%s: %d records, %d pending (shard %s) -> %s", model, task, cond, len(recs), len(pending),
                     args.shard, out)
            if pending:
                yield task, cond, pending, out


def pred_row(rec: dict, model: str, probs: dict[str, float], latency_s: float,
             input_tokens: int | None = None, meta: dict | None = None, error: str | None = None) -> dict:
    pred = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
    return {
        "id": rec["id"], "task": rec["task"], "condition": rec["condition"], "model": model,
        "probs": probs, "pred": pred, "latency_s": latency_s, "input_tokens": input_tokens,
        "meta": meta or {}, "error": error,
    }
