from __future__ import annotations

import asyncio
import os
import random
import time

from ..config import API_PROVIDERS
from ..prompts import QUESTION_ID, jev_payload
from ..utils import append_jsonl, get_logger, normalize
from .common import base_parser, jobs, pred_row

log = get_logger("api")


def parse_response(rec: dict, data: dict) -> tuple[dict[str, float], dict]:
    ans = data["answers"][QUESTION_ID]
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        raise ValueError(f"no probabilities in answer: {ans}")
    meta = {
        "choice": ans.get("choice"), "confidence": ans.get("confidence"),
        "model_returned": data.get("model"), "usage": data.get("usage"),
    }
    return normalize(probs, rec["options"]), meta


async def _one(client, sem, rec: dict, key: str, cfg: dict, model_key: str) -> dict:
    payload = jev_payload(rec, cfg["model"])
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay = 1.0
    last_err = None
    async with sem:
        for attempt in range(cfg["max_retries"]):
            t0 = time.perf_counter()
            try:
                r = await client.post(cfg["base_url"], json=payload, headers=headers)
                lat = time.perf_counter() - t0
                if r.status_code == 200:
                    data = r.json()
                    probs, meta = parse_response(rec, data)
                    usage = data.get("usage") or {}
                    return pred_row(rec, model_key, probs, lat, usage.get("input_tokens"), meta)
                if r.status_code in (400, 401, 403, 422):
                    return pred_row(rec, model_key, {}, lat, error=f"HTTP {r.status_code}: {r.text[:300]}")
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = max(delay, float(retry_after))
            except Exception as e:
                last_err = repr(e)[:300]
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 60)
    return pred_row(rec, model_key, {}, 0.0, error=f"gave up: {last_err}")


async def run_file(recs: list[dict], out, key: str, cfg: dict, model_key: str, flush_every: int = 50):
    import httpx

    sem = asyncio.Semaphore(cfg["concurrency"])
    async with httpx.AsyncClient(timeout=cfg["timeout_s"]) as client:
        tasks = [asyncio.create_task(_one(client, sem, r, key, cfg, model_key)) for r in recs]
        buf, n_err = [], 0
        for fut in asyncio.as_completed(tasks):
            row = await fut
            n_err += bool(row["error"])
            buf.append(row)
            if len(buf) >= flush_every:
                append_jsonl(out, buf)
                buf = []
        if buf:
            append_jsonl(out, buf)
    log.info("done %s (%d errors)", out, n_err)


def main():
    ap = base_parser("Run a hosted decision-model API on records")
    ap.add_argument("--provider", choices=list(API_PROVIDERS), required=True)
    args = ap.parse_args()
    cfg = API_PROVIDERS[args.provider]
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        raise SystemExit(f"Set {cfg['api_key_env']}")
    log.info("provider=%s endpoint=%s model=%s concurrency=%d", args.provider, cfg["base_url"], cfg["model"],
             cfg["concurrency"])
    for task, cond, recs, out in jobs(args, args.provider):
        asyncio.run(run_file(recs, out, key, cfg, args.provider))


if __name__ == "__main__":
    main()
