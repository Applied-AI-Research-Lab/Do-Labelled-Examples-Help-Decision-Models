from __future__ import annotations

import sys
from pathlib import Path

from ..config import HF_MODELS
from ..prompts import pplx_row
from ..utils import Timer, append_jsonl, chunked, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("pplx")
MODEL_KEY = "pplx"


def load():
    from huggingface_hub import snapshot_download

    ckpt = Path(snapshot_download(HF_MODELS["pplx"]))
    sys.path.insert(0, str(ckpt / "source" / "src"))
    from autojev.model import DecisionModel, answer

    return DecisionModel(ckpt, device="cuda"), answer


def _to_list(p):
    try:
        return [float(x) for x in p.tolist()]
    except AttributeError:
        return [float(x) for x in p]


def run_records(state, recs: list[dict], out, batch_size: int):
    model, answer_fn = state
    n_mismatch = 0
    for chunk in chunked(recs, batch_size):
        rows_in = [pplx_row(r) for r in chunk]
        try:
            with Timer() as t:
                preds = model.predict(rows_in)
            per = t.seconds / len(chunk)
            rows = []
            for r, row_in, p in zip(chunk, rows_in, preds):
                vec = _to_list(p)
                keys = list(row_in["question"]["criteria"].keys())
                if len(vec) != len(keys):
                    raise ValueError(f"{r['id']}: {len(vec)} probabilities for {len(keys)} options")
                probs = dict(zip(keys, vec))
                lib = answer_fn(row_in["question"], p)
                lib_choice = lib.get("choice") if isinstance(lib, dict) else lib
                ours = max(probs, key=probs.get)
                if lib_choice is not None and str(lib_choice) != ours:
                    n_mismatch += 1
                rows.append(pred_row(r, MODEL_KEY, probs, per, meta={"lib_choice": str(lib_choice)}))
        except Exception as e:
            if len(chunk) == 1:
                rows = [pred_row(chunk[0], MODEL_KEY, {}, 0.0, error=repr(e)[:300])]
            else:
                log.warning("batch failed (%s); retrying one by one", repr(e)[:120])
                run_records(state, chunk, out, 1)
                continue
        append_jsonl(out, rows)
    if n_mismatch:
        log.warning("argmax vs library answer() mismatches: %d (check option ordering!)", n_mismatch)


def main():
    ap = base_parser("Run Perplexity Decider on records")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    state = load()
    for task, cond, recs, out in jobs(args, MODEL_KEY):
        run_records(state, recs, out, args.batch_size)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
