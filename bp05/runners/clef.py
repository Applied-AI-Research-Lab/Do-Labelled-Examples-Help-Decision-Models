from __future__ import annotations

import sys

from ..config import HF_MODELS
from ..prompts import clef_record
from ..utils import Timer, append_jsonl, chunked, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("clef")


def load(variant: str):
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(HF_MODELS[variant])
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(path, device="cuda")
    model.eval()
    return model, processor, encode_record, collate_records


def predict_batch(model, processor, encode_record, collate_records, recs: list[dict]):
    import torch

    encoded = [encode_record(processor.tokenizer, clef_record(r), processor=processor) for r in recs]
    batch = collate_records(encoded, processor.tokenizer.pad_token_id, torch.device("cuda"))
    with torch.inference_mode():
        logits = model(batch)
    out = []
    for r, enc, rec_logits in zip(recs, encoded, logits):
        q = enc.questions[0]
        p = rec_logits[0].float().softmax(-1).tolist()
        probs = dict(zip(q.option_ids, p))
        if set(probs) != set(r["options"]):
            raise ValueError(f"option mismatch for {r['id']}: {sorted(probs)[:5]}...")
        out.append(probs)
    return out


def run_records(state, recs: list[dict], out, model_key: str, batch_size: int):
    model, processor, enc_fn, col_fn = state
    import torch

    recs = sorted(recs, key=lambda r: len(r["text"]) + sum(len(v) for v in r["criteria"].values()))
    for chunk in chunked(recs, batch_size):
        bs = len(chunk)
        try:
            with Timer() as t:
                probs_list = predict_batch(model, processor, enc_fn, col_fn, chunk)
            rows = [pred_row(r, model_key, p, t.seconds / bs) for r, p in zip(chunk, probs_list)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("OOM at batch %d; retrying one by one", bs)
            rows = []
            for r in chunk:
                with Timer() as t:
                    p = predict_batch(model, processor, enc_fn, col_fn, [r])[0]
                rows.append(pred_row(r, model_key, p, t.seconds))
        append_jsonl(out, rows)


def main():
    ap = base_parser("Run Clef / Clef-flash on records")
    ap.add_argument("--variant", choices=["clef", "clef_flash"], default="clef")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    state = load(args.variant)
    for task, cond, recs, out in jobs(args, args.variant):
        run_records(state, recs, out, args.variant, args.batch_size)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
