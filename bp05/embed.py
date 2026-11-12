from __future__ import annotations

import numpy as np

from .config import HF_MODELS, TASKS, TASKS_DIR
from .utils import get_logger, read_jsonl

log = get_logger("embed")


def load_model():
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer
    device = os.environ.get("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(HF_MODELS["embed"], padding_side="left")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(HF_MODELS["embed"], dtype=dtype).to(device).eval()
    return tok, model, device


def embed_texts(texts: list[str], tok, model, device, batch_size: int = 64, max_len: int = 256) -> np.ndarray:
    import torch
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state[:, -1]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main():
    tok, model, device = load_model()
    for task in TASKS:
        d = TASKS_DIR / task
        for part in ("train", "test"):
            texts = [r["text"] for r in read_jsonl(d / f"{part}.jsonl")]
            emb = embed_texts(texts, tok, model, device)
            np.save(d / f"emb_{'pool' if part == 'train' else 'test'}.npy", emb)
            log.info("%s %s: %s", task, part, emb.shape)


if __name__ == "__main__":
    main()
