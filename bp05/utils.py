from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    if not data:
        return
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for r in iter_jsonl(path):
        if not r.get("error"):
            out.add(r["id"])
    return out


def latest_ok(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for r in iter_jsonl(path):
            if not r.get("error"):
                out[r["id"]] = r
    return out


def chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def normalize(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    vals = [max(float(probs.get(o, 0.0)), 0.0) for o in options]
    s = sum(vals)
    if s <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: v / s for o, v in zip(options, vals)}


def argmax_label(probs: dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0]
