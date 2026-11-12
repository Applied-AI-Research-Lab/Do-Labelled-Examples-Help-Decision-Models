from __future__ import annotations

import math
from typing import Callable

NextLogprobFn = Callable[[list[list[int]]], list[dict[int, float]]]
DIGITS = "0123456789"
CLOSE = "]"


def label_strings(k: int) -> list[str]:
    return [str(i) for i in range(1, k + 1)]


def _prefix_sets(k: int) -> tuple[set[str], set[str]]:
    labels = set(label_strings(k))
    prefixes = {lab[:j] for lab in labels for j in range(1, len(lab) + 1)}
    return labels, prefixes


def char_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for ch in DIGITS + CLOSE:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"character {ch!r} is not a single token: {enc}")
        ids[ch] = enc[0]
    probe = tokenizer.encode("Best answer: [12]", add_special_tokens=False)
    if probe[-3:] != [ids["1"], ids["2"], ids[CLOSE]]:
        raise ValueError(f"'[12]' does not tokenize as separate digit tokens: {probe[-5:]}")
    return ids


def score_options(
    prefix_ids: list[list[int]],
    n_options: list[int],
    next_logprobs: NextLogprobFn,
    cids: dict[str, int],
    prune: float = 1e-5,
) -> tuple[list[list[float]], list[dict]]:
    n = len(prefix_ids)
    sets = [_prefix_sets(k) for k in n_options]
    label_lp: list[dict[str, float]] = [dict() for _ in range(n)]
    calls = [0] * n
    frontier = [(i, "", 0.0) for i in range(n)]
    while frontier:
        batch = [prefix_ids[i] + [cids[c] for c in s] for (i, s, _) in frontier]
        outs = next_logprobs(batch)
        nxt = []
        for (i, s, lp), out in zip(frontier, outs):
            calls[i] += 1
            labels, prefixes = sets[i]
            if s in labels:
                close_lp = out.get(cids[CLOSE])
                if close_lp is not None:
                    label_lp[i][s] = lp + close_lp
            for d in DIGITS:
                child = s + d
                if child not in prefixes:
                    continue
                d_lp = out.get(cids[d])
                if d_lp is None:
                    continue
                new_lp = lp + d_lp
                if math.exp(new_lp) >= prune:
                    nxt.append((i, child, new_lp))
        frontier = nxt

    probs, diags = [], []
    for i in range(n):
        k = n_options[i]
        raw = [math.exp(label_lp[i].get(lab, float("-inf"))) for lab in label_strings(k)]
        mass = sum(raw)
        p = [r / mass for r in raw] if mass > 0 else [1.0 / k] * k
        probs.append(p)
        diags.append({"mass": mass, "calls": calls[i]})
    return probs, diags


def vllm_backend(llm, top_n: int) -> NextLogprobFn:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_n)

    def fn(batch: list[list[int]]) -> list[dict[int, float]]:
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in batch], sp, use_tqdm=False)
        return [{tid: lp.logprob for tid, lp in o.outputs[0].logprobs[0].items()} for o in outs]

    return fn


def hf_backend(model, candidate_ids: list[int], device: str) -> NextLogprobFn:
    import torch

    def fn(batch: list[list[int]]) -> list[dict[int, float]]:
        res = []
        for ids in batch:
            with torch.inference_mode():
                logits = model(torch.tensor([ids], device=device)).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            res.append({t: lp[t].item() for t in candidate_ids})
        return res

    return fn
