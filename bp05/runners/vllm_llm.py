from __future__ import annotations

import json
import re

from ..config import (HF_MODELS, READOUT_PRUNE, READOUT_TOP_LOGPROBS, SEED, THINK, VLLM_GPU_MEM,
                      VLLM_MAX_MODEL_LEN)
from ..prompts import (READOUT_PREFILL, generative_schema, generative_user_prompt,
                       readout_user_prompt, verbalized_probs)
from ..readout import char_token_ids, score_options, vllm_backend
from ..utils import Timer, append_jsonl, chunked, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("vllm")


def build_llm(llm_key: str):
    from vllm import LLM

    kwargs = dict(model=HF_MODELS[llm_key], max_model_len=VLLM_MAX_MODEL_LEN,
                  gpu_memory_utilization=VLLM_GPU_MEM, enable_prefix_caching=True,
                  max_logprobs=READOUT_TOP_LOGPROBS, seed=SEED)
    try:
        return LLM(**kwargs, limit_mm_per_prompt={"image": 0, "video": 0})
    except TypeError:
        return LLM(**kwargs)


def chat_ids(tokenizer, user_text: str, prefill: str = "", thinking: bool = False) -> list[int]:
    kwargs = {"enable_thinking": thinking}
    if thinking:
        kwargs["reasoning_effort"] = THINK["reasoning_effort"]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, **kwargs,
    )
    return tokenizer.encode(text + prefill, add_special_tokens=False)


def run_readout(llm, model_key: str, recs: list[dict], out, chunk_size: int = 128):
    tok = llm.get_tokenizer()
    cids = char_token_ids(tok)
    backend = vllm_backend(llm, READOUT_TOP_LOGPROBS)
    for chunk in chunked(recs, chunk_size):
        prefixes = [chat_ids(tok, readout_user_prompt(r), READOUT_PREFILL) for r in chunk]
        with Timer() as t:
            probs, diags = score_options(prefixes, [len(r["options"]) for r in chunk], backend, cids,
                                         prune=READOUT_PRUNE)
        per = t.seconds / len(chunk)
        rows = []
        for r, p, d, ids in zip(chunk, probs, diags, prefixes):
            rows.append(pred_row(r, model_key, dict(zip(r["options"], p)), per, len(ids), meta=d))
        append_jsonl(out, rows)


def _sampling_params(schema: dict):
    from vllm import SamplingParams

    base = dict(temperature=0.0, max_tokens=96, seed=SEED)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**base, structured_outputs=StructuredOutputsParams(json=schema)), "structured_outputs"
    except Exception:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=schema)), "guided_decoding"
    except Exception:
        return SamplingParams(**base), "none"


def parse_generation(text: str) -> tuple[str | None, float | None]:
    try:
        obj = json.loads(text)
        return obj.get("answer"), obj.get("confidence")
    except Exception:
        a = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        c = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', text)
        return (a.group(1) if a else None), (float(c.group(1)) if c else None)


def run_generative(llm, model_key: str, recs: list[dict], out, chunk_size: int = 256):
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    for chunk in chunked(recs, chunk_size):
        prompts, params, mode = [], [], None
        for r in chunk:
            sp, mode = _sampling_params(generative_schema(r))
            prompts.append(TokensPrompt(prompt_token_ids=chat_ids(tok, generative_user_prompt(r))))
            params.append(sp)
        with Timer() as t:
            outs = llm.generate(prompts, params, use_tqdm=False)
        per = t.seconds / len(chunk)
        rows = []
        for r, o, pr in zip(chunk, outs, prompts):
            text = o.outputs[0].text
            ans, conf = parse_generation(text)
            valid = ans in r["options"] and conf is not None
            rows.append(pred_row(r, model_key, verbalized_probs(ans, conf, r["options"]), per,
                                 len(pr["prompt_token_ids"]),
                                 meta={"raw": text[:200], "valid": valid, "decoding": mode}))
        append_jsonl(out, rows)


def final_answer_text(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def run_think(llm, model_key: str, recs: list[dict], out, chunk_size: int = 256):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=THINK["temperature"], top_p=THINK["top_p"], top_k=THINK["top_k"],
                        max_tokens=THINK["max_tokens"], seed=SEED)
    for chunk in chunked(recs, chunk_size):
        prompts = [TokensPrompt(prompt_token_ids=chat_ids(tok, generative_user_prompt(r), thinking=True))
                   for r in chunk]
        with Timer() as t:
            outs = llm.generate(prompts, sp, use_tqdm=False)
        per = t.seconds / len(chunk)
        rows = []
        for r, o, pr in zip(chunk, outs, prompts):
            text = o.outputs[0].text
            ans, conf = parse_generation(final_answer_text(text))
            valid = ans in r["options"] and conf is not None
            rows.append(pred_row(r, model_key, verbalized_probs(ans, conf, r["options"]), per,
                                 len(pr["prompt_token_ids"]),
                                 meta={"answer_text": final_answer_text(text)[:200], "valid": valid,
                                       "output_tokens": len(o.outputs[0].token_ids),
                                       "finished": o.outputs[0].finish_reason}))
        append_jsonl(out, rows)


MODE_KEY = {"readout": "readout", "generative": "gen", "think": "think"}


def main():
    ap = base_parser("Run Qwen/Gemma with vLLM (readout, generative or think)")
    ap.add_argument("--llm", choices=["qwen", "gemma"], required=True)
    ap.add_argument("--mode", choices=list(MODE_KEY), required=True)
    args = ap.parse_args()
    if args.mode == "think" and args.llm != "qwen":
        raise SystemExit("think mode is planned for Qwen3.8 only")
    model_key = f"{args.llm}_{MODE_KEY[args.mode]}"
    llm = build_llm(args.llm)
    run = {"readout": run_readout, "generative": run_generative, "think": run_think}[args.mode]
    for task, cond, recs, out in jobs(args, model_key):
        run(llm, model_key, recs, out)
        log.info("done %s", out)


if __name__ == "__main__":
    main()
