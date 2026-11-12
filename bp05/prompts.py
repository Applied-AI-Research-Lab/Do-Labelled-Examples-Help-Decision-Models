from __future__ import annotations

QUESTION_ID = "label"
READOUT_PREFILL = "Best answer: ["


def state_text(rec: dict) -> str:
    return rec["state"]


def systemone_question(rec: dict) -> dict:
    return {"type": "choice", "instructions": rec["instructions"], "criteria": dict(rec["criteria"])}


def jev_payload(rec: dict, model: str) -> dict:
    return {"model": model, "state": state_text(rec), "questions": {QUESTION_ID: systemone_question(rec)}}


def clef_record(rec: dict) -> dict:
    return {"state": state_text(rec), "questions": {QUESTION_ID: systemone_question(rec)}}


def pplx_row(rec: dict) -> dict:
    return {"state": state_text(rec), "question": systemone_question(rec)}


def readout_user_prompt(rec: dict) -> str:
    lines = [f"[{i}] {opt}: {rec['criteria'][opt]}" for i, opt in enumerate(rec["options"], start=1)]
    return (
        f"{state_text(rec)}\n\n"
        f"Question: {rec['instructions']}\n\n"
        "Options:\n" + "\n".join(lines) + "\n\n"
        "Reply with the number of the single best option."
    )


def generative_user_prompt(rec: dict) -> str:
    lines = [f"- {opt}: {rec['criteria'][opt]}" for opt in rec["options"]]
    return (
        f"{state_text(rec)}\n\n"
        f"Question: {rec['instructions']}\n\n"
        "Options (use the option id before the colon):\n" + "\n".join(lines) + "\n\n"
        'Respond with JSON only: {"answer": "<option id>", "confidence": <integer 0-100, '
        "your probability that the answer is correct>}"
    )


def generative_schema(rec: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list(rec["options"])},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }


def verbalized_probs(answer: str | None, confidence: float | None, options: list[str]) -> dict[str, float]:
    k = len(options)
    if answer not in options or confidence is None:
        return {o: 1.0 / k for o in options}
    c = min(max(float(confidence) / 100.0, 0.0), 1.0)
    rest = (1.0 - c) / (k - 1) if k > 1 else 0.0
    return {o: (c if o == answer else rest) for o in options}
