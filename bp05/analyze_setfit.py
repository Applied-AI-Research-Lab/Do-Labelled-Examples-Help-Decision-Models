from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import metrics as M
from .config import BUDGETS, TASKS, records_path, results_dir
from .utils import get_logger, latest_ok, read_jsonl

log = get_logger("analyze_setfit")
PROMPTED = ["jev", "pplx_api", "pplx", "clef", "qwen_readout", "gemma_readout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="setfit")
    args = ap.parse_args()
    bud = pd.read_csv(results_dir("main") / "tables" / "budget.csv", dtype={"budget_per_class": str}).set_index(["task", "budget_per_class"])
    rows = []
    for t in TASKS:
        recs = read_jsonl(records_path(t, "k0"))
        opts = recs[0]["options"]
        for b in BUDGETS:
            p = results_dir(args.run_name) / "raw" / f"setfit_b{b}" / t / "k0.jsonl"
            if not p.exists():
                log.warning("missing %s", p)
                continue
            preds = latest_ok(p)
            P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs])
            y = np.array([opts.index(r["gold"]) for r in recs])
            r0 = bud.loc[(t, str(b))]
            icl = [r0[f"{m}_icl_acc"] for m in PROMPTED]
            meta = preds[recs[0]["id"]]["meta"]
            rows.append({"task": t, "budget_per_class": b, "n": len(preds), "setfit_acc": M.accuracy(P, y),
                         "setfit_ece": M.ece_equal_mass(P, y), "embedlr_acc": r0["classifier_acc"],
                         "prompted_min": min(icl), "prompted_max": max(icl),
                         "prompted_above_setfit": int(sum(a > M.accuracy(P, y) for a in icl)),
                         "train_size": meta.get("train_size"), "pairs": meta.get("pairs"),
                         "loss_first": meta.get("loss_first"), "loss_last": meta.get("loss_last"), "train_s": meta.get("train_s")})
    df = pd.DataFrame(rows)
    out = results_dir(args.run_name) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "setfit_budget.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))
    log.info("written %s", out / "setfit_budget.csv")


if __name__ == "__main__":
    main()
