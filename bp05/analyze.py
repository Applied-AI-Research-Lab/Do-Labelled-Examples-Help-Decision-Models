from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as S
from .config import API_PROVIDERS, BOOTSTRAP_N, BUDGETS, CONDITIONS, GPU_HOUR_USD, K_VALUES, TASKS, records_path, results_dir
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")
LLM_LIKE = ["jev", "pplx_api", "clef", "clef_flash", "pplx", "qwen_readout", "gemma_readout", "mock"]


def discover(run):
    raw = results_dir(run) / "raw"
    return sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []


def load(run, model, task, cond):
    rp, pp = records_path(task, cond), results_dir(run) / "raw" / model / task / f"{cond}.jsonl"
    if not rp.exists() or not pp.exists():
        return None
    recs, preds = read_jsonl(rp), latest_ok(pp)
    n_err = sum(1 for r in iter_jsonl(pp) if r.get("error") and r["id"] not in preds)
    recs = [r for r in recs if r["id"] in preds]
    if not recs:
        return None
    opts = recs[0]["options"]
    P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    y = np.array([opts.index(r["gold"]) for r in recs])
    return {"recs": recs, "preds": preds, "P": P, "y": y, "n_err": n_err, "ids": [r["id"] for r in recs]}


def per_item_cost(model, preds):
    rows = list(preds.values())
    if model in API_PROVIDERS:
        return float(np.mean([r["input_tokens"] or 0 for r in rows])) * API_PROVIDERS[model]["usd_per_m_input"] / 1e6
    return float(np.mean([r["latency_s"] or 0 for r in rows])) * GPU_HOUR_USD / 3600


def paired(a, b):
    ids = sorted(set(a["ids"]) & set(b["ids"]))
    pa, pb = {i: k for k, i in enumerate(a["ids"])}, {i: k for k, i in enumerate(b["ids"])}
    ia, ib = [pa[i] for i in ids], [pb[i] for i in ids]
    return a["P"][ia], a["y"][ia], b["P"][ib], b["y"][ib], ids


def regression_rows(cache, model) -> pd.DataFrame:
    rows = []
    for (mm, t, c), d in cache.items():
        if mm != model:
            continue
        k = 0 if c == "k0" else int(c.split("_")[0][1:])
        sel = "none" if c == "k0" else ("knn" if c.endswith("_knn") else ("random" if "rand" in c else None))
        if sel is None:
            continue
        corr = (d["P"].argmax(1) == d["y"]).astype(int)
        rows += [{"y": int(v), "lk": np.log2(k + 1), "sel": sel, "knn": int(sel == "knn"), "task": t, "item": i}
                 for v, i in zip(corr, d["ids"])]
    return pd.DataFrame(rows)


def regression_table(cache, models) -> list[dict]:
    import statsmodels.formula.api as smf
    out = []
    for m in models:
        g = regression_rows(cache, m)
        if len(g) < 50 or g.sel.nunique() < 3:
            continue
        try:
            res = smf.logit("y ~ lk + lk:knn + C(task)", data=g).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.item)[0]})
        except Exception as e:
            log.warning("regression failed for %s: %s", m, e)
            continue
        ci = res.conf_int()
        for term in res.params.index:
            if term.startswith("lk"):
                out.append({"model": m, "term": term, "coef": res.params[term], "lo": ci.loc[term, 0],
                            "hi": ci.loc[term, 1], "p": res.pvalues[term]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    args = ap.parse_args()
    run = args.run_name
    out_t, out_f = results_dir(run) / "tables", results_dir(run) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    models = discover(run)
    log.info("models found: %s", models)

    cache, cov, main_rows = {}, [], []
    for m in models:
        for t in TASKS:
            for c in CONDITIONS:
                d = load(run, m, t, c)
                if d is None:
                    continue
                cache[(m, t, c)] = d
                cov.append({"model": m, "task": t, "condition": c, "answered": len(d["ids"]), "unresolved_errors": d["n_err"]})
                main_rows.append({"model": m, "task": t, "condition": c, **M.summary(d["P"], d["y"]),
                                  "input_tokens_mean": float(np.mean([p["input_tokens"] or 0 for p in d["preds"].values()])),
                                  "usd_per_1k": per_item_cost(m, d["preds"]) * 1000})
    main_df = pd.DataFrame(main_rows)

    curves = []
    for (m, t), g in main_df.groupby(["model", "task"]) if len(main_df) else []:
        g = g.set_index("condition")
        if "k0" not in g.index:
            continue
        for sel in ("random", "knn"):
            for k in [0] + K_VALUES:
                conds = ["k0"] if k == 0 else ([f"k{k}_rand1", f"k{k}_rand2"] if sel == "random" else [f"k{k}_knn"])
                conds = [c for c in conds if c in g.index]
                if conds:
                    curves.append({"model": m, "task": t, "selection": sel, "k": k,
                                   "accuracy": g.loc[conds, "accuracy"].mean(), "ece": g.loc[conds, "ece"].mean()})

    dip, length = [], []
    for m in models:
        for t in TASKS:
            base = cache.get((m, t, "k0"))
            if base is None:
                continue
            for k in (2, 4):
                for c in (f"k{k}_rand1", f"k{k}_knn"):
                    d = cache.get((m, t, c))
                    if d is None:
                        continue
                    Pa, ya, Pb, yb, ids = paired(d, base)
                    rng = np.random.default_rng(0)
                    point = M.ece_equal_mass(Pa, ya) - M.ece_equal_mass(Pb, yb)
                    boots = np.array([M.ece_equal_mass(Pa[ix], ya[ix]) - M.ece_equal_mass(Pb[ix], yb[ix])
                                      for ix in (rng.integers(0, len(ids), len(ids)) for _ in range(min(args.n_boot, 2000)))])
                    dip.append({"model": m, "task": t, "condition": c, "n": len(ids), "ece_increase": point,
                                "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
                                "p_one_sided": float(np.mean(boots <= 0))})
            for ref in ("k0", "k8_rand1"):
                a, b = cache.get((m, t, "k8_len")), cache.get((m, t, ref))
                if a is None or b is None:
                    continue
                Pa, ya, Pb, yb, ids = paired(a, b)
                diff, lo, hi = S.bootstrap_diff_ci(Pa.argmax(1) == ya, Pb.argmax(1) == yb, n_boot=args.n_boot)
                length.append({"model": m, "task": t, "comparison": f"k8_len - {ref}", "n": len(ids),
                               "acc_diff": diff, "lo": lo, "hi": hi})
    dip_df = pd.DataFrame(dip)
    if len(dip_df):
        dip_df["p_holm"] = S.holm(dip_df.p_one_sided.tolist())

    regression = regression_table(cache, models)

    budget = []
    for t in TASKS:
        for b in BUDGETS + ["full"]:
            clf = cache.get((f"embedlr_{'b' + str(b) if b != 'full' else 'full'}", t, "k0"))
            row = {"task": t, "budget_per_class": b,
                   "classifier_acc": M.accuracy(clf["P"], clf["y"]) if clf else np.nan}
            for m in LLM_LIKE:
                c = f"k8_knn_b{b}" if b != "full" else "k8_knn"
                d = cache.get((m, t, c))
                if d is not None:
                    row[f"{m}_icl_acc"] = M.accuracy(d["P"], d["y"])
            budget.append(row)

    eff = main_df[["model", "task", "condition", "input_tokens_mean", "usd_per_1k"]] if len(main_df) else pd.DataFrame()
    tables = {"coverage": pd.DataFrame(cov), "main": main_df, "curves": pd.DataFrame(curves), "dip": dip_df,
              "length": pd.DataFrame(length), "regression": pd.DataFrame(regression), "budget": pd.DataFrame(budget),
              "efficiency": eff}
    for name, df in tables.items():
        df.to_csv(out_t / f"{name}.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(pd.DataFrame(curves), pd.DataFrame(budget), out_f)


def make_figures(curves, budget, out_f: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if curves.empty:
        return
    for metric in ("accuracy", "ece"):
        fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 3.8), squeeze=False)
        for ax, t in zip(axes[0], TASKS):
            g = curves[curves.task == t]
            for (m, sel), gg in g.groupby(["model", "selection"]):
                ax.plot(gg.k, gg[metric], marker="o", ls="-" if sel == "knn" else "--", label=f"{m} ({sel})")
            ax.set_xscale("symlog", base=2); ax.set_title(t); ax.set_xlabel("examples in the prompt (k)"); ax.set_ylabel(metric)
        axes[0][0].legend(fontsize=5)
        fig.tight_layout(); fig.savefig(out_f / f"{metric}_vs_k.png", dpi=200); plt.close(fig)
    if not budget.empty:
        fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 3.8), squeeze=False)
        for ax, t in zip(axes[0], TASKS):
            g = budget[(budget.task == t) & (budget.budget_per_class != "full")]
            x = g.budget_per_class.astype(int)
            ax.plot(x, g.classifier_acc, "k-o", label="embedding classifier")
            for col in [c for c in g.columns if c.endswith("_icl_acc")]:
                ax.plot(x, g[col], marker=".", label=col.replace("_icl_acc", " (8 retrieved)"))
            ax.set_xscale("log", base=2); ax.set_title(t); ax.set_xlabel("labelled examples per class")
        axes[0][0].legend(fontsize=5); fig.tight_layout(); fig.savefig(out_f / "label_budget.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)


if __name__ == "__main__":
    main()
