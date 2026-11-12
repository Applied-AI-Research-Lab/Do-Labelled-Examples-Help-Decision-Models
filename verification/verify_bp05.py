import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RAW, REC, TASKD, TAB = ROOT / "results/main/raw", ROOT / "data/records", ROOT / "data/tasks", ROOT / "results/main/tables"
TASKS = ["banking77", "bitext", "fomc", "tw_topic"]
KS = [2, 4, 8, 16]
CONDS = ["k0"] + [f"k{k}_{s}" for k in KS for s in ("rand1", "rand2", "knn")] + ["k8_len"] + [f"k8_knn_b{b}" for b in (1, 4, 16)]
DECISION = ["jev", "pplx_api", "clef", "pplx", "qwen_readout", "gemma_readout"]
CLASSIF = ["embedlr_b1", "embedlr_b4", "embedlr_b16", "embedlr_full", "modernbert"]
PRICE = {"jev": 0.042, "pplx_api": 0.02}
GPU_USD = 2.50
report = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    report.append(s)


def jl(p):
    with open(p) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


records = {(t, c): list(jl(REC / t / f"{c}.jsonl")) for t in TASKS for c in CONDS}
data, cover = {}, []
for m in DECISION + CLASSIF:
    for t in TASKS:
        for c in (CONDS if m in DECISION else ["k0"]):
            rows = list(jl(RAW / m / t / f"{c}.jsonl"))
            ok = {}
            for r in rows:
                if not r.get("error"):
                    ok[r["id"]] = r
            errs = {r["id"] for r in rows if r.get("error")} - set(ok)
            recs = [r for r in records[(t, c)] if r["id"] in ok]
            opts = recs[0]["options"]
            P = np.array([[ok[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
            P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            y = np.array([opts.index(r["gold"]) for r in recs])
            data[(m, t, c)] = {"P": P, "y": y, "ids": [r["id"] for r in recs],
                               "tok": np.array([ok[r["id"]].get("input_tokens") or 0 for r in recs], float),
                               "lat": np.array([ok[r["id"]].get("latency_s") or 0 for r in recs], float),
                               "pred": [ok[r["id"]].get("pred") for r in recs],
                               "mass": [(ok[r["id"]].get("meta") or {}).get("mass") for r in recs],
                               "model_echo": {(ok[r["id"]].get("meta") or {}).get("model") for r in recs}}
            cover.append({"model": m, "task": t, "condition": c, "rows": len(rows), "answered": len(recs),
                          "unique_ids": len({r["id"] for r in rows}), "unresolved_errors": len(errs),
                          "records": len(records[(t, c)])})
cov = pd.DataFrame(cover)
cov.to_csv(OUT / "coverage.csv", index=False)
say("== coverage ==")
say("files:", len(cov), " all 400 answered:", bool((cov.answered == 400).all()), " unresolved errors:", int(cov.unresolved_errors.sum()),
    " duplicate rows (rows > unique ids):", int((cov.rows > cov.unique_ids).sum()), " records 400 each:", bool((cov.records == 400).all()))
say("decisions per decision system:", {m: int(cov[cov.model == m].answered.sum()) for m in DECISION})
say("decisions per classifier:", {m: int(cov[cov.model == m].answered.sum()) for m in CLASSIF})


def acc(P, y):
    return float(np.mean(P.argmax(1) == y))


def macro_f1(P, y):
    K, yh = P.shape[1], P.argmax(1)
    f = []
    for k in range(K):
        tp, fp, fn = np.sum((yh == k) & (y == k)), np.sum((yh == k) & (y != k)), np.sum((yh != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece(P, y, order=None, nb=15):
    conf, cor = P.max(1), (P.argmax(1) == y).astype(float)
    o = np.argsort(conf) if order is None else order
    n, tot = len(conf), 0.0
    sizes = [n // nb + (1 if i < n % nb else 0) for i in range(nb)]
    s = 0
    for z in sizes:
        b = o[s:s + z]
        s += z
        if z:
            tot += z / n * abs(cor[b].mean() - conf[b].mean())
    return tot


def brier(P, y):
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    return float(np.mean(((P - Y) ** 2).sum(1)))


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def risk_curve(P, y):
    o = np.argsort(-P.max(1), kind="stable")
    e = (P.argmax(1) != y).astype(float)[o]
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(e) / k


def aurc(P, y):
    return float(risk_curve(P, y)[1].mean())


def cov2(P, y):
    c, r = risk_curve(P, y)
    ok = np.where(r <= 0.02)[0]
    return float(c[ok].max()) if len(ok) else 0.0


def cost_1k(m, d):
    if m in PRICE:
        return d["tok"].mean() * PRICE[m] / 1e6 * 1000
    return d["lat"].mean() * GPU_USD / 3600 * 1000


rows = []
for (m, t, c), d in data.items():
    P, y = d["P"], d["y"]
    rows.append({"model": m, "task": t, "condition": c, "n": len(y), "accuracy": acc(P, y), "macro_f1": macro_f1(P, y),
                 "ece": ece(P, y), "brier": brier(P, y), "nll": nll(P, y), "aurc": aurc(P, y), "cov_at_2pct_risk": cov2(P, y),
                 "input_tokens_mean": d["tok"].mean(), "usd_per_1k": cost_1k(m, d)})
main = pd.DataFrame(rows)
main.to_csv(OUT / "main.csv", index=False)
orig = pd.read_csv(TAB / "main.csv")
mg = orig.merge(main, on=["model", "task", "condition"], suffixes=("_o", "_v"))
say("\n== main.csv ==  rows orig", len(orig), "recomputed", len(main), "merged", len(mg))
for col in ["n", "accuracy", "macro_f1", "ece", "brier", "nll", "aurc", "cov_at_2pct_risk", "input_tokens_mean", "usd_per_1k"]:
    say(f"  {col:18s} max|diff| = {np.max(np.abs(mg[col + '_o'] - mg[col + '_v'])):.3g}")
bad_ece = mg[np.abs(mg.ece_o - mg.ece_v) > 1e-9]
say("  ECE cells differing (tie order):", len(bad_ece))

rng_t = np.random.default_rng(12345)
tie = []
for (m, t, c), d in data.items():
    conf = d["P"].max(1)
    vals = []
    for _ in range(2000):
        o = np.lexsort((rng_t.random(len(conf)), conf))
        vals.append(ece(d["P"], d["y"], order=o))
    tie.append({"model": m, "task": t, "condition": c, "ece_min": min(vals), "ece_max": max(vals), "ece_sd": float(np.std(vals)),
                "n_unique_conf": len(np.unique(conf))})
tie = pd.DataFrame(tie).merge(orig[["model", "task", "condition", "ece"]], on=["model", "task", "condition"])
tie["within"] = (tie.ece >= tie.ece_min - 1e-12) & (tie.ece <= tie.ece_max + 1e-12)
tie["range"] = tie.ece_max - tie.ece_min
tie.to_csv(OUT / "ece_tie_range.csv", index=False)
say("  frozen ECE within random-tie range:", int(tie.within.sum()), "/", len(tie), " max range", round(tie.range.max(), 4),
    "at", tie.loc[tie.range.idxmax(), ["model", "task", "condition"]].tolist())
for m in DECISION + CLASSIF:
    say(f"    {m:14s} max tie range {tie[tie.model == m].range.max():.4f}")

cur = []
for m in DECISION:
    for t in TASKS:
        g = main[(main.model == m) & (main.task == t)].set_index("condition")
        for sel in ("random", "knn"):
            for k in [0] + KS:
                cs = ["k0"] if k == 0 else ([f"k{k}_rand1", f"k{k}_rand2"] if sel == "random" else [f"k{k}_knn"])
                cur.append({"model": m, "task": t, "selection": sel, "k": k, "accuracy": g.loc[cs, "accuracy"].mean(),
                            "ece_from_orig": orig.set_index(["model", "task", "condition"]).loc[[(m, t, x) for x in cs], "ece"].mean()})
cur = pd.DataFrame(cur)
cur.to_csv(OUT / "curves.csv", index=False)
oc = pd.read_csv(TAB / "curves.csv").merge(cur, on=["model", "task", "selection", "k"], suffixes=("_o", "_v"))
say("\n== curves.csv == rows", len(oc), " max|acc diff|", np.max(np.abs(oc.accuracy_o - oc.accuracy_v)),
    " max|ece diff| (vs orig per-cell ECE)", np.max(np.abs(oc.ece - oc.ece_from_orig)))


def paired(a, b):
    ids = sorted(set(a["ids"]) & set(b["ids"]))
    pa, pb = {i: k for k, i in enumerate(a["ids"])}, {i: k for k, i in enumerate(b["ids"])}
    ia, ib = [pa[i] for i in ids], [pb[i] for i in ids]
    return a["P"][ia], a["y"][ia], b["P"][ib], b["y"][ib], ids


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


dip = []
for m in DECISION:
    for t in TASKS:
        for k in (2, 4):
            for c in (f"k{k}_rand1", f"k{k}_knn"):
                Pa, ya, Pb, yb, ids = paired(data[(m, t, c)], data[(m, t, "k0")])
                rng = np.random.default_rng(0)
                n = len(ids)
                pt = ece(Pa, ya) - ece(Pb, yb)
                bs = np.array([ece(Pa[ix], ya[ix]) - ece(Pb[ix], yb[ix]) for ix in (rng.integers(0, n, n) for _ in range(2000))])
                dip.append({"model": m, "task": t, "condition": c, "n": n, "ece_increase": pt, "lo": np.percentile(bs, 2.5),
                            "hi": np.percentile(bs, 97.5), "p_one_sided": float(np.mean(bs <= 0))})
dip = pd.DataFrame(dip)
dip["p_holm"] = holm(dip.p_one_sided)
dip.to_csv(OUT / "dip.csv", index=False)
od = pd.read_csv(TAB / "dip.csv").merge(dip, on=["model", "task", "condition"], suffixes=("_o", "_v"))
say("\n== dip.csv == tests", len(od))
for col in ["ece_increase", "lo", "hi", "p_one_sided", "p_holm"]:
    say(f"  {col:13s} max|diff| = {np.max(np.abs(od[col + '_o'] - od[col + '_v'])):.4g}")
say("  significant after Holm (orig / recomputed):", int((od.p_holm_o < 0.05).sum()), "/", int((od.p_holm_v < 0.05).sum()),
    "  min p_holm orig", od.p_holm_o.min(), " min p_one_sided orig", od.p_one_sided_o.min())
say("  ECE increase > 0 (orig):", int((od.ece_increase_o > 0).sum()), "of", len(od))
for k in (2, 4):
    for sel in ("rand1", "knn"):
        s = od[od.condition == f"k{k}_{sel}"]
        say(f"  mean ECE change k{k}_{sel}: orig {s.ece_increase_o.mean():.4f}  recomputed {s.ece_increase_v.mean():.4f}  "
            f"(n={len(s)}, >0: {int((s.ece_increase_o > 0).sum())}; mean of main.csv diffs "
            f"{np.mean([orig.set_index(['model','task','condition']).loc[(r.model, r.task, r.condition), 'ece'] - orig.set_index(['model','task','condition']).loc[(r.model, r.task, 'k0'), 'ece'] for r in s.itertuples()]):.4f})")

ln = []
for m in DECISION:
    for t in TASKS:
        for ref in ("k0", "k8_rand1"):
            Pa, ya, Pb, yb, ids = paired(data[(m, t, "k8_len")], data[(m, t, ref)])
            dlt = (Pa.argmax(1) == ya).astype(float) - (Pb.argmax(1) == yb).astype(float)
            rng = np.random.default_rng(0)
            n = len(dlt)
            bs = np.array([dlt[rng.integers(0, n, n)].mean() for _ in range(10000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ln.append({"model": m, "task": t, "comparison": f"k8_len - {ref}", "n": n, "acc_diff": dlt.mean(), "lo": lo, "hi": hi})
ln = pd.DataFrame(ln)
ln.to_csv(OUT / "length.csv", index=False)
ol = pd.read_csv(TAB / "length.csv").merge(ln, on=["model", "task", "comparison"], suffixes=("_o", "_v"))
say("\n== length.csv == rows", len(ol), " max|diff| acc/lo/hi:",
    [float(np.max(np.abs(ol[c + "_o"] - ol[c + "_v"]))) for c in ("acc_diff", "lo", "hi")])
s0 = ol[ol.comparison == "k8_len - k0"]
say("  vs k0: CI excludes 0 for", s0[(s0.lo_o > 0) | (s0.hi_o < 0)][["model", "task", "acc_diff_o", "lo_o", "hi_o"]].values.tolist())
s1 = ol[ol.comparison == "k8_len - k8_rand1"]
say("  vs k8_rand1: CI excludes 0 for", s1[(s1.lo_o > 0) | (s1.hi_o < 0)][["model", "task", "acc_diff_o", "lo_o", "hi_o"]].values.tolist())


def logit_cluster(X, yv, groups):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (yv - p))
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    p = 1 / (1 + np.exp(-X @ b))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    sc = X * (yv - p)[:, None]
    G = pd.Series(range(len(groups))).groupby(np.asarray(groups)).apply(list)
    S = np.array([sc[ix].sum(0) for ix in G])
    meat = S.T @ S
    n, k, g = X.shape[0], X.shape[1], len(G)
    V = Hinv @ meat @ Hinv * (g / (g - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V))


from scipy.stats import norm

reg, regdata = [], {}
for m in DECISION:
    R = []
    for t in TASKS:
        for c in CONDS:
            k = 0 if c == "k0" else int(c.split("_")[0][1:])
            sel = "none" if c == "k0" else ("knn" if c.endswith("_knn") else ("random" if "rand" in c else None))
            if sel is None:
                continue
            d = data[(m, t, c)]
            cor = (d["P"].argmax(1) == d["y"]).astype(int)
            R += [(v, np.log2(k + 1), int(sel == "knn"), t, i) for v, i in zip(cor, d["ids"])]
    g = pd.DataFrame(R, columns=["y", "lk", "knn", "task", "item"])
    regdata[m] = g
    X = np.column_stack([np.ones(len(g))] + [(g.task == t).astype(float) for t in TASKS[1:]] + [g.lk, g.lk * g.knn])
    b, se = logit_cluster(X, g.y.values.astype(float), g.item.values)
    for name, j in (("lk", 4), ("lk:knn", 5)):
        z = b[j] / se[j]
        reg.append({"model": m, "term": name, "coef": b[j], "lo": b[j] - norm.ppf(0.975) * se[j], "hi": b[j] + norm.ppf(0.975) * se[j],
                    "p": 2 * norm.sf(abs(z)), "n_obs": len(g)})
reg = pd.DataFrame(reg)
reg.to_csv(OUT / "regression.csv", index=False)
orr = pd.read_csv(TAB / "regression.csv").merge(reg, on=["model", "term"], suffixes=("_o", "_v"))
say("\n== regression.csv == rows", len(orr), " n_obs per model", sorted(set(reg.n_obs)))
for col in ["coef", "lo", "hi"]:
    say(f"  {col:4s} max|diff| = {np.max(np.abs(orr[col + '_o'] - orr[col + '_v'])):.3g}")
say("  p max rel diff =", float(np.max(np.abs(orr.p_o - orr.p_v) / orr.p_o)))
try:
    import statsmodels.formula.api as smf
    mx = 0.0
    for m in DECISION:
        g = regdata[m]
        res = smf.logit("y ~ lk + lk:knn + C(task)", data=g).fit(disp=0, cov_type="cluster",
                                                                 cov_kwds={"groups": pd.factorize(g.item)[0]})
        for term in ("lk", "lk:knn"):
            mx = max(mx, abs(res.params[term] - reg[(reg.model == m) & (reg.term == term)].coef.iloc[0]))
    say("  statsmodels cross-check max|coef diff| =", mx)
except ImportError:
    say("  statsmodels not installed: cross-check skipped")

bud = []
for t in TASKS:
    for b in (1, 4, 16, "full"):
        clf = data[(f"embedlr_{'b' + str(b) if b != 'full' else 'full'}", t, "k0")]
        row = {"task": t, "budget_per_class": str(b), "classifier_acc": acc(clf["P"], clf["y"])}
        for m in DECISION:
            d = data[(m, t, f"k8_knn_b{b}" if b != "full" else "k8_knn")]
            row[f"{m}_icl_acc"] = acc(d["P"], d["y"])
        bud.append(row)
bud = pd.DataFrame(bud)
bud.to_csv(OUT / "budget.csv", index=False)
ob = pd.read_csv(TAB / "budget.csv", dtype={"budget_per_class": str})
cols = [c for c in ob.columns if c.endswith("acc")]
say("\n== budget.csv == max|diff|", float(np.max(np.abs(ob[cols].values - bud[cols].values))))
icl = bud[[f"{m}_icl_acc" for m in DECISION]]
bud["icl_min"], bud["icl_max"] = icl.min(1), icl.max(1)
say("  ICL min > classifier at budgets 1/4/16:", bool((bud[bud.budget_per_class != "full"].icl_min > bud[bud.budget_per_class != "full"].classifier_acc).all()))
say(bud[["task", "budget_per_class", "classifier_acc", "icl_min", "icl_max"]].to_string(index=False))

say("\n== run facts ==")
for m in ("jev", "pplx_api"):
    tok = sum(data[(m, t, c)]["tok"].sum() for t in TASKS for c in CONDS)
    say(f"  {m}: requests {sum(len(data[(m, t, c)]['ids']) for t in TASKS for c in CONDS)}, input tokens {tok:,.0f}, "
        f"cost ${tok * PRICE[m] / 1e6:.4f}")
gpu = {}
for m in ["clef", "pplx", "qwen_readout", "gemma_readout"] + CLASSIF:
    cs = CONDS if m in DECISION else ["k0"]
    gpu[m] = sum(data[(m, t, c)]["lat"].sum() for t in TASKS for c in cs) / 3600
say("  GPU-h by model:", {m: round(v, 3) for m, v in gpu.items()}, " total", round(sum(gpu.values()), 3))
for m in ("qwen_readout", "gemma_readout"):
    ms = np.array([x for t in TASKS for c in CONDS for x in data[(m, t, c)]["mass"]], float)
    say(f"  {m} mass: n {len(ms)}, min {ms.min():.4f}, < 0.95: {int((ms < 0.95).sum())}, min>=0.9998? {ms.min() >= 0.9998}")
for m in ("qwen_readout",):
    for t in TASKS:
        say(f"    {m} {t} mean latency k0 {data[(m, t, 'k0')]['lat'].mean():.3f}s, all conditions "
            f"{np.mean(np.concatenate([data[(m, t, c)]['lat'] for c in CONDS])):.3f}s")
agree, n_ag, accd = 0, 0, []
for t in TASKS:
    for c in CONDS:
        a, l = data[("pplx_api", t, c)], data[("pplx", t, c)]
        pa = dict(zip(a["ids"], a["P"].argmax(1)))
        pl = dict(zip(l["ids"], l["P"].argmax(1)))
        agree += sum(pa[i] == pl[i] for i in pa)
        n_ag += len(pa)
        accd.append(abs(acc(a["P"], a["y"]) - acc(l["P"], l["y"])))
say(f"  hosted vs local Perplexity agreement {agree / n_ag:.4f} (n={n_ag}); max |acc diff| per task x condition {max(accd):.4f}")

say("\n== data facts ==")
LABEL_RE = re.compile(r"^(\d+)\. (.*) -> ([^\s]+)$")


def examples(state):
    if not state.startswith("Labelled examples:\n"):
        return []
    block = state.split("\n\nItem to decide:\n")[0]
    return [LABEL_RE.match(x).group(3) for x in block.split("\n")[1:] if LABEL_RE.match(x)]


for t in TASKS:
    tr = list(jl(TASKD / t / "train.jsonl"))
    te = list(jl(TASKD / t / "test.jsonl"))
    dd = list(jl(TASKD / t / "pool_dedup.jsonl"))
    labels = json.loads((TASKD / t / "labels.json").read_text())
    ep, et = np.load(TASKD / t / "emb_pool.npy").astype(np.float32), np.load(TASKD / t / "emb_test.npy").astype(np.float32)
    keep = (ep @ et.T).max(1) <= 0.95
    overlap = len({r["text"] for r in te} & {r["text"] for r in dd})
    shares = {}
    for c in ("k8_knn", "k8_rand1", "k8_rand2", "k8_knn_b1", "k8_knn_b4", "k8_knn_b16", "k2_knn", "k16_knn"):
        s, nex = [], []
        for r in records[(t, c)]:
            ex = examples(r["state"])
            nex.append(len(ex))
            s.append(np.mean([e == r["gold"] for e in ex]))
        shares[c] = (round(float(np.mean(s)), 4), sorted(set(nex)))
    def blk(r):
        return len(r["state"]) - len(r["state"].split("\n\nItem to decide:\n")[-1])
    ratio = np.array([blk(a) / blk(b) for a, b in zip(records[(t, "k8_len")], records[(t, "k8_rand1")])])
    test_labels = pd.Series([r["label"] for r in te]).value_counts()
    say(f"  {t}: pool {len(tr)}, dedup {len(dd)} (recomputed keep {int(keep.sum())}), test {len(te)}, labels {len(labels)}, "
        f"test classes present {len(test_labels)}, exact-text overlap test/dedup {overlap}")
    say(f"     label share {shares}")
    say(f"     k8_len / k8_rand1 block-length ratio: mean {ratio.mean():.4f}, min {ratio.min():.4f}, max {ratio.max():.4f}")
    say(f"     mean tokens Jev k0 {data[('jev', t, 'k0')]['tok'].mean():.1f}, k8_knn {data[('jev', t, 'k8_knn')]['tok'].mean():.1f}, "
        f"k16_knn {data[('jev', t, 'k16_knn')]['tok'].mean():.1f}; pplx_api k0 {data[('pplx_api', t, 'k0')]['tok'].mean():.1f}, "
        f"k16_knn {data[('pplx_api', t, 'k16_knn')]['tok'].mean():.1f}")

(OUT / "verify_report.txt").write_text("\n".join(report) + "\n")
print("\nreport written to", OUT / "verify_report.txt")
