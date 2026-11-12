import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RAW, REC, TASKD = ROOT / "results/main/raw", ROOT / "data/records", ROOT / "data/tasks"
SEED = 20261010
TASKS = ["banking77", "bitext", "fomc", "tw_topic"]
KS = [2, 4, 8, 16]
DEC = ["jev", "pplx_api", "clef", "pplx", "qwen_readout", "gemma_readout"]
PREFIX = {"banking77": "Customer message: ", "bitext": "Customer message: ",
          "fomc": "Sentence from a U.S. Federal Reserve (FOMC) communication: ", "tw_topic": "Financial news tweet: "}
CONDS = ["k0"] + [f"k{k}_{s}" for k in KS for s in ("rand1", "rand2", "knn")] + ["k8_len"] + [f"k8_knn_b{b}" for b in (1, 4, 16)]
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def jl(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def block(ex):
    return "Labelled examples:\n" + "\n".join(f"{i}. {e['text']} -> {e['label']}" for i, e in enumerate(ex, 1)) + "\n\nItem to decide:\n"


def budget_idx(pool, b, task):
    rng = random.Random(f"{SEED}-{task}-budget{b}")
    by = {}
    for i, p in enumerate(pool):
        by.setdefault(p["label"], []).append(i)
    out = []
    for lab in sorted(by):
        out += rng.sample(by[lab], min(b, len(by[lab])))
    return np.array(sorted(out))


share_rows = []
say("== reconstruction of example blocks ==")
for t in TASKS:
    pool = jl(TASKD / t / "pool_dedup.jsonl")
    ep = np.load(TASKD / t / "emb_pool_dedup.npy").astype(np.float32)
    et = np.load(TASKD / t / "emb_test.npy").astype(np.float32)
    test = jl(TASKD / t / "test.jsonl")
    sims = et @ ep.T
    recs = {c: jl(REC / t / f"{c}.jsonl") for c in CONDS}
    bidx = {b: budget_idx(pool, b, t) for b in (1, 4, 16)}
    mism = {c: 0 for c in CONDS if c not in ("k0", "k8_len")}
    labmism = dict(mism)
    shares = {c: [] for c in mism}
    for ti, item in enumerate(test):
        exp = {}
        for k in KS:
            for draw in (1, 2):
                rng = random.Random(f"{SEED}-{t}-{item['id']}-rand{draw}-k{k}")
                exp[f"k{k}_rand{draw}"] = [pool[i] for i in rng.sample(range(len(pool)), k)]
            exp[f"k{k}_knn"] = [pool[i] for i in np.argsort(-sims[ti])[:k][::-1]]
        for b in (1, 4, 16):
            idx = bidx[b]
            exp[f"k8_knn_b{b}"] = [pool[i] for i in idx[np.argsort(-sims[ti][idx])[:8]][::-1]]
        for c, ex in exp.items():
            r = recs[c][ti]
            assert r["id"] == item["id"]
            if r["state"] != block(ex) + PREFIX[t] + item["text"]:
                mism[c] += 1
                lab_rec = [x.rsplit(" -> ", 1)[-1] for x in r["state"].split("\n\nItem to decide:\n")[0].split("\n") if " -> " in x]
                if lab_rec != [e["label"] for e in ex]:
                    labmism[c] += 1
            shares[c].append(np.mean([e["label"] == item["label"] for e in ex]))
    say(f"  {t}: pool_dedup {len(pool)}; block mismatches {sum(mism.values())} (label-sequence mismatches {sum(labmism.values())})",
        {c: v for c, v in mism.items() if v})
    for c, v in shares.items():
        share_rows.append({"task": t, "condition": c, "label_share": float(np.mean(v)),
                           "n_shown": {"k8_knn_b1": len(bidx[1]) if t == "fomc" else 8}.get(c, None)})
    rat = []
    for a, b in zip(recs["k8_len"], recs["k8_rand1"]):
        la = len(a["state"]) - len(PREFIX[t]) - len(a["text"])
        lb = len(b["state"]) - len(PREFIX[t]) - len(b["text"])
        rat.append(la / lb)
    say(f"     k8_len block length / k8_rand1 block length: mean {np.mean(rat):.4f} min {np.min(rat):.4f} max {np.max(rat):.4f}")
    say(f"     FOMC-style check: budget pool sizes {[len(bidx[b]) for b in (1, 4, 16)]}")
sh = pd.DataFrame(share_rows)
sh.to_csv(OUT / "label_share.csv", index=False)
say(sh.pivot(index="condition", columns="task", values="label_share").round(4).to_string())

recs0 = {t: jl(REC / t / "k0.jsonl") for t in TASKS}
C, PRED, TOK, LAT, CONF = {}, {}, {}, {}, {}
for m in DEC + ["embedlr_b1", "embedlr_b4", "embedlr_b16", "embedlr_full", "modernbert"]:
    for t in TASKS:
        order = [r["id"] for r in recs0[t]]
        gold = {r["id"]: r["gold"] for r in recs0[t]}
        for c in (CONDS if m in DEC else ["k0"]):
            ok = {}
            for r in jl(RAW / m / t / f"{c}.jsonl"):
                if not r.get("error"):
                    ok[r["id"]] = r
            opts = recs0[t][0]["options"]
            PM = np.array([[ok[i]["probs"].get(o, 0.0) for o in opts] for i in order], float)
            PM = PM / np.clip(PM.sum(1, keepdims=True), 1e-12, None)
            CONF[(m, t, c)] = PM.max(1)
            pred = []
            for i in order:
                p = ok[i]["probs"]
                pred.append(max(opts, key=lambda o: (p.get(o, 0.0), -opts.index(o))))
            PRED[(m, t, c)] = np.array(pred)
            C[(m, t, c)] = np.array([pp == gold[i] for pp, i in zip(pred, order)], float)
            TOK[(m, t, c)] = np.mean([ok[i].get("input_tokens") or 0 for i in order])
            LAT[(m, t, c)] = np.array([ok[i].get("latency_s") or 0 for i in order])


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


g = []
for t in TASKS:
    for m in DEC:
        k0 = C[(m, t, "k0")]
        row = {"task": t, "model": m, "k0": k0.mean()}
        for k in KS:
            row[f"rand{k}"] = (C[(m, t, f"k{k}_rand1")].mean() + C[(m, t, f"k{k}_rand2")].mean()) / 2
            row[f"knn{k}"] = C[(m, t, f"k{k}_knn")].mean()
        row["d_knn2"], row["d_knn2_lo"], row["d_knn2_hi"] = boot(C[(m, t, "k2_knn")] - k0)
        row["d_knn16"], row["d_knn16_lo"], row["d_knn16_hi"] = boot(C[(m, t, "k16_knn")] - k0)
        rr = (C[(m, t, "k16_rand1")] + C[(m, t, "k16_rand2")]) / 2
        row["d_rand16"], row["d_rand16_lo"], row["d_rand16_hi"] = boot(rr - k0)
        row["best_knn"] = max(row[f"knn{k}"] for k in KS)
        row["best_knn_k"] = max(KS, key=lambda k: (row[f"knn{k}"], -k))
        g.append(row)
g = pd.DataFrame(g)
g.to_csv(OUT / "gains.csv", index=False)
say("\n== accuracy gains (retrieved / random vs k0) ==")
say(g[["task", "model", "k0", "knn2", "knn16", "rand16", "d_knn2", "d_knn16", "d_knn16_lo", "d_knn16_hi", "d_rand16", "d_rand16_lo", "d_rand16_hi", "best_knn", "best_knn_k"]].round(4).to_string(index=False))
for t in TASKS:
    s = g[g.task == t]
    say(f"  {t}: d_knn16 range {s.d_knn16.min():.4f}..{s.d_knn16.max():.4f}; d_knn2 range {s.d_knn2.min():.4f}..{s.d_knn2.max():.4f}; "
        f"d_rand16 range {s.d_rand16.min():.4f}..{s.d_rand16.max():.4f}; best-k gain range {(s.best_knn - s.k0).min():.4f}..{(s.best_knn - s.k0).max():.4f}; "
        f"knn16 CI excl 0: {int((s.d_knn16_lo > 0).sum())}/6; rand16 CI excl 0: {int(((s.d_rand16_lo > 0) | (s.d_rand16_hi < 0)).sum())}/6 "
        f"{s[(s.d_rand16_lo > 0) | (s.d_rand16_hi < 0)][['model', 'd_rand16']].values.tolist()}")
dm = g.copy()
dm["share_at_2"] = (dm.knn2 - dm.k0) / (dm.knn16 - dm.k0)
say("  share of the k16 retrieved gain already reached at k = 2 (tasks with gain >= 0.05):",
    dm[(dm.knn16 - dm.k0) >= 0.05].groupby("task").share_at_2.agg(["min", "median", "max"]).round(3).to_dict("index"))
say("  mean over the six systems, retrieved accuracy by k:", g.groupby("task")[["k0", "knn2", "knn4", "knn8", "knn16"]].mean().round(4).to_dict("index"))
say("  mean over the six systems, random accuracy by k:", g.groupby("task")[["k0", "rand2", "rand4", "rand8", "rand16"]].mean().round(4).to_dict("index"))

ch = []
for t in TASKS:
    for m in DEC:
        row = {"task": t, "model": m}
        for k in KS:
            row[f"flip_rand_k{k}"] = float(np.mean(PRED[(m, t, f"k{k}_rand1")] != PRED[(m, t, f"k{k}_rand2")]))
            row[f"accgap_rand_k{k}"] = abs(C[(m, t, f"k{k}_rand1")].mean() - C[(m, t, f"k{k}_rand2")].mean())
        fl = PRED[(m, t, "k8_rand1")] != PRED[(m, t, "k8_rand2")]
        cm = (CONF[(m, t, "k8_rand1")] + CONF[(m, t, "k8_rand2")]) / 2
        row["conf_flipped_k8"], row["conf_stable_k8"] = float(cm[fl].mean()), float(cm[~fl].mean())
        row["flip_len_vs_k0"] = float(np.mean(PRED[(m, t, "k8_len")] != PRED[(m, t, "k0")]))
        row["flip_knn16_vs_k0"] = float(np.mean(PRED[(m, t, "k16_knn")] != PRED[(m, t, "k0")]))
        ch.append(row)
ch = pd.DataFrame(ch)
ch.to_csv(OUT / "churn.csv", index=False)
say("\n== decision churn ==")
say(ch[["task", "model", "flip_rand_k2", "flip_rand_k8", "flip_rand_k16", "flip_len_vs_k0", "accgap_rand_k8", "accgap_rand_k16"]].round(4).to_string(index=False))
say("  flip rand1 vs rand2 range per k:", {k: (round(ch[f'flip_rand_k{k}'].min(), 4), round(ch[f'flip_rand_k{k}'].max(), 4)) for k in KS})
say("  max |acc rand1 - rand2| per k:", {k: round(ch[f'accgap_rand_k{k}'].max(), 4) for k in KS},
    " at", ch.loc[ch.accgap_rand_k16.idxmax(), ["task", "model"]].tolist())
say("  flip k8_len vs k0 range:", round(ch.flip_len_vs_k0.min(), 4), round(ch.flip_len_vs_k0.max(), 4))
say("  flips by task (mean over systems):", ch.groupby("task")[["flip_rand_k8", "flip_len_vs_k0"]].mean().round(4).to_dict("index"))

PRICE = {"jev": 0.042, "pplx_api": 0.02}
co = []
for m in DEC:
    for c in ("k0", "k8_knn", "k16_knn", "k8_len"):
        tok = np.mean([TOK[(m, t, c)] for t in TASKS])
        if m in PRICE:
            usd = np.mean([TOK[(m, t, c)] * PRICE[m] / 1e6 * 1000 for t in TASKS])
        else:
            usd = np.mean([LAT[(m, t, c)].mean() * 2.5 / 3600 * 1000 for t in TASKS])
        co.append({"model": m, "condition": c, "tokens_mean_over_tasks": tok, "usd_per_1k_mean_over_tasks": usd})
co = pd.DataFrame(co)
co.to_csv(OUT / "costs.csv", index=False)
say("\n== cost per 1,000 decisions (mean over the four tasks) ==")
say(co.pivot(index="model", columns="condition", values="usd_per_1k_mean_over_tasks").round(4).to_string())
say(co.pivot(index="model", columns="condition", values="tokens_mean_over_tasks").round(1).to_string())
pt = []
for m in DEC:
    for t in TASKS:
        pt.append({"model": m, "task": t, **{c: (TOK[(m, t, c)] * PRICE[m] / 1e6 * 1000 if m in PRICE else LAT[(m, t, c)].mean() * 2.5 / 3600 * 1000)
                                             for c in ("k0", "k8_knn", "k16_knn")},
                   **{f"tok_{c}": TOK[(m, t, c)] for c in ("k0", "k8_knn", "k16_knn")}})
pt = pd.DataFrame(pt)
pt.to_csv(OUT / "costs_by_task.csv", index=False)
say(pt.round(4).to_string(index=False))
for m in PRICE:
    s = pt[pt.model == m]
    say(f"  {m}: token ratio k16_knn/k0 per task {(s.tok_k16_knn / s.tok_k0).round(3).tolist()}")
for t in TASKS:
    allc = np.concatenate([LAT[("qwen_readout", t, c)] for c in CONDS])
    say(f"  qwen_readout {t}: mean latency all conditions {allc.mean():.3f} s; k0 {LAT[('qwen_readout', t, 'k0')].mean():.3f}; k16_knn {LAT[('qwen_readout', t, 'k16_knn')].mean():.3f}")

def cov_threshold(conf, correct, max_risk=0.02):
    best = 0.0
    for th in np.unique(conf):
        s = conf >= th
        if 1 - correct[s].mean() <= max_risk:
            best = max(best, s.mean())
    return best


cv = []
for m in DEC + ["embedlr_full", "modernbert"]:
    for t in TASKS:
        for c in (["k0", "k8_knn", "k16_knn"] if m in DEC else ["k0"]):
            cv.append({"model": m, "task": t, "condition": c, "coverage_2pct": cov_threshold(CONF[(m, t, c)], C[(m, t, c)]),
                       "n_unique_conf": len(np.unique(CONF[(m, t, c)]))})
cv = pd.DataFrame(cv)
cv.to_csv(OUT / "coverage_threshold.csv", index=False)
say("\n== automation coverage at <= 2% error, threshold policy (ties not split) ==")
say(cv.pivot_table(index=["model"], columns=["task", "condition"], values="coverage_2pct").round(4).to_string())

from collections import Counter


def shown(state, k):
    blk = state.split("\n\nItem to decide:\n")[0]
    pos, st = [], 0
    for i in range(1, k + 1):
        j = blk.find(f"\n{i}. ", st)
        assert j >= 0
        pos.append(j)
        st = j + 1
    return [blk[pos[i]:(pos[i + 1] if i + 1 < k else len(blk))].rsplit(" -> ", 1)[1].strip() for i in range(k)]


KNN_CONDS = [("k2_knn", 2), ("k8_knn", 8), ("k16_knn", 16), ("k8_knn_b1", 8), ("k8_knn_b4", 8), ("k8_knn_b16", 8)]
kv, kd = [], []
for t in TASKS:
    for c, k in KNN_CONDS:
        kk = 3 if (t == "fomc" and c == "k8_knn_b1") else k
        recs = {r["id"]: r for r in jl(REC / t / f"{c}.jsonl")}
        order = [r["id"] for r in recs0[t]]
        one, vote, vconf = [], [], []
        for i in order:
            L = shown(recs[i]["state"], kk)
            cnt = Counter(L)
            top = max(cnt.values())
            pick = next(l for l in reversed(L) if cnt[l] == top)
            one.append(L[-1] == recs[i]["gold"])
            vote.append(pick == recs[i]["gold"])
            vconf.append(top / kk)
        one, vote, vconf = np.array(one, float), np.array(vote, float), np.array(vconf)
        sysacc = {m: C[(m, t, c)].mean() for m in DEC}
        row = {"task": t, "condition": c, "k_shown": kk, "one_nn": one.mean(), "vote": vote.mean(),
               "vote_cov_2pct": cov_threshold(vconf, vote), "sys_min": min(sysacc.values()), "sys_max": max(sysacc.values())}
        nab = nbe = 0
        for m in DEC:
            d, lo, hi = boot(C[(m, t, c)] - vote)
            kd.append({"task": t, "condition": c, "model": m, "diff": d, "lo": lo, "hi": hi})
            nab += lo > 0
            nbe += hi < 0
        row["n_above"], row["n_below"] = nab, nbe
        kv.append(row)
kv, kd = pd.DataFrame(kv), pd.DataFrame(kd)
kv.to_csv(OUT / "knn_vote.csv", index=False)
kd.to_csv(OUT / "knn_diff.csv", index=False)
say("\n== nearest-label baseline (vote over the shown examples; 1-NN = most similar example) ==")
say(kv.round(4).to_string(index=False))

SF = ROOT / "results/setfit/raw"
sf_rows, sf_diff = [], []
for t in TASKS:
    order = [r["id"] for r in recs0[t]]
    gold = {r["id"]: r["gold"] for r in recs0[t]}
    opts = recs0[t][0]["options"]
    for bgt in (1, 4, 16):
        rows_ = jl(SF / f"setfit_b{bgt}" / t / "k0.jsonl")
        ok_ = {r["id"]: r for r in rows_ if not r.get("error")}
        assert len(rows_) == 400 and len(ok_) == 400 and set(ok_) == set(order), (t, bgt)
        P = np.array([[ok_[i]["probs"].get(o, 0.0) for o in opts] for i in order])
        P = P / P.sum(1, keepdims=True)
        y = np.array([opts.index(gold[i]) for i in order])
        cs_ = (P.argmax(1) == y).astype(float)
        conf_ = P.max(1)
        meta = ok_[order[0]]["meta"]
        lr_acc = C[(f"embedlr_b{bgt}", t, "k0")].mean()
        prompted = {m: C[(m, t, f"k8_knn_b{bgt}")].mean() for m in DEC}
        best_clf = max(lr_acc, cs_.mean())
        sf_rows.append({"task": t, "budget": bgt, "setfit_acc": cs_.mean(), "setfit_mean_conf": conf_.mean(), "embedlr_acc": lr_acc,
                        "best_clf": best_clf, "best_clf_is": "setfit" if cs_.mean() > lr_acc else "embedlr",
                        "prompted_min": min(prompted.values()), "prompted_max": max(prompted.values()),
                        "n_above_best": sum(v > best_clf for v in prompted.values()), "n_tie_best": sum(v == best_clf for v in prompted.values()),
                        "n_below_best": sum(v < best_clf for v in prompted.values()),
                        "train_size": meta["train_size"], "pairs": meta["pairs"], "train_s": meta["train_s"],
                        "loss_first": meta["loss_first"], "loss_last": meta["loss_last"]})
        best_vec = cs_ if cs_.mean() > lr_acc else C[(f"embedlr_b{bgt}", t, "k0")]
        for m in DEC:
            d, lo, hi = boot(C[(m, t, f"k8_knn_b{bgt}")] - cs_)
            db, lob, hib = boot(C[(m, t, f"k8_knn_b{bgt}")] - best_vec)
            sf_diff.append({"task": t, "budget": bgt, "model": m, "diff_vs_setfit": d, "lo": lo, "hi": hi,
                            "diff_vs_best": db, "lo_best": lob, "hi_best": hib})
sf_rows, sf_diff = pd.DataFrame(sf_rows), pd.DataFrame(sf_diff)
sf_rows.to_csv(OUT / "setfit.csv", index=False)
sf_diff.to_csv(OUT / "setfit_diff.csv", index=False)
say("\n== SetFit-style classifiers (contrastive fine-tuning of the same encoder on the same budget pools) ==")
say(sf_rows.round(4).to_string(index=False))
say("prompted minus SetFit, 16 per class:")
say(sf_diff[sf_diff.budget == 16].round(4).to_string(index=False))
say(f"SetFit training time total {sf_rows.train_s.sum():.1f} s")
sig = sf_diff.groupby("budget").apply(lambda g: pd.Series({"above_sig": int((g.lo_best > 0).sum()), "below_sig": int((g.hi_best < 0).sum()), "n": len(g)}))
say("prompted vs the better classifier (significant above / below):")
say(sig.to_string())

(OUT / "extras_report.txt").write_text("\n".join(lines) + "\n")
