import json, sys
from bp05.config import CONDITIONS, TASKS, results_dir

run = sys.argv[1] if len(sys.argv) > 1 else "main"
win = int(sys.argv[2]) if len(sys.argv) > 2 else 500
raw = results_dir(run) / "raw"
series = []
for t in TASKS:
    for c in CONDITIONS:
        fa, fl = raw / "pplx_api" / t / f"{c}.jsonl", raw / "pplx" / t / f"{c}.jsonl"
        if not fa.exists() or not fl.exists():
            continue
        L = {r["id"]: r["pred"] for r in map(json.loads, open(fl)) if not r.get("error")}
        series += [(f"{t}/{c}", r["pred"] == L[r["id"]]) for r in map(json.loads, open(fa))
                   if not r.get("error") and r["id"] in L]
ok = True
for i in range(0, len(series), win):
    w = series[i:i + win]
    a = sum(x for _, x in w) / len(w)
    bad = a < 0.85 and len(w) >= win // 2
    ok &= not bad
    print(f"window {i // win:3d} [{w[0][0]} .. {w[-1][0]}] n={len(w)} agreement {a:.3f}" + ("  <-- DEGRADED" if bad else ""))
print("VERIFY", "HEALTHY" if ok else "DEGRADED", f"overall {sum(x for _, x in series) / max(len(series), 1):.3f}, n={len(series)}")
sys.exit(0 if ok else 1)
