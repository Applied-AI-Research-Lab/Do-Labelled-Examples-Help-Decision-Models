import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from bp05.config import HF_MODELS, SEED, TASKS_DIR, records_path
from bp05.records import budget_pool
from bp05.utils import read_jsonl
from bp05.runners.setfit import _embed
task = "banking77"
pool = read_jsonl(TASKS_DIR / task / "pool_dedup.jsonl"); recs = read_jsonl(records_path(task, "k0"))
test = read_jsonl(TASKS_DIR / task / "test.jsonl")
print("rec text == test text:", [r["text"] for r in recs] == [t["text"] for t in test] if "text" in test[0] else "n/a",
      "| ids same order:", [r["id"] for r in recs] == [t["id"] for t in test])
tok = AutoTokenizer.from_pretrained(HF_MODELS["embed"], padding_side="left")
model = AutoModel.from_pretrained(HF_MODELS["embed"], dtype=torch.float32).to("cuda")
emb_test = _embed([r["text"] for r in recs], tok, model, "cuda")
stored = np.load(TASKS_DIR / task / "emb_test.npy").astype(np.float32)
print("cos(setfit-path test emb, stored emb) mean:", float((emb_test * stored).sum(1).mean()))
for b in (1, 4):
    idx = budget_pool(pool, b, task); y = [pool[i]["label"] for i in idx]
    e = _embed([pool[i]["text"] for i in idx], tok, model, "cuda")
    clf = LogisticRegression(max_iter=3000, C=10.0, random_state=SEED).fit(e, y)
    acc = np.mean(clf.predict(emb_test) == np.array([r["gold"] for r in recs]))
    print(f"b{b} zero-step accuracy {acc:.4f}")
