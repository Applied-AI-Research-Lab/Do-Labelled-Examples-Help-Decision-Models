from __future__ import annotations

import argparse
import hashlib
import json

from .config import SEED, TASKS_DIR, TEST_N, TRAIN_CAP
from .utils import get_logger, write_jsonl

log = get_logger("data")

BANKING_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}
BITEXT = ("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
          "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
FOMC_REPO = "gtfintechlab/fomc_communication"
TOPIC_REPO = "zeroshot/twitter-financial-news-topic"
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
    "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
    "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]


def slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("&", " and ")).strip("_")


def _hub(repo, fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset")


def load_raw(task):
    import pandas as pd
    if task == "banking77":
        tr, te = pd.read_csv(BANKING_URLS["train"]), pd.read_csv(BANKING_URLS["test"])
        tr, te = tr.rename(columns={"category": "label"}), te.rename(columns={"category": "label"})
        return tr, te, sorted(tr.label.unique()), "PolyAI BANKING77 (GitHub)", "CC BY 4.0"
    if task == "bitext":
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(_hub(*BITEXT))[["instruction", "intent"]].dropna().drop_duplicates("instruction")
        df = df.rename(columns={"instruction": "text", "intent": "label"})
        tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
        te = te[~te.text.str.lower().isin(set(tr.text.str.lower()))]
        return tr, te, sorted(df.label.unique()), BITEXT[0], "CDLA-Sharing-1.0"
    if task == "fomc":
        names = {0: "dovish", 1: "hawkish", 2: "neutral"}
        tr = pd.read_csv(_hub(FOMC_REPO, "train.csv"), encoding="utf-8-sig")
        te = pd.read_csv(_hub(FOMC_REPO, "test.csv"), encoding="utf-8-sig")
        f = lambda d: pd.DataFrame({"text": d.sentence, "label": d.label.map(names)})
        return f(tr), f(te), ["dovish", "hawkish", "neutral"], FOMC_REPO, "CC BY-NC 4.0"
    if task == "tw_topic":
        labels = [slug(n) for n in TOPIC_NAMES]
        tr, te = pd.read_csv(_hub(TOPIC_REPO, "topic_train.csv")), pd.read_csv(_hub(TOPIC_REPO, "topic_valid.csv"))
        f = lambda d: pd.DataFrame({"text": d.text, "label": d.label.map(lambda i: labels[int(i)])})
        return f(tr), f(te), labels, TOPIC_REPO, "MIT"
    raise ValueError(task)


def main():
    from sklearn.model_selection import train_test_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    manifest = []
    for task in ["banking77", "bitext", "fomc", "tw_topic"]:
        tr, te, labels, src, lic = load_raw(task)
        n_test = min(args.limit or TEST_N, len(te))
        te, _ = train_test_split(te, train_size=n_test, stratify=te.label if n_test >= len(labels) else None,
                                 random_state=SEED) if n_test < len(te) else (te, None)
        if len(tr) > TRAIN_CAP:
            tr, _ = train_test_split(tr, train_size=TRAIN_CAP, stratify=tr.label, random_state=SEED)
        if args.limit:
            tr = tr.groupby("label", group_keys=False).head(20)
        d = TASKS_DIR / task
        write_jsonl(d / "train.jsonl", [{"id": f"{task}-pool-{i:05d}", "text": t, "label": l}
                                        for i, (t, l) in enumerate(zip(tr.text, tr.label))])
        write_jsonl(d / "test.jsonl", [{"id": f"{task}-test-{i:05d}", "text": t, "label": l}
                                       for i, (t, l) in enumerate(zip(te.text, te.label))])
        (d / "labels.json").write_text(json.dumps(labels, indent=1))
        log.info("%s: pool=%d test=%d labels=%d", task, len(tr), len(te), len(labels))
        manifest.append({"task": task, "source": src, "license": lic, "pool": len(tr), "test": len(te),
                         "sha256_test": hashlib.sha256((d / "test.jsonl").read_bytes()).hexdigest()})
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
