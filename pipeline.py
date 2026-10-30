import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import agent1, agent2, agent3

VAL = "val.csv"
BATCH = 100_000


def norm(s):
    return s.astype(str).str.strip().str.upper()


def main():
    raw = pd.read_csv(VAL)
    raw.columns = raw.columns.str.strip()
    outs = []
    for i in range(0, len(raw), BATCH):
        chunk = raw.iloc[i:i + BATCH]
        a1_class, a1_conf, _ = agent1.predict(chunk)
        a2 = agent2.score(chunk)
        fused = agent3.fuse_df(a1_class, a1_conf, a2)
        out = pd.DataFrame({
            "record_id": chunk["record_id"].values if "record_id" in chunk else chunk.index,
            "a1_class": a1_class, "a1_conf": a1_conf.round(4),
            "a2_score": a2["anomaly_score"].round(4).values,
            "a2_flag": a2["is_anomalous"].values,
            "a2_top_features": a2["top_features"].apply(lambda f: "; ".join(f)).values,
        })
        for c in fused.columns:
            out[c] = fused[c].values
        if "label" in chunk:
            out.insert(1, "true_label", norm(chunk["label"]).values)
        outs.append(out)
        print(f"processed {min(i + BATCH, len(raw))}/{len(raw)}")
    res = pd.concat(outs, ignore_index=True)
    res.to_csv("results.csv", index=False)

    print("\n=== Evidence status counts ===")
    print(res["evidence_status"].value_counts())
    print("\nPolicy counts:\n", res["policy"].value_counts())
    print("\nFused class differs from Agent 1 in", int((res.a1_class != res.final_class).sum()), "rows")

    if "true_label" in res:
        y = res["true_label"]
        for name, col in [("Agent 1 alone", "a1_class"), ("Fused (final)", "final_class")]:
            print(f"\n=== {name} ===")
            print("Accuracy:", round(accuracy_score(y, res[col]), 4))
            print("Macro-F1:", round(f1_score(y, res[col], average="macro"), 4))
        print("\n", classification_report(y, res["final_class"], digits=4))
        labels = sorted(set(y) | set(res["final_class"]))
        print("Labels:", labels)
        print(confusion_matrix(y, res["final_class"], labels=labels))
        res["agree"] = (res["a1_class"] != "BENIGN") == res["a2_flag"]
        print("\nAgents agree on attack/normal:", round(res["agree"].mean() * 100, 1), "%")
        print("Accuracy when they agree   :", round((res.loc[res.agree, 'final_class'] == y[res.agree]).mean(), 4))
        print("Accuracy when they disagree:", round((res.loc[~res.agree, 'final_class'] == y[~res.agree]).mean(), 4))


if __name__ == "__main__":
    main()