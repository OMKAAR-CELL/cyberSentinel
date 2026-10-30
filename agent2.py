import sys, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from agent1 import clean_df

THRESH = 0.99          # anomaly_score >= this -> is_anomalous
_ART = None


def _slog(X):
    """Signed log: tames the huge heavy tails in flow features."""
    return np.sign(X) * np.log1p(np.abs(X))


def fit(raw, features=None, medians=None, max_benign=200_000, out="agent2.joblib"):
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()
    X, y, features, medians = clean_df(raw, features, medians)
    y = y.astype(str).str.strip().str.upper()
    Xb = _slog(X[(y == "BENIGN").values].values.astype("float64"))
    if len(Xb) == 0:
        raise ValueError("No BENIGN rows found; check label names")
    if len(Xb) > max_benign:
        Xb = Xb[np.random.RandomState(42).choice(len(Xb), max_benign, replace=False)]
    mean = Xb.mean(axis=0)
    std = np.maximum(Xb.std(axis=0), 1e-3)
    Z = (Xb - mean) / std
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Z)
    ref = np.sort(-iso.score_samples(Z))      # BENIGN reference scores (higher = more odd)
    art = {"iso": iso, "mean": mean, "std": std, "ref": ref,
           "features": features, "medians": medians}
    joblib.dump(art, out)
    return art


def score(df, art=None):
    """Returns DataFrame: anomaly_score (0-1), is_anomalous, top_features (list of 3)."""
    global _ART
    if art is None:
        if _ART is None:
            _ART = joblib.load("agent2.joblib")
        art = _ART
    X, _, _, _ = clean_df(df, art["features"], art["medians"])
    Z = (_slog(X.values.astype("float64")) - art["mean"]) / art["std"]
    raw = -art["iso"].score_samples(Z)
    # anomaly_score = share of BENIGN training flows that look LESS odd than this one
    s = np.searchsorted(art["ref"], raw, side="right") / len(art["ref"])
    top = np.argsort(-np.abs(Z), axis=1)[:, :3]
    feats = art["features"]
    tf = [[f"{feats[j]}: {Z[i, j]:+.1f} sd" for j in top[i]] for i in range(len(Z))]
    return pd.DataFrame({"anomaly_score": s, "is_anomalous": s >= THRESH,
                         "top_features": tf}, index=df.index)


def _demo():
    """Synthetic test so you can check the code without the real data."""
    rng = np.random.RandomState(0)
    def make(n, label, dur, pps, syn, pkts):
        return pd.DataFrame({
            "record_id": [f"{label}{i}" for i in range(n)],
            "Flow Duration": rng.lognormal(dur, 1, n),
            "Flow Packets/s": rng.lognormal(pps, 1, n),
            "SYN Flag Count": rng.binomial(1, syn, n),
            "Total Fwd Packets": rng.poisson(pkts, n) + 1,
            "label": label})
    df = pd.concat([make(5000, "BENIGN", 10, 3, 0.1, 10),
                    make(300, "PORT_SCAN", 3, 9, 0.95, 1),
                    make(300, "DDOS", 6, 8, 0.9, 200)], ignore_index=True)
    art = fit(df, out="agent2_demo.joblib")
    res = score(df, art)
    res["label"] = df["label"]
    print(res.groupby("label")["is_anomalous"].mean().round(3))
    print(res[res.label == "PORT_SCAN"].head(3).to_string())


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        raw = pd.read_csv("train.csv")
        feats = meds = None
        if os.path.exists("agent1.joblib"):          # reuse Agent 1's exact features
            a1 = joblib.load("agent1.joblib")
            feats, meds = a1["features"], a1["medians"]
        fit(raw, feats, meds)
        res = score(raw.sample(min(50000, len(raw)), random_state=1))
        lab = raw.loc[res.index, "label"].astype(str).str.strip().str.upper()
        print("Share flagged anomalous per class:")
        print(res["is_anomalous"].groupby(lab).mean().round(3))
        