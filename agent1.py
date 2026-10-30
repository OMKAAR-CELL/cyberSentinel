import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

ID_COLS = ["record_id"]
LABEL = "label"
_ART = None


def clean_df(df, features=None, medians=None):
    """Clean a raw dataframe. Pass features/medians (from training) for val data."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    y = df.pop(LABEL) if LABEL in df.columns else None
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if features is None:
        features = [c for c in df.columns if df[c].nunique(dropna=True) > 1]
        medians = df[features].median().fillna(0).to_dict()
    df = df.reindex(columns=features)
    df = df.fillna(pd.Series(medians)).fillna(0)
    return df.astype("float32"), y, features, medians


def train(path="train.csv"):
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    raw = raw.drop_duplicates(subset=[c for c in raw.columns if c not in ID_COLS])
    X, y, features, medians = clean_df(raw)
    y = y.astype(str).str.strip().str.upper()
    print("Class counts:\n", y.value_counts(), "\n")

    le = LabelEncoder()
    yi = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, yi, test_size=0.2, stratify=yi, random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=63,
        class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print("Accuracy :", round(accuracy_score(yte, pred), 4))
    print("Macro-F1 :", round(f1_score(yte, pred, average="macro"), 4))
    print("\nClasses:", list(le.classes_))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    print("\n", classification_report(yte, pred, target_names=le.classes_, digits=4))

    joblib.dump({"model": model, "le": le, "features": features,
                 "medians": medians}, "agent1.joblib")
    print("Saved agent1.joblib")


def predict(df):
    """Returns (pred_class array, confidence array, full probability matrix)."""
    global _ART
    if _ART is None:
        _ART = joblib.load("agent1.joblib")
    X, _, _, _ = clean_df(df, _ART["features"], _ART["medians"])
    proba = _ART["model"].predict_proba(X.values)
    idx = proba.argmax(axis=1)
    return _ART["le"].inverse_transform(idx), proba.max(axis=1), proba


if __name__ == "__main__":
    train()