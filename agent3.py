import pandas as pd

# Tunable thresholds
CONF_HIGH = 0.90          # classifier confidence treated as strong
CONF_LOW = 0.60           # below this, a lone attack call is dropped
ANOMALY_EXTREME = 0.999   # BENIGN + anomaly above this -> BLOCK

BENIGN = "BENIGN"


def fuse(a1_class, a1_conf, a2_score, a2_flag, a2_top_features):
    """Combine Agent 1 (classifier) and Agent 2 (anomaly).
    Returns dict: final_class, evidence_status, policy, rationale."""
    feats = ", ".join(a2_top_features) if len(a2_top_features) else "none"
    a1 = f"Classifier says {a1_class} ({a1_conf:.2f})"
    a2 = f"anomaly agent score {a2_score:.3f} ({'anomalous' if a2_flag else 'normal'}; top features: {feats})"
    is_attack = a1_class != BENIGN

    if is_attack and a2_flag:
        status, final, policy = "corroborated", a1_class, "BLOCK"
        why = "Both agents agree this flow is suspicious."
    elif is_attack and a1_conf >= CONF_HIGH:
        status, final, policy = "classifier_only", a1_class, "BLOCK"
        why = "Anomaly agent disagrees, but classifier confidence is high, so trust the classifier."
    elif is_attack and a1_conf >= CONF_LOW:
        status, final, policy = "weak_disagreement", a1_class, "BLOCK"
        why = "Anomaly agent disagrees and classifier confidence is moderate; keep the attack label, flag as weak."
    elif is_attack:
        status, final, policy = "weak_disagreement", BENIGN, "ALLOW"
        why = "Anomaly agent disagrees and classifier confidence is low; treat as benign."
    elif a2_flag:
        status, final = "anomaly_only_review", BENIGN
        if a2_score >= ANOMALY_EXTREME:
            policy = "BLOCK"
            why = "Classifier says benign but the flow is extremely anomalous; block and review."
        else:
            policy = "ALLOW"
            why = "Classifier says benign but the flow looks unusual; allow and flag for review."
    else:
        status, final, policy = "corroborated_benign", BENIGN, "ALLOW"
        why = "Both agents agree this flow is normal."

    return {"final_class": final, "evidence_status": status, "policy": policy,
            "rationale": f"{a1}; {a2}. {why} Final: {final}, {policy}."}


def fuse_df(a1_class, a1_conf, a2_df):
    """Batch version. a1_class and a1_conf are arrays; a2_df is the output of agent2.score()."""
    rows = [fuse(c, p, s, f, t) for c, p, s, f, t in zip(
        a1_class, a1_conf, a2_df["anomaly_score"], a2_df["is_anomalous"], a2_df["top_features"])]
    return pd.DataFrame(rows, index=a2_df.index)


if __name__ == "__main__":
    tests = [
        ("PORT_SCAN", 0.97, 1.000, True, ["SYN Flag Count: +9.4 sd", "Flow Packets/s: +6.1 sd"]),
        ("DDOS", 0.95, 0.40, False, ["Flow Duration: +1.2 sd"]),
        ("DOS", 0.75, 0.30, False, ["Fwd IAT Mean: -0.8 sd"]),
        ("BRUTE_FORCE", 0.45, 0.20, False, ["Flow Bytes/s: +0.5 sd"]),
        ("BENIGN", 0.99, 0.995, True, ["Idle Max: +3.2 sd"]),
        ("BENIGN", 0.99, 1.000, True, ["Total Fwd Packets: +20 sd"]),
        ("BENIGN", 0.99, 0.30, False, ["Flow IAT Std: +0.4 sd"]),
    ]
    for t in tests:
        r = fuse(*t)
        print(f"{t[0]:12} conf={t[1]:.2f} score={t[2]:.3f} -> {r['evidence_status']:22} {r['final_class']:12} {r['policy']}")
    print("\nExample rationale:\n", fuse(*tests[0])["rationale"])