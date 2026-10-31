"""Level 2 replay: runs all five agents over the hidden records in seq order.

Usage:  python level2.py records.csv
Writes to ./output/: classification.csv, replay.csv, policies.csv (the three uploads)
plus decision_log.csv, policy_log.csv, groups.csv (for the panel demo).
"""
import os, sys
import pandas as pd
import agent1, agent2, agent3
from agent4 import TemporalCorrelator, abs_time
from agent5 import AdaptiveDefense

# Agent 3's detailed status -> the three values the validator accepts
STATUS = {"corroborated": "AGREE", "corroborated_benign": "AGREE",
          "classifier_only": "CONFLICT", "anomaly_only_review": "CONFLICT",
          "weak_disagreement": "UNCERTAIN"}


def main(path="records.csv", out="output"):
    df = pd.read_csv(path).sort_values("seq").reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    a4, a5 = TemporalCorrelator(), AdaptiveDefense(df["record_id"])
    cls_rows, rep_rows, log_rows = [], [], []

    for i in range(len(df)):                      # strictly in replay order; nothing from later rows is used
        one = df.iloc[[i]]
        rec = one.iloc[0].to_dict()
        rec["ts"] = abs_time(rec["day"], rec["time_offset"])
        seq = int(rec["seq"])

        pid = a5.check(rec, seq)                  # 1. active policies come FIRST

        c, conf, _ = agent1.predict(one)          # 2. stateless agents (also needed for classification.csv of blocked flows)
        a2 = agent2.score(one)
        score, flag, top = float(a2["anomaly_score"].iloc[0]), bool(a2["is_anomalous"].iloc[0]), a2["top_features"].iloc[0]
        f = agent3.fuse(c[0], float(conf[0]), score, flag, top)
        status = STATUS[f["evidence_status"]]
        cls_rows.append({"record_id": rec["record_id"], "predicted_class": f["final_class"],
                         "anomaly_score": f"{min(max(score, 0.0), 1.0):.6f}", "evidence_status": status})

        if pid:                                   # blocked before analysis: no correlation, no new policy
            rep_rows.append({"record_id": rec["record_id"], "action": "BLOCK", "active_policy_id": pid})
            note, new = f"blocked by active policy {pid}; classified only", None
        else:                                     # 3. allowed -> analyse, correlate, maybe create a policy for LATER records
            rep_rows.append({"record_id": rec["record_id"], "action": "ALLOW", "active_policy_id": "NONE"})
            ev = a4.observe(rec, {"final_class": f["final_class"], "a1_conf": float(conf[0])})
            new = a5.consider(rec, seq, ev)
            note = ev["note"]
        log_rows.append({"seq": seq, "record_id": rec["record_id"], "source_ip": rec["source_ip"],
                         "destination_ip": rec["destination_ip"], "dst_port": rec["destination_port"],
                         "a1_class": c[0], "a1_conf": round(float(conf[0]), 4), "a2_score": round(score, 4),
                         "evidence_status": status, "final_class": f["final_class"],
                         "action": rep_rows[-1]["action"], "policy": rep_rows[-1]["active_policy_id"],
                         "new_policy": new["policy_id"] if new else "", "a4_evidence": note,
                         "a3_rationale": f["rationale"]})

    pd.DataFrame(cls_rows).to_csv(f"{out}/classification.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(f"{out}/replay.csv", index=False)
    a5.policies_table().to_csv(f"{out}/policies.csv", index=False)
    pd.DataFrame(log_rows).to_csv(f"{out}/decision_log.csv", index=False)
    a5.event_log().to_csv(f"{out}/policy_log.csv", index=False)
    a4.groups_table().to_csv(f"{out}/groups.csv", index=False)

    r = pd.DataFrame(rep_rows)
    print(f"records: {len(r)}   ALLOW: {(r.action == 'ALLOW').sum()}   BLOCK: {(r.action == 'BLOCK').sum()}")
    print("predicted classes:\n", pd.DataFrame(cls_rows).predicted_class.value_counts().to_string())
    print("evidence_status:\n", pd.DataFrame(cls_rows).evidence_status.value_counts().to_string())
    print(f"policies created: {len(a5.policies)}   groups: {len(a4.groups)}")
    print(a5.event_log().to_string(index=False) if a5.policies else "(no policies)")
    print(f"\nWrote files to ./{out}/ . Now run: python check_outputs.py {path} {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["records.csv"]))
