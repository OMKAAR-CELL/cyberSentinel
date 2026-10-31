"""Shows each agent's output on its own, for the panel demo.

Run AFTER level2.py:   python demo.py records.csv
Writes output/agent1_output.csv ... agent3_output.csv (one row per record) and prints
short samples for all five agents. Agents 4 and 5 are stateful, so they are shown from
the replay logs that level2.py wrote.
"""
import os, sys
import pandas as pd
import agent1, agent2, agent3
from level2 import STATUS

path = sys.argv[1] if len(sys.argv) > 1 else "records.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "output"
if not os.path.exists(f"{out}/decision_log.csv"):
    sys.exit(f"Run 'python level2.py {path}' first (it creates {out}/decision_log.csv).")

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
df = pd.read_csv(path).sort_values("seq").reset_index(drop=True)
log = pd.read_csv(f"{out}/decision_log.csv")


def show(title, table):
    print(f"\n{'=' * 8} {title} {'=' * 8}")
    print(table.to_string(index=False))


# ---- Agent 1: threat classifier (each record is judged on its own features) ----
cls, conf, _ = agent1.predict(df)
a1 = pd.DataFrame({"record_id": df.record_id, "a1_class": cls, "a1_confidence": conf.round(4)})
a1.to_csv(f"{out}/agent1_output.csv", index=False)

# ---- Agent 2: anomaly agent (independent of the labels and of Agent 1) ----
s2 = agent2.score(df)
a2 = pd.DataFrame({"record_id": df.record_id, "anomaly_score": s2["anomaly_score"].round(4).values,
                   "is_anomalous": s2["is_anomalous"].values,
                   "top_features": s2["top_features"].apply(lambda f: "; ".join(f)).values})
a2.to_csv(f"{out}/agent2_output.csv", index=False)

# ---- Agent 3: evidence fusion of Agents 1 and 2 ----
rows = [agent3.fuse(c, float(p), float(s), bool(f), t)
        for c, p, s, f, t in zip(cls, conf, s2["anomaly_score"], s2["is_anomalous"], s2["top_features"])]
a3 = pd.DataFrame(rows)
a3.insert(0, "record_id", df.record_id)
a3["official_status"] = a3["evidence_status"].map(STATUS)
a3.to_csv(f"{out}/agent3_output.csv", index=False)

# sample = first record of each predicted class (so every class is shown once)
first = a3.drop_duplicates("final_class").record_id.tolist()
pick = lambda t: t[t.record_id.isin(first)]
show("AGENT 1  Threat Detection: class + confidence", pick(a1))
show("AGENT 2  Anomaly: score + most unusual features", pick(a2))
show("AGENT 3  Evidence Fusion: final class, status, policy hint, rationale",
     pick(a3)[["record_id", "final_class", "official_status", "evidence_status", "policy", "rationale"]])
print("\nAgent 3 status counts:\n", a3["official_status"].value_counts().to_string())

# ---- Agent 4: temporal correlation (needs history, so shown from the replay) ----
g = pd.read_csv(f"{out}/groups.csv")
show("AGENT 4  Temporal Correlation: groups of related suspicious flows",
     g.drop(columns=["members"]).assign(first_member=g["members"].str.split().str[0],
                                        last_member=g["members"].str.split().str[-1]))

# ---- Agent 5: adaptive defense ----
show("AGENT 5  Adaptive Defense: policies (activation, scope, expiry, flows blocked)",
     pd.read_csv(f"{out}/policy_log.csv").drop(columns=["reason"]))
new = log[log["new_policy"].notna() & (log["new_policy"] != "")]
if len(new):
    s = int(new.iloc[0]["seq"])
    win = log[(log.seq >= s - 1) & (log.seq <= s + 3)]
    show(f"AGENT 5  First policy {new.iloc[0]['new_policy']} is created at {new.iloc[0]['record_id']} "
         f"and applies only to LATER records",
         win[["seq", "record_id", "source_ip", "final_class", "action", "policy", "new_policy"]])
print("\nReplay totals:", log["action"].value_counts().to_dict())
print(f"\nPer-agent files written to ./{out}/: agent1_output.csv, agent2_output.csv, agent3_output.csv")