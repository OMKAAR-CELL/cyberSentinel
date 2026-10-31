"""Checks the three CSVs against the validator rules shown on the stage page.
Usage: python check_outputs.py records.csv output"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "records.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "output"
rec = pd.read_csv(path).sort_values("seq").reset_index(drop=True)
cls, rep, pol = (pd.read_csv(f"{out}/{n}.csv", dtype=str, keep_default_na=False)
                 for n in ("classification", "replay", "policies"))
errs = []
seq_of = dict(zip(rec.record_id, rec.seq))
ids = set(rec.record_id)

for name, d, cols in [("classification", cls, ["record_id", "predicted_class", "anomaly_score", "evidence_status"]),
                      ("replay", rep, ["record_id", "action", "active_policy_id"]),
                      ("policies", pol, ["policy_id", "activated_after_record_id", "scope_type", "scope_value", "expires_after_record_id"])]:
    if list(d.columns) != cols:
        errs.append(f"{name}.csv header is {list(d.columns)}, expected {cols}")
for name, d in [("classification", cls), ("replay", rep)]:
    if d.record_id.duplicated().any() or set(d.record_id) != ids:
        errs.append(f"{name}.csv must contain every hidden record_id exactly once")
if not cls.predicted_class.isin(["BENIGN", "BRUTE_FORCE", "DOS", "DDOS", "PORT_SCAN"]).all():
    errs.append("invalid predicted_class value")
if not cls.evidence_status.isin(["AGREE", "CONFLICT", "UNCERTAIN"]).all():
    errs.append("invalid evidence_status value")
sc = pd.to_numeric(cls.anomaly_score, errors="coerce")
if sc.isna().any() or ((sc < 0) | (sc > 1)).any():
    errs.append("anomaly_score must be a number from 0 to 1")
if not rep.action.isin(["ALLOW", "BLOCK"]).all():
    errs.append("invalid action value")

endpoints = set(rec.source_ip) | set(rec.destination_ip)
if pol.policy_id.duplicated().any():
    errs.append("duplicate policy_id")
for _, p in pol.iterrows():
    if p.scope_type not in ("SOURCE_IP", "DESTINATION_IP", "SRC_DST_PAIR"):
        errs.append(f"{p.policy_id}: bad scope_type")
    vals = p.scope_value.split("|") if p.scope_type == "SRC_DST_PAIR" else [p.scope_value]
    if any(v not in endpoints for v in vals):
        errs.append(f"{p.policy_id}: scope_value is not an observed endpoint")
    if p.activated_after_record_id not in ids or p.expires_after_record_id not in ids:
        errs.append(f"{p.policy_id}: activation/expiry record id is not a hidden record")
    elif seq_of[p.expires_after_record_id] < seq_of[p.activated_after_record_id]:
        errs.append(f"{p.policy_id}: expires before it activates")


def matching(r):
    out_ = []
    for _, p in pol.iterrows():
        if p.activated_after_record_id not in seq_of or p.expires_after_record_id not in seq_of:
            continue
        if not (seq_of[p.activated_after_record_id] < r.seq <= seq_of[p.expires_after_record_id]):
            continue
        key = {"SOURCE_IP": r.source_ip, "DESTINATION_IP": r.destination_ip,
               "SRC_DST_PAIR": f"{r.source_ip}|{r.destination_ip}"}.get(p.scope_type)
        if key == p.scope_value:
            out_.append(p.policy_id)
    return sorted(out_)


rp = rep.set_index("record_id")
for _, r in rec.iterrows():
    if r.record_id not in rp.index:
        continue
    m, row = matching(r), rp.loc[r.record_id]
    if row.action == "BLOCK" and (not m or row.active_policy_id != m[0]):
        errs.append(f"{r.record_id}: BLOCK must report the smallest matching active policy {m[:1]}, got {row.active_policy_id}")
    if row.action == "ALLOW" and (row.active_policy_id != "NONE" or m):
        errs.append(f"{r.record_id}: ALLOW must report NONE with no matching active policy (matches: {m})")

print("OK: all checks passed" if not errs else f"{len(errs)} problem(s):\n" + "\n".join(errs[:30]))
