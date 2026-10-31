"""Agent 5: Adaptive Defense.

Creates temporary simulated BLOCK policies from Agent 4's evidence. A policy created
while handling record N becomes active for records AFTER N and stays active through its
expiry record (inclusive). Policies are checked BEFORE a newly arriving flow is analysed.
"""
import pandas as pd

MIN_ATTACK_FLOWS = 2     # attack-classified flows from one source inside one group ...
MIN_HIGH_CONF = 2        # ... of which at least this many are high-confidence
TTL_RECORDS = 100         # a policy stays active for this many records after it is created


class AdaptiveDefense:
    def __init__(self, record_ids):
        self.ids = list(record_ids)      # record ids in replay order
        self.policies = []
        self.hits = {}                   # policy_id -> number of flows it blocked

    def _applies(self, p, rec, seq):
        if not (p["activated_seq"] < seq <= p["expires_seq"]):
            return False
        t, v = p["scope_type"], p["scope_value"]
        if t == "SOURCE_IP":
            return rec["source_ip"] == v
        if t == "DESTINATION_IP":
            return rec["destination_ip"] == v
        return f'{rec["source_ip"]}|{rec["destination_ip"]}' == v

    def check(self, rec, seq):
        """Policies come first. Returns the lexicographically smallest matching active policy_id, or None."""
        m = sorted(p["policy_id"] for p in self.policies if self._applies(p, rec, seq))
        if m:
            self.hits[m[0]] = self.hits.get(m[0], 0) + 1
            return m[0]
        return None

    def consider(self, rec, seq, ev):
        """Called only for flows that were ALLOWED. May create a policy affecting LATER records."""
        if seq >= len(self.ids):
            return None
        src = rec["source_ip"]
        reasons = []
        if ev["n_attack"] >= MIN_ATTACK_FLOWS and ev["n_high_conf"] >= MIN_HIGH_CONF:
            reasons.append(f"{ev['n_attack']} attack-classified flows ({ev['n_high_conf']} high-confidence) "
                           f"from {src} in group {ev['group_id']}")
        if ev["scan"]:
            reasons.append(f"scan-like fan-out: {ev['distinct_ports']} destination ports from {src}")
        if not reasons:
            return None
        pid = f"P{len(self.policies) + 1:04d}"
        exp_seq = min(seq + TTL_RECORDS, len(self.ids))
        p = {"policy_id": pid, "activated_after_record_id": rec["record_id"], "scope_type": "SOURCE_IP",
             "scope_value": src, "expires_after_record_id": self.ids[exp_seq - 1],
             "activated_seq": seq, "expires_seq": exp_seq, "reason": "; ".join(reasons)}
        self.policies.append(p)
        return p

    def policies_table(self):
        cols = ["policy_id", "activated_after_record_id", "scope_type", "scope_value", "expires_after_record_id"]
        return pd.DataFrame([{c: p[c] for c in cols} for p in self.policies], columns=cols)

    def event_log(self):
        rows = []
        for p in self.policies:
            rows.append({"policy_id": p["policy_id"], "scope": f'{p["scope_type"]}={p["scope_value"]}',
                         "active_from_record": self.ids[p["activated_seq"]],
                         "active_through_record": p["expires_after_record_id"],
                         "flows_blocked": self.hits.get(p["policy_id"], 0), "reason": p["reason"]})
        return pd.DataFrame(rows)
