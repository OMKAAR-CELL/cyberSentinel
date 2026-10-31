"""Agent 4: Temporal Correlation.

Groups related suspicious flows by source endpoint and time window. It only ever sees
flows that were already analysed (never future records, never labels).
It also looks for scan-like fan-out: one source touching many different destination
ports on one destination inside a time window.
"""
import pandas as pd
from collections import Counter

BENIGN = "BENIGN"
GROUP_GAP_S = 300      # a flow joins the source's open group if its last suspicious flow was <= 300 s ago
SCAN_PORTS = 5         # >= 5 distinct destination ports, same source -> same destination ...
SCAN_WINDOW_S = 600    # ... inside this many seconds
HIGH_CONF = 0.90       # classifier confidence counted as high


def abs_time(day, offset):
    """Absolute seconds from `day` (YYYY-MM-DD) and `offset` (mm:ss.f or hh:mm:ss.f)."""
    secs = sum(w * float(p) for w, p in zip((1, 60, 3600), reversed(str(offset).split(":"))))
    days = (pd.Timestamp(day) - pd.Timestamp("2017-01-01")).days
    return days * 86400 + secs


class TemporalCorrelator:
    def __init__(self):
        self.groups = {}   # group_id -> dict
        self.open = {}     # source_ip -> group_id of its latest group
        self.seen = []     # (source, destination, port, ts) of every flow analysed so far

    def observe(self, rec, verdict):
        """rec: record dict (needs record_id, source_ip, destination_ip, destination_port, ts).
        verdict: dict with final_class, a1_conf from Agents 1-3. Returns the evidence dict."""
        src, dst, port, ts = rec["source_ip"], rec["destination_ip"], int(rec["destination_port"]), rec["ts"]
        ports = {p for s, d, p, t in self.seen if s == src and d == dst and 0 <= ts - t <= SCAN_WINDOW_S} | {port}
        self.seen.append((src, dst, port, ts))
        scan = len(ports) >= SCAN_PORTS
        attack = verdict["final_class"] != BENIGN
        ev = {"group_id": "", "group_size": 0, "n_attack": 0, "n_high_conf": 0,
              "distinct_ports": len(ports), "scan": scan, "suspicious": attack or scan, "note": "not suspicious"}
        if not ev["suspicious"]:
            return ev

        gid = self.open.get(src)
        if gid is None or ts - self.groups[gid]["last_ts"] > GROUP_GAP_S:
            gid = f"G{len(self.groups) + 1:03d}"
            self.groups[gid] = {"group_id": gid, "source_ip": src, "members": [], "classes": Counter(),
                                "destinations": set(), "ports": set(), "n_attack": 0, "n_high_conf": 0,
                                "first_ts": ts, "last_ts": ts, "scan": False}
            self.open[src] = gid
        g = self.groups[gid]
        g["members"].append(rec["record_id"])
        g["classes"][str(verdict["final_class"])] += 1
        g["destinations"].add(dst)
        g["ports"].add(port)
        g["last_ts"] = ts
        g["scan"] = g["scan"] or scan
        if attack:
            g["n_attack"] += 1
            if verdict["a1_conf"] >= HIGH_CONF:
                g["n_high_conf"] += 1

        ev.update(group_id=gid, group_size=len(g["members"]), n_attack=g["n_attack"], n_high_conf=g["n_high_conf"])
        ev["note"] = (f"{src}: {len(g['members'])} suspicious flows in {gid} over {g['last_ts'] - g['first_ts']:.0f}s, "
                      f"classes {dict(g['classes'])}, {len(g['ports'])} destination ports"
                      + (", scan-like fan-out" if g["scan"] else ""))
        return ev

    def groups_table(self):
        rows = []
        for g in self.groups.values():
            rows.append({"group_id": g["group_id"], "source_ip": g["source_ip"],
                         "n_flows": len(g["members"]), "classes": dict(g["classes"]),
                         "destinations": "|".join(sorted(g["destinations"])),
                         "n_dest_ports": len(g["ports"]), "scan_like": g["scan"],
                         "span_s": round(g["last_ts"] - g["first_ts"], 1),
                         "members": " ".join(g["members"])})
        return pd.DataFrame(rows)
