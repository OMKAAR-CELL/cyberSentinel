# CyberSentinel: Multi-Agent Network Intrusion Detection

Offline traffic-replay prototype for the Jaypee Agentic AI International Summit 2026 (Track 1). It classifies network flows from CICIDS2017 into five classes, scores anomalies independently, and fuses both agents' evidence into an explainable, simulated ALLOW/BLOCK decision.

Classes: BENIGN, BRUTE_FORCE, DOS, DDOS, PORT_SCAN.

## Agents

| Agent | File | What it does |
| --- | --- | --- |
| 1. Threat Detection | `agent1.py` | LightGBM multiclass classifier (class-weighted). Outputs predicted class and confidence. |
| 2. Anomaly | `agent2.py` | Isolation Forest fit on BENIGN flows only. Outputs an anomaly score (0-1), an anomalous flag and the top 3 deviating features (z-scores). |
| 3. Evidence Fusion | `agent3.py` | Rule-based fusion of agents 1 and 2. Outputs final class, evidence status, ALLOW/BLOCK policy and a rationale string. |

`pipeline.py` runs all three agents on the validation file and prints accuracy, macro-F1 and agreement/disagreement counts.

## Fusion rules

| Agent 1 | Agent 2 | Evidence status | Action |
| --- | --- | --- | --- |
| Attack | Anomalous | corroborated | BLOCK |
| Attack, confidence >= 0.90 | Normal | classifier_only | BLOCK |
| Attack, confidence 0.60-0.90 | Normal | weak_disagreement | BLOCK |
| Attack, confidence < 0.60 | Normal | weak_disagreement | ALLOW (as BENIGN) |
| BENIGN | Anomalous | anomaly_only_review | ALLOW (BLOCK if score >= 0.999) |
| BENIGN | Normal | corroborated_benign | ALLOW |

Thresholds are constants at the top of `agent3.py`.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Place `train.csv` and `val.csv` in this folder (not included in the repo; the dataset is provided by the organizers), then:

```
python agent1.py      # train classifier, saves agent1.joblib
python agent2.py      # fit anomaly model, saves agent2.joblib
python agent3.py      # self-test of the fusion rules
python pipeline.py    # run all agents on val.csv, writes results.csv
```

## Day 1 results (114,185 validation records)

| | Accuracy | Macro-F1 |
| --- | --- | --- |
| Agent 1 alone | 1.0000 | 0.9999 |
| Fused output | 1.0000 | 0.9999 |

Evidence status counts: corroborated_benign 59,321; classifier_only 40,415; corroborated 13,768; anomaly_only_review 681.

## Known limitations

- Agent 2 detects port scans well (about 93% flagged) but largely misses DoS, DDoS and brute-force flows, which look normal to a benign-only model.
- Fusion currently trusts a confident classifier, so it did not change Agent 1's labels on validation data.
- Validation resembles the training distribution, so Level 2 results may be lower.

## Roadmap (Level 2)

- Agent 4: temporal correlation of suspicious flows
- Agent 5: adaptive defense with temporary, expiring SOURCE_IP block policies