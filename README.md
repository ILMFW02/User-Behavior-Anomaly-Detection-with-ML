# UEBA & Anomaly Detection in Enterprise Environments Using Machine Learning

## 1. Project Overview

This project focuses on researching and developing a prototype **User and Entity Behavior Analytics (UEBA)** system for detecting abnormal user behavior in an enterprise environment using Machine Learning.

The overall system is designed around the following pipeline:

```text
Enterprise Logs
      ↓
Data Normalization
      ↓
User Behavior Construction
      ↓
Behavioral Baseline
      ↓
Peer Group / User Clustering
      ↓
Anomaly Detection
      ↓
Risk Scoring + Correlation
      ↓
Alert
      ↓
Evaluation + Dashboard
```

The project is currently at the **design and prototype stage**. The final dataset, feature set, algorithms, scoring weights, thresholds, and implementation details will be finalized through group discussion and experimental validation.

---

## 2. Project Objectives

The project aims not only to identify isolated anomalous data points, but to develop a pipeline that can:

- Represent individual user behavior from multiple enterprise log sources.
- Establish the normal behavior of each user (**Personal Baseline**).
- Compare a user with other users exhibiting similar behavior (**Peer-group Baseline**).
- Detect deviations from these behavioral patterns using Machine Learning.
- Combine multiple anomaly signals into a **Risk Score** rather than triggering an alert from a single anomaly.
- Identify suspicious sequences of behaviors (**Correlation / Attack Chain**).
- Generate alerts with understandable evidence and reasons for analysts.
- Evaluate the system at the **user-day, user, and incident levels**.
- Investigate whether combining personal baseline, peer groups, anomaly models, and correlation improves detection quality.

---

## 3. Research Scope

The project focuses on **insider threat and abnormal user behavior** in enterprise environments.

Potential behavior sources include:

- Authentication / Logon
- Device / USB
- HTTP / Web activity
- Email
- File access
- Other enterprise resource-related activities

The **CERT Insider Threat dataset** may be used as a primary research dataset, together with explicit ground truth or custom lab ground truth when appropriate.

---

## 4. Proposed Architecture

### Module 1 — Threat Model

Define the types of abnormal behavior and insider-threat scenarios that the system aims to detect.

### Module 2 — Log Collection

Receive data from source logs, CSV/JSON files, databases, or other sources used in the prototype.

### Module 3 — Log Normalization

Convert different log sources into a consistent representation and handle timestamp conventions, duplicates, missing fields, and user/event naming.

### Module 4 — Entity Resolution

Map different identifiers to the same user or entity.

### Module 5 — Feature Engineering

Transform raw events into behavioral features, typically at the **user-day** level.

Examples include:

- Number of logins.
- Number of off-hours logins.
- Number of PCs used.
- Number of file accesses.
- Number of external email recipients.
- Network activity or connection counts.
- USB- and file-related behavior.

### Module 6 — Behavioral Baseline

Build:

- **Personal Baseline:** the normal behavior pattern of an individual user.
- **Peer-group Baseline:** the normal behavior pattern of similar users.

Personal baseline may be represented using historical statistics and deviation measures such as z-scores.

### Module 7 — User Clustering

Represent each user using an aggregated behavioral profile and group users with similar behavioral patterns.

The intended process is:

```text
User Behavior Profile
        ↓
Handle Missing Values
        ↓
Scaling
        ↓
K-Means
        ↓
cluster_id
```

### Module 8 — Anomaly Detection

Detect abnormal user-day behavior after behavioral baselines and peer groups have been established.

Initial candidate algorithms include:

- Isolation Forest
- LOF
- OCSVM, if useful as a comparison model

One research direction is to investigate whether a separate anomaly model for each peer cluster is more appropriate than a single model for all users.

### Module 9 — Risk Scoring

Combine multiple signals into a unified Risk Score.

The conceptual design is:

```text
Personal Anomaly
        +
Peer-group Anomaly
        +
Model Anomaly
        +
Rule Severity
        +
Correlation / Attack Chain
        ↓
Risk Score
        ↓
Severity
```

The final weights and severity thresholds will be determined experimentally rather than treated as fixed values at this stage.

### Module 10 — Correlation / Insider Threat Simulation

Combine multiple behaviors over time to identify suspicious activity sequences.

Initial attack-chain hypotheses include:

```text
Unusual Login
    → Sensitive Resource Access
```

```text
Unusual Login
    → Unusual Host
    → Mass File Access
```

```text
Privilege Change
    → Sensitive Resource Access
```

```text
Unusual Host
    → File Staging / High File Access
    → Network Activity Spike
```

```text
Repeated Authentication Anomaly
    → Suspicious Activity
```

These chains are only **initial research hypotheses**. They must be aligned with the project's Threat Model and actual event schema before being used as final rules.

### Module 11 — Dashboard

The dashboard is expected to present:

- Abnormal user.
- Time of activity.
- Feature(s) contributing to the alert.
- How the behavior differs from the baseline.
- Anomaly Score.
- Risk Score.
- Severity.
- Attack Chain.
- Supporting evidence for analyst investigation.

Streamlit or Kibana may be used for the prototype.

---

## 5. Expected Data Structure

### Behavior Data

A row may represent the behavior of one user during one day:

```text
user
day
behavioral features...
```

### Baseline Information

Potentially includes:

```text
personal baseline
personal deviation / z-score
peer baseline
peer deviation
```

### Clustering Information

```text
user_id
cluster_id
```

### Anomaly Output

```text
user
day
anomaly_score
is_anomaly
```

### Risk Output

```text
user
day
anomaly_score
risk_score
severity
alert_reason
```

### Incident Ground Truth

```text
incident_id
user
attack_start
attack_end
scenario_id
technique_id
```

The exact fields will be finalized after the group agrees on the data contract between the modules.

---

## 6. Ground Truth and Evaluation

Ground truth is used as the **reference answer** for determining whether the system correctly detects malicious behavior or incidents.

Possible ground-truth sources include:

- CERT answer keys.
- Custom lab ground truth created by the project team.

Ground truth may be transformed into:

### Malicious Events

A list of events identified as malicious, including user, timestamp, and source information.

### User-day Labels

Labels indicating whether a user-day corresponds to malicious activity.

### Incidents

Time intervals and supporting evidence used for incident-level evaluation.

---

## 7. Experimental Objectives

The project is intended to **compare and validate individual system components**, rather than only run a single model.

### Experiment 1 — Isolation Forest

Evaluate the ability of Isolation Forest to identify anomalous behavior from behavioral features.

### Experiment 2 — Contamination / Threshold Sensitivity

Study how contamination settings and alert thresholds affect anomaly rate and false-positive behavior.

### Experiment 3 — Global vs Per-Cluster

Compare:

```text
One model for all users
vs.
One model for each peer cluster
```

### Experiment 4 — Personal Baseline Ablation

Compare:

```text
Raw features
vs.
Raw + Personal baseline/deviation
```

### Experiment 5 — Peer-group Ablation

Evaluate whether adding peer-group information improves detection or ranking quality.

### Experiment 6 — Correlation

Evaluate whether combining multiple behaviors into attack chains improves incident detection or reduces false positives.

### Experiment 7 — Model Comparison

Compare Isolation Forest with LOF and/or other comparison models using the same chronological folds.

### Experiment 8 — Full System

Evaluate the complete pipeline:

```text
Features
→ Personal Baseline
→ Peer Group
→ Anomaly Detection
→ Risk Scoring
→ Correlation
→ Alert
```

---

## 8. Evaluation Metrics

### User-day Level

- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- F1@K

### User Level

- Precision@K
- Recall@K
- F1@K
- MAP@K
- NDCG@K

### Incident Level

- Incident Detection Rate
- Early Warning Rate
- Median Lead Time
- Alerts per Detected Incident
- False Alerts per 100 User-Days

The objective is to evaluate not only anomaly detection, but also whether the system can **rank users for investigation and identify incidents early**.

---

## 9. Main Project Components

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ground_truth/
│
├── src/
│   ├── normalization.py
│   ├── entity_resolution.py
│   ├── features.py
│   ├── baseline.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── risk_engine.py
│   ├── correlation.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── experiments/
├── notebooks/
├── models/
├── outputs/
└── README.md
```

The exact file structure may change as the team integrates the existing codebase.

---

## 10. Current Research Status

The project should currently be treated as a **prototype and research framework**, not as a finished production system.

The following items still need to be finalized:

- Dataset and dataset version.
- Feature definitions and feature selection.
- Data contract between team members.
- Personal baseline methodology.
- Peer-group definition and clustering strategy.
- Isolation Forest configuration.
- Comparison models.
- Risk-scoring formula and weights.
- Alert thresholds.
- Correlation rules and time windows.
- Ground-truth assumptions.
- Evaluation methodology.

The implementation should remain modular so that these decisions can be changed without rewriting the entire pipeline.

---

## 11. Intended Research Outcome

The desired outcome is not simply a working anomaly detector. The project aims to produce **experimental evidence** showing:

1. Whether Machine Learning can detect meaningful deviations in enterprise user behavior.
2. Whether personal behavioral baselines improve anomaly detection.
3. Whether peer-group information provides additional detection value.
4. Whether correlation and risk scoring improve alert quality.
5. Which anomaly detection approach performs best under the project's evaluation setting.
6. Whether the resulting system can identify real or simulated insider-threat incidents early enough to be useful for security analysts.

The final implementation, model configuration, and conclusions should be driven by the experimental results rather than fixed in advance.
