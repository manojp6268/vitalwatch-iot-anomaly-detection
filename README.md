# VitalWatch - IoT ECG Anomaly Detection System

> *Every heartbeat tells a story. VitalWatch listens.*

ML-powered anomaly detection on simulated wearable ECG data - built at the 
intersection of IoT, Healthcare, and Machine Learning.

---

## The Problem

A wearable device continuously monitors a patient's heart rate.
Most beats are normal. A few are not.

The challenge: abnormal cardiac events are rare - sometimes less than 2% of 
all beats. A model that predicts "Normal" for everything achieves 98% accuracy 
and misses every dangerous event. Traditional ML fails here.

VitalWatch is built around this specific challenge.

---

## Dataset

**MIT-BIH Arrhythmia Database** - the gold standard for cardiac ML research.
- Patient Record: 100 (69M, medicated)
- Sampling Rate: 360 Hz
- Duration: ~30 minutes (~650,000 samples)
- Total beats annotated: 2,274
- Abnormal beats: 35 (~1.5%)
- Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

Streamed directly via `wfdb` - no manual download required.

---

## Approach

### Features Engineered (6 total)
| Feature | Description |
|---|---|
| `rr_interval` | Time between consecutive heartbeats (ms) |
| `heart_rate` | Beats per minute derived from RR interval |
| `rr_diff` | Beat-to-beat change in RR interval |
| `rolling_mean_rr` | Local rhythm baseline (5-beat window) |
| `rolling_std_rr` | Local rhythm variability (5-beat window) |
| `relative_rr` | Current RR normalised against local baseline |

### Model — Isolation Forest
Chosen for three specific reasons:
1. **Unsupervised** - no labels needed, matches real IoT deployment
2. **Imbalance-resistant** - learns normal, flags deviation
3. **Interpretable** - every flag traceable to a specific feature value

### IoT Simulation
Model trained on first 80% of beats (historical).
Last 20% fed one beat at a time - simulating a live wearable stream.
Scaler fitted on training data only - no data leakage.

---

## Results

| Phase | Features | Detection Rate | False Positives |
|---|---|---|---|
| Phase 3 - Baseline | 3 | 44.1% | 20 |
| Phase 4 - Enhanced | 6 | 52.9% | 17 |
| Phase 5 - Live Stream | 6 | ~37.5% precision | 5 |

### Model Comparison
| Model | Flagged | Caught | False+ | Detection Rate |
|---|---|---|---|---|
| Isolation Forest | 35 | 18 | 17 | 52.9% |
| DBSCAN (default) | 195 | 24 | 171 | 70.6% |
| DBSCAN (tuned) | 70 | 6 | 64 | 17.6% |

DBSCAN's sensitivity to epsilon makes it architecturally unsuitable 
for heavily imbalanced cardiac data. Isolation Forest is the right 
tool for this specific problem.

---

## Project Structure

```
vitalwatch-iot-anomaly-detection/

│

├── phase1_data_exploration.ipynb          # EDA, signal loading, annotation analysis

├── phase2_feature_engineering.ipynb       # RR intervals, heart rate, rolling features

├── phase3_anomaly_detection.ipynb         # Baseline Isolation Forest (3 features)

├── phase4_model_improvement.ipynb         # Enhanced model (6 features)

├── phase4_model_comparison.ipynb         # DBSCAN vs Isolation Forest

├── phase5_iot_simulation.ipynb            # Live stream simulation + visualisation

│

├── vitalwatch_final.png               # Final three-panel clinical visualisation

└── README.md

---
```

## Setup

```bash
# Create environment
conda create -n vitalwatch python=3.10
conda activate vitalwatch

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter wfdb neurokit2

# Launch
jupyter notebook
```

Run notebooks in order 01 -> 05.

---

## Key Learnings

- Class imbalance in healthcare is extreme - standard accuracy is meaningless
- Model selection is problem-specific, not reputation-based
- Data leakage is the most common reason models fail in production
- Interpretability matters as much as performance in clinical systems
- Rolling context features (mean, std, relative) are more powerful 
  than absolute values alone for rhythm anomaly detection

---

## What's Next - V2 Roadmap

- [ ] Autoencoder trained on normal beats only - reconstruction error as anomaly score
- [ ] Multi-patient training across 10+ MIT-BIH records
- [ ] Precision-recall curves and F1 evaluation
- [ ] Real-time dashboard visualisation

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · WFDB · Jupyter · Anaconda

---

*Built from scratch as part of a self-directed learning journey in 
Healthcare ML and IoT systems.*
