# Phase 1: ECG Basics and Data Understanding

This phase focused on understanding the data itself: its structure, format, and clinical meaning.

## What was learned

- ECG data is not standard tabular data. It comes in special binary formats (`212-bit` compression used by MIT-BIH)
- The `wfdb` library is necessary because raw ECG signals cannot be easily loaded with pandas
- Beat annotations matter—they reveal where each heartbeat occurs and what type it is
- The signal must be paired with annotations to extract meaningful information

## Why it mattered

The project could not proceed without understanding:
- How to load the data correctly
- What information is in annotations
- The clinical meaning of RR intervals and heartbeat types
- Why raw signal amplitude alone is insufficient for anomaly detection

## Main takeaways

- Normal beats and abnormal beats are not equally represented in the dataset
- The model must learn from rhythm patterns (beat timing) rather than raw signal amplitude
- Feature engineering is not an afterthought; it is central to solving this problem
- Understanding the domain (cardiology, ECG structure) is necessary to make good engineering decisions

## Files

- **Notebook**: `phase1_data_exploration.ipynb`
