# Phase 2: Feature Engineering

This phase focused on turning ECG timing information into useful model inputs.

## Main features explored

1. **RR interval** – Time between consecutive R-peaks (heartbeats)
2. **Heart rate** – Derived from RR interval (60,000 / RR_ms)
3. **RR difference** – Change from previous RR interval
4. **Rolling mean RR** – Local rhythm baseline over 5 beats
5. **Rolling std RR** – Local rhythm variability over 5 beats
6. **Relative RR** – Current RR normalized by rolling mean

## Why these matter

The key insight: rhythm behavior is often more informative than raw signal intensity when the goal is abnormal heartbeat detection.

A premature ventricular contraction (PVC) is not characterized by different voltage—it is characterized by arriving earlier than expected. The model needs features that capture timing, not amplitude.

## What was understood

The project shifted from thinking about raw waveforms to thinking about beat-level summaries. This made the problem much more tractable for unsupervised learning.

## Key insight

The model does not need to see the full waveform to identify rhythm anomalies. It needs the right summary of beat-to-beat behavior.

This insight guided all subsequent model development.

## Files

- **Notebook**: `phase2_feature_engineering.ipynb`
