# Phase 3: Isolation Forest

This phase introduced the first real anomaly-detection model.

## Why Isolation Forest

The dataset is unsupervised in practice (no labelled anomalies at deployment time) and heavily imbalanced (mostly normal beats). Isolation Forest is a strong fit because:
- It requires no labels
- It handles class imbalance naturally
- It does not rely on distance metrics (which can fail in high-dimensional spaces)

## What the model measures

Isolation Forest asks: **How easy is it to isolate a point from the rest of the distribution?**

Anomaly intuition: Anomalous points sit in sparse regions of the feature space. They need fewer random splits to isolate completely. Normal points, surrounded by neighbors, need many more splits.

## Main learning

The model was effective but depended heavily on the `contamination` hyperparameter—the prior assumption about what fraction of the data are anomalies.

Setting `contamination=0.015` (1.5%) made sense for one patient but became problematic when tested on the full dataset where the overall rate was 30.8% abnormal.

## Challenge identified

Isolation Forest works, but the **calibration prior mismatch** emerges when patient distributions vary widely.

This finding motivated later work on adaptive thresholding.

## Files

- **Notebook**: `phase3_anomaly_detection.ipynb`
