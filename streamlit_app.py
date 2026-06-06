import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import wfdb

# Import our Google Fit auth module
from google_fit_auth import authenticate, get_heart_rate, bpm_to_features

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VitalWatch - Live Monitor",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🫀 VitalWatch - Live ECG Anomaly Monitor")
st.markdown(
    "*Real-time heart rate monitoring powered by "
    "MIT-BIH trained ML models and Google Fit*"
)
st.markdown("---")

# ─────────────────────────────────────────────
# Load and train models on MIT-BIH data
# (cached so it only runs once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train_models():
    """
    Loads MIT-BIH data, trains Isolation Forest
    and Autoencoder. Cached - runs once per session.
    """
    st.write("Training models on MIT-BIH data...")

    RECORDS = ['100', '101', '103', '105',
               '106', '108', '109', '111']
    features = ['rr_interval', 'heart_rate', 'rr_diff',
                'rolling_mean_rr', 'rolling_std_rr',
                'relative_rr']

    all_dfs = []
    for record_id in RECORDS:
        try:
            record = wfdb.rdrecord(
                record_id, pn_dir='mitdb')
            annotation = wfdb.rdann(
                record_id, 'atr', pn_dir='mitdb')
            fs = record.fs
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol

            rr_ms = (np.diff(ann_samples) / fs) * 1000
            heart_rate = 60000 / rr_ms
            rr_diff = np.append(0, np.diff(rr_ms))
            rolling_mean = pd.Series(rr_ms).rolling(
                window=5, min_periods=1).mean().values
            rolling_std = pd.Series(rr_ms).rolling(
                window=5, min_periods=1).std().fillna(0).values
            relative_rr = rr_ms / rolling_mean

            df = pd.DataFrame({
                'symbol':          ann_symbols[1:],
                'rr_interval':     rr_ms,
                'heart_rate':      heart_rate,
                'rr_diff':         rr_diff,
                'rolling_mean_rr': rolling_mean,
                'rolling_std_rr':  rolling_std,
                'relative_rr':     relative_rr,
            })
            all_dfs.append(df)
        except Exception as e:
            pass

    combined = pd.concat(all_dfs, ignore_index=True)
    normal = combined[combined['symbol'] == 'N']

    # Scale
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(
        normal[features].values)

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.015,
        random_state=42)
    iso.fit(X_normal)

    # Autoencoder
    input_dim = X_normal.shape[1]
    inputs  = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(inputs)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(input_dim,
                    activation='linear')(decoded)
    ae = Model(inputs=inputs, outputs=decoded)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_normal, X_normal,
           epochs=50, batch_size=32,
           validation_split=0.1, verbose=0,
           callbacks=[EarlyStopping(
               patience=5,
               restore_best_weights=True)])

    # Threshold
    recon = ae.predict(X_normal, verbose=0)
    train_errors = np.mean(
        np.power(X_normal - recon, 2), axis=1)
    ae_threshold = np.percentile(train_errors, 95)

    return scaler, iso, ae, ae_threshold, features


# ─────────────────────────────────────────────
# Authenticate Google Fit
# ─────────────────────────────────────────────
@st.cache_resource
def get_google_fit_service():
    return authenticate()


# ─────────────────────────────────────────────
# Run anomaly detection on live data
# ─────────────────────────────────────────────
def run_detection(df, scaler, iso, ae,
                  ae_threshold, features):
    if df is None or len(df) < 3:
        return None

    X = scaler.transform(df[features].values)

    # Isolation Forest
    iso_pred = iso.predict(X)
    df['iso_flag'] = (iso_pred == -1).astype(int)

    # Autoencoder
    recon = ae.predict(X, verbose=0)
    errors = np.mean(np.power(X - recon, 2), axis=1)
    df['ae_error'] = errors
    df['ae_flag'] = (errors > ae_threshold).astype(int)

    # OR Ensemble
    df['anomaly'] = (
        (df['iso_flag'] == 1) |
        (df['ae_flag'] == 1)
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────

# Load models
with st.spinner("Loading VitalWatch models..."):
    scaler, iso, ae, ae_threshold, features = \
        load_and_train_models()
st.success("Models ready.")

# Authenticate
with st.spinner("Connecting to Google Fit..."):
    service = get_google_fit_service()
st.success("Google Fit connected.")

st.markdown("---")

# Dashboard layout
col1, col2, col3, col4 = st.columns(4)

# Refresh controls
refresh_rate = st.sidebar.slider(
    "Refresh every (seconds)", 30, 300, 60)
hours_back = st.sidebar.slider(
    "Hours of data to fetch", 1, 72, 24)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** OR Ensemble")
st.sidebar.markdown(
    "*(Isolation Forest + Autoencoder)*")
st.sidebar.markdown(
    "**Trained on:** MIT-BIH Arrhythmia Database")
st.sidebar.markdown("**Patients:** 8 training records")

# Live monitoring loop
placeholder = st.empty()

while True:
    with placeholder.container():

        # Fetch data
        with st.spinner("Fetching from Google Fit..."):
            readings = get_heart_rate(
                service, hours_back=hours_back)
            df = bpm_to_features(readings)

        if df is None or len(df) < 3:
            st.warning(
                "Not enough heart rate data yet. "
                "Make sure your Noise watch has "
                "synced to Google Fit recently.")
            st.info(
                f"Refreshing in {refresh_rate} "
                f"seconds...")
            time.sleep(refresh_rate)
            continue

        # Run detection
        result_df = run_detection(
            df.copy(), scaler, iso,
            ae, ae_threshold, features)

        if result_df is None:
            st.warning("Detection failed.")
            continue

        # Metrics
        total = len(result_df)
        anomalies = result_df['anomaly'].sum()
        avg_bpm = result_df['bpm'].mean()
        max_bpm = result_df['bpm'].max()
        min_bpm = result_df['bpm'].min()

        col1.metric("Average HR", f"{avg_bpm:.0f} bpm")
        col2.metric("Max HR", f"{max_bpm:.0f} bpm")
        col3.metric("Min HR", f"{min_bpm:.0f} bpm")
        col4.metric(
            "Anomalies Flagged",
            f"{anomalies}/{total}",
            delta=f"{anomalies/total*100:.1f}%"
            if total > 0 else "0%",
            delta_color="inverse")

        st.markdown("---")

        # Heart rate chart
        st.subheader("Heart Rate Timeline")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(result_df['timestamp'],
                result_df['bpm'],
                color='royalblue',
                linewidth=1.5,
                label='Heart Rate')

        # Mark anomalies
        anomaly_df = result_df[
            result_df['anomaly'] == 1]
        if len(anomaly_df) > 0:
            ax.scatter(
                anomaly_df['timestamp'],
                anomaly_df['bpm'],
                color='red', s=80, zorder=5,
                label='Anomaly Flagged')

        ax.axhline(y=100, color='orange',
                   linestyle='--', alpha=0.5,
                   label='Tachycardia (100 bpm)')
        ax.axhline(y=60, color='green',
                   linestyle='--', alpha=0.5,
                   label='Bradycardia (60 bpm)')
        ax.set_xlabel('Time')
        ax.set_ylabel('BPM')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Alert log
        st.subheader("Alert Log")
        if len(anomaly_df) > 0:
            st.error(
                f"⚠️ {len(anomaly_df)} anomalous "
                f"readings detected")
            display_cols = [
                'timestamp', 'bpm',
                'rr_interval', 'rr_diff',
                'iso_flag', 'ae_flag']
            st.dataframe(
                anomaly_df[display_cols].round(2))
        else:
            st.success(
                "✅ All readings within normal range")

        # Raw data
        with st.expander("View raw data"):
            st.dataframe(result_df[[
                'timestamp', 'bpm',
                'rr_interval', 'heart_rate',
                'rr_diff', 'anomaly'
            ]].round(2))

        # Last updated
        from datetime import datetime
        st.caption(
            f"Last updated: "
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"- refreshing in {refresh_rate}s")

    time.sleep(refresh_rate)