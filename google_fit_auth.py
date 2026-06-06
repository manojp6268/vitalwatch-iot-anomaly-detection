import os
import json
import time
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Scope - read heart rate only
SCOPES = ['https://www.googleapis.com/auth/fitness.heart_rate.read']

def authenticate():
    """
    Handles OAuth2 authentication.
    First run opens browser for consent.
    Subsequent runs use saved token.json.
    """
    creds = None

    # Load saved token if it exists
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file(
            'token.json', SCOPES)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for future runs
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('fitness', 'v1', credentials=creds)


def get_heart_rate(service, hours_back=48):
    """
    Uses Google Fit aggregation API instead of dataset API.
    More reliable for consumer wearables like Noise.
    """
    from datetime import datetime, timedelta
    import datetime as dt

    now = datetime.utcnow()
    start = now - timedelta(hours=hours_back)

    # Aggregation API uses milliseconds not nanoseconds
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    body = {
        "aggregateBy": [{
            "dataTypeName": "com.google.heart_rate.bpm"
        }],
        "bucketByTime": {
            "durationMillis": 3600000  # 1 hour buckets
        },
        "startTimeMillis": start_ms,
        "endTimeMillis":   end_ms
    }

    try:
        response = service.users().dataset().aggregate(
            userId='me',
            body=body
        ).execute()

        readings = []
        for bucket in response.get('bucket', []):
            for dataset in bucket.get('dataset', []):
                for point in dataset.get('point', []):
                    timestamp_ns = int(
                        point['startTimeNanos'])
                    timestamp = datetime.fromtimestamp(
                        timestamp_ns / 1e9)
                    for value in point.get('value', []):
                        if 'fpVal' in value:
                            readings.append({
                                'timestamp': timestamp,
                                'bpm': value['fpVal']
                            })

        return readings

    except Exception as e:
        print(f"Error: {e}")
        return []


def bpm_to_features(readings):
    """
    Converts BPM readings into the 6-feature format
    VitalWatch expects.

    BPM → RR interval → all 6 features
    """
    import numpy as np
    import pandas as pd

    if len(readings) < 6:
        print(f"Not enough readings: {len(readings)}. "
              f"Need at least 6.")
        return None

    bpm_values = [r['bpm'] for r in readings]
    timestamps = [r['timestamp'] for r in readings]

    # Derive RR intervals from BPM
    rr_ms = [60000 / bpm for bpm in bpm_values]
    rr_ms = np.array(rr_ms)

    # Build all 6 features
    heart_rate   = np.array(bpm_values)
    rr_diff      = np.append(0, np.diff(rr_ms))
    rolling_mean = pd.Series(rr_ms).rolling(
        window=5, min_periods=1).mean().values
    rolling_std  = pd.Series(rr_ms).rolling(
        window=5, min_periods=1).std().fillna(0).values
    relative_rr  = rr_ms / rolling_mean

    import pandas as pd
    df = pd.DataFrame({
        'timestamp':       timestamps,
        'bpm':             heart_rate,
        'rr_interval':     rr_ms,
        'heart_rate':      heart_rate,
        'rr_diff':         rr_diff,
        'rolling_mean_rr': rolling_mean,
        'rolling_std_rr':  rolling_std,
        'relative_rr':     relative_rr,
    })

    return df


# Test authentication when run directly
if __name__ == '__main__':
    print("Authenticating...")
    service = authenticate()
    print("Authenticated.\n")

    print("Fetching heart rate via aggregation API...")
    readings = get_heart_rate(service, hours_back=72)

    if readings:
        print(f"✓ Found {len(readings)} readings!")
        for r in readings[:5]:
            print(f"  {r['timestamp']} — {r['bpm']:.1f} bpm")
        df = bpm_to_features(readings)
        if df is not None:
            print(f"\nFeature DataFrame ready — shape: {df.shape}")
            print(df.head())
    else:
        print("✗ Still no data.")
        print("\nChecking raw bucket response...")
        # Debug — print raw response
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        start = now - timedelta(hours=72)
        body = {
            "aggregateBy": [{
                "dataTypeName": "com.google.heart_rate.bpm"
            }],
            "bucketByTime": {"durationMillis": 3600000},
            "startTimeMillis": int(start.timestamp() * 1000),
            "endTimeMillis":   int(now.timestamp() * 1000)
        }
        response = service.users().dataset().aggregate(
            userId='me', body=body).execute()
        print(f"Buckets returned: {len(response.get('bucket', []))}")
        print(f"Raw response sample:")
        import json
        print(json.dumps(response, indent=2)[:500])