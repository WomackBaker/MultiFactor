import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys
import random
from typing import List, Tuple

OUTPUT_DIR = "output"

# Add 'overseas_ip' so it flows through preprocessing and into the model
BOOLEAN_COLS: List[str] = ['is_rooted', 'vpn_tor_usage', 'overseas_ip']

NUMERIC_COLS: List[str] = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code',
    'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
    'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
    'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
    'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts',
    'historic_risk_score', 'system_mode_code'
]

# U.S. region boxes (same as generate.py) for location-based overseas flag
US_BOXES = [
    # 1.1
    {"lat_min": 25.0, "lat_max": 47.0, "lon_min": -84.0,  "lon_max": -67.0},
    # 1.2
    {"lat_min": 30.0, "lat_max": 45.0, "lon_min": -101.0, "lon_max": -90.0},
    # 1.3
    {"lat_min": 32.0, "lat_max": 48.0, "lon_min": -125.0, "lon_max": -114.0},
]

def in_any_us_box(lat: float, lon: float) -> bool:
    """Return True if (lat, lon) is inside any of the known U.S. boxes."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    for b in US_BOXES:
        if (b["lat_min"] <= lat <= b["lat_max"]) and (b["lon_min"] <= lon <= b["lon_max"]):
            return True
    return False

def generate_ip_spoof_attack(user_row):
    spoofed_row = user_row.copy()
    # make the network look off
    spoofed_row['ip_address_as_int'] += random.randint(2000, 5000)
    spoofed_row['location_conf_radius'] += random.randint(200, 500)
    spoofed_row['time_since_last_login_mins'] = random.randint(1, 3)

    # force stronger attacker signals (still plausible)
    # vpn/tor: very likely on
    spoofed_row['vpn_tor_usage'] = 1
    # ip rep: skew to bad (3) most of the time, sometimes 2
    spoofed_row['ip_reputation_code'] = 3 if random.random() < 0.7 else 2
    # overseas: attackers often appear from abroad
    spoofed_row['overseas_ip'] = 1 if random.random() < 0.6 else 0

    # trust: 95% low, 5% “camouflage” in a tiny overlap band
    if random.random() < 0.95:
        spoofed_row['trust_score'] = random.uniform(0.0, 0.35)
    else:
        spoofed_row['trust_score'] = random.uniform(0.65, 0.70)

    spoofed_row['label'] = 1
    return spoofed_row


def _bool_to_int(x) -> int:
    if isinstance(x, bool): return int(x)
    if isinstance(x, str): return 1 if x.lower() == "true" else 0
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0

def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    df_copy = df.copy()
    for col in BOOLEAN_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(_bool_to_int)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_copy.drop(columns=['label']))
    scaled_df = pd.DataFrame(scaled_values, columns=df_copy.drop(columns=['label']).columns, index=df_copy.index)
    scaled_df['label'] = df_copy['label'].values
    return scaled_df, scaler

def flip_labels(df: pd.DataFrame, flip_fraction=0.01):
    total = len(df)
    flip_count = int(total * flip_fraction)
    flip_indices = np.random.choice(df.index, flip_count, replace=False)
    df.loc[flip_indices, 'label'] = 1 - df.loc[flip_indices, 'label']
    print(f"[DEBUG] Flipped {flip_count} labels for noise injection.")
    return df

def adjust_trust_scores(df: pd.DataFrame):
    # Attackers slightly lower trust, normals slightly higher
    df.loc[df['label'] == 1, 'trust_score'] *= np.random.uniform(0.7, 0.9)
    df.loc[df['label'] == 0, 'trust_score'] *= np.random.uniform(1.0, 1.1)
    return df

def derive_overseas_flag(df: pd.DataFrame) -> pd.Series:
    """
    Location-based 'overseas' flag:
    overseas_ip = 1 if (gps_latitude, gps_longitude) is outside all U.S. boxes, else 0.
    """
    return (~df.apply(lambda r: in_any_us_box(r.get('gps_latitude'), r.get('gps_longitude')), axis=1)).astype(int)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if len(sys.argv) < 3:
        print("Usage: python split.py <attacker_count> <data_path>")
        sys.exit(1)

    ATTACKER_COUNT = int(sys.argv[1])
    data_path = sys.argv[2]
    if not os.path.exists(data_path):
        print("Error: Input data file not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # --- Compute overseas flag on the *raw* data ---
    df['overseas_ip'] = derive_overseas_flag(df)

    feature_cols = [col for col in NUMERIC_COLS + BOOLEAN_COLS + ['trust_score'] if col in df.columns]
    normal_df = df[feature_cols].copy()
    normal_df['label'] = 0

    # --- Generate attackers ---
    attackers = []
    for _ in range(ATTACKER_COUNT):
        sample_user = normal_df.sample(n=1).iloc[0]
        attacker_row = generate_ip_spoof_attack(sample_user)
        attackers.append(attacker_row)
    attackers_df = pd.DataFrame(attackers)

    combined_df = pd.concat([normal_df, attackers_df], ignore_index=True)

    # Recompute 'overseas_ip' if GPS changed upstream (safety); here we keep GPS untouched,
    # so we only ensure the column exists (attackers_df inherits it from sample_user).
    if 'overseas_ip' not in combined_df.columns:
        combined_df['overseas_ip'] = derive_overseas_flag(combined_df)

    # --- Enforce rule: overseas => trust <= 0.20 (automatic low) ---
    if 'trust_score' in combined_df.columns:
        mask_overseas = combined_df['overseas_ip'] == 1
        combined_df.loc[mask_overseas, 'trust_score'] = np.minimum(combined_df.loc[mask_overseas, 'trust_score'], 0.20)

    # Inject slight label noise
    combined_df = flip_labels(combined_df, flip_fraction=0.02)

    # Adjust trust scores slightly for realism (after noise injection is fine)
    combined_df = adjust_trust_scores(combined_df)

    print("\n[DEBUG] Trust score mean by class BEFORE normalization:")
    print(combined_df.groupby('label')['trust_score'].mean())

    combined_df_scaled, _ = normalize_features(combined_df)

    X = combined_df_scaled.drop(columns=['label'])
    y = combined_df_scaled['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    train_df_final = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df_final = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_df_final.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df_final.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"\n[+] Successfully wrote train.csv and test.csv to '{OUTPUT_DIR}'")
    print(f"Total samples: {len(combined_df_scaled)}")
    print(f"Train size: {len(train_df_final)} (Attackers: {y_train.sum()})")
    print(f"Test size: {len(test_df_final)} (Attackers: {y_test.sum()})")
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))

if __name__ == "__main__":
    main()
