import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys
import random
from typing import List, Tuple, Dict

OUTPUT_DIR = "output"

BOOLEAN_COLS: List[str] = ['is_rooted', 'vpn_tor_usage']

NUMERIC_COLS: List[str] = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code',
    'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
    'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
    'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
    'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts',
    'historic_risk_score', 'system_mode_code'
]

# --- IP Spoof Attack Generator ---
def generate_ip_spoof_attack(user_row: pd.Series) -> pd.Series:
    spoofed_row = user_row.copy()
    # Slightly different IP (e.g., same subnet but different last octet)
    spoofed_row['ip_address_as_int'] += random.randint(100, 1000)
    # Still an impossible travel but smaller
    spoofed_row['location_conf_radius'] += 500
    # Quick login, but not zero time
    spoofed_row['time_since_last_login_mins'] = random.randint(1, 5)
    return spoofed_row

def _bool_to_int(x) -> int:
    if isinstance(x, bool): return int(x)
    if isinstance(x, str): return 1 if x.lower() == "true" else 0
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0

def normalize_features(df: pd.DataFrame, features_to_scale: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    df_copy = df.copy()
    for col in BOOLEAN_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(_bool_to_int)
    cols_to_scale_existing = [col for col in features_to_scale if col in df_copy.columns]
    scaler = MinMaxScaler()
    df_scaled_values = scaler.fit_transform(df_copy[cols_to_scale_existing])
    df_scaled = pd.DataFrame(df_scaled_values, columns=cols_to_scale_existing, index=df_copy.index)
    return df_scaled, scaler

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

    ml_feature_cols = [col for col in NUMERIC_COLS + BOOLEAN_COLS if col in df.columns]
    df_features_original = df[ml_feature_cols].copy()
    df_features_original['label'] = 0

    # normalize original features
    df_scaled_features_original, _ = normalize_features(df_features_original, ml_feature_cols)
    df_scaled_features_original['label'] = 0  # normal data

    # --- Generate IP Spoofing Attacks ---
    attackers = []
    for _ in range(ATTACKER_COUNT):
        sample_user = df_features_original.sample(n=1).iloc[0]
        attacker_row = generate_ip_spoof_attack(sample_user)
        attackers.append(attacker_row)
    attackers_df = pd.DataFrame(attackers)
    attackers_df['label'] = 1  # attackers

    # combine normal and attack data
    combined_df = pd.concat([df_features_original, attackers_df], ignore_index=True)
    combined_df_scaled, _ = normalize_features(combined_df.drop(columns=['label']), ml_feature_cols)
    combined_df_scaled['label'] = combined_df['label']

    # train/test split
    X = combined_df_scaled.drop(columns=['label'])
    y = combined_df_scaled['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    train_df_final = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df_final = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_df_final.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df_final.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"[+] Successfully wrote train.csv and test.csv to '{OUTPUT_DIR}'")
    print(f"Total samples: {len(combined_df_scaled)}")
    print(f"Train size: {len(train_df_final)} (Attackers: {y_train.sum()})")
    print(f"Test size: {len(test_df_final)} (Attackers: {y_test.sum()})")
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))

if __name__ == "__main__":
    main()