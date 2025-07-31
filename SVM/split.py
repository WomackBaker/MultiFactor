import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys
import random

OUTPUT_DIR = "output"

BOOLEAN_COLS = ['is_rooted', 'vpn_tor_usage']
NUMERIC_COLS = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code',
    'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
    'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
    'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
    'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts',
    'historic_risk_score', 'system_mode_code'
]

def generate_ip_spoof_attack(user_row):
    spoofed_row = user_row.copy()
    spoofed_row['ip_address_as_int'] += random.randint(2000, 5000)
    spoofed_row['location_conf_radius'] += random.randint(200, 500)
    spoofed_row['time_since_last_login_mins'] = random.randint(1, 3)
    spoofed_row['trust_score'] = random.uniform(0.0, 0.4)  # Low trust for attacker
    spoofed_row['label'] = 1
    return spoofed_row

def _bool_to_int(x):
    if isinstance(x, bool): return int(x)
    if isinstance(x, str): return 1 if x.lower() == "true" else 0
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0

def normalize_features(df):
    df_copy = df.copy()
    for col in BOOLEAN_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(_bool_to_int)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_copy.drop(columns=['label']))
    scaled_df = pd.DataFrame(scaled_values, columns=df_copy.drop(columns=['label']).columns)
    scaled_df['label'] = df_copy['label']
    return scaled_df, scaler

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

    feature_cols = [col for col in NUMERIC_COLS + BOOLEAN_COLS if col in df.columns]
    normal_df = df[feature_cols].copy()
    normal_df['label'] = 0
    normal_df['trust_score'] = [random.uniform(0.7, 1.0) for _ in range(len(normal_df))]  # High trust normal

    # Generate attackers
    attackers = []
    for _ in range(ATTACKER_COUNT):
        sample_user = normal_df.sample(n=1).iloc[0]
        attacker_row = generate_ip_spoof_attack(sample_user)
        attackers.append(attacker_row)
    attackers_df = pd.DataFrame(attackers)

    # Combine normal + attacker
    combined_df = pd.concat([normal_df, attackers_df], ignore_index=True)

    print("\n[DEBUG] Sample attacker rows BEFORE normalization:")
    print(attackers_df.head())

    combined_df_scaled, _ = normalize_features(combined_df)

    # Split into train/test
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
