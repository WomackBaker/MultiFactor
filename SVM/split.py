import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from typing import List, Tuple, Dict
import sys

OUTPUT_DIR = "output"

BOOLEAN_COLS: List[str] = ['is_rooted', 'vpn_tor_usage']

NUMERIC_COLS: List[str] = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code',
    'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
    'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
    'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
    'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score', 'system_mode_code'
]

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

def generate_synthetic_data(base_df: pd.DataFrame, count: int) -> pd.DataFrame:
    synthetic_samples = base_df.sample(n=count, random_state=42, replace=True).copy()
    noise = np.random.normal(0, 0.03, synthetic_samples.shape)
    synthetic_samples += noise
    synthetic_samples = np.clip(synthetic_samples, 0, 1)
    return pd.DataFrame(synthetic_samples, columns=base_df.columns)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ATTACKER_COUNT = sys.argv[1]
    data_path = sys.argv[2]
    if not ATTACKER_COUNT.isdigit() or int(ATTACKER_COUNT) <= 0:
        print("python split.py <attacker_count>")
        exit(1)
    if not os.path.exists(data_path):
        print("Default data will be used")
        data_path = "../GAN/output_with_trust_scores.csv"

    INPUT_CSV_PATH = data_path

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please check the path.")
        return
    ml_feature_cols = [col for col in NUMERIC_COLS + BOOLEAN_COLS if col != 'scope_code' and col in df.columns]
    missing_cols_in_df = [col for col in ml_feature_cols if col not in df.columns]
    if missing_cols_in_df:
        print(f"Error: Missing expected feature columns in '{INPUT_CSV_PATH}': {missing_cols_in_df}")
        return
    df_features_original = df[ml_feature_cols].copy()
    df_scaled_features_original, _ = normalize_features(df_features_original, ml_feature_cols)
    df_scaled_features_original['label'] = 0
    synthetic_features = generate_synthetic_data(df_scaled_features_original.drop(columns=['label']), int(ATTACKER_COUNT))
    synthetic_features['label'] = 1
    full_df_with_labels = pd.concat([df_scaled_features_original, synthetic_features], ignore_index=True)
    full_df_with_labels = full_df_with_labels.sample(frac=1, random_state=42).reset_index(drop=True)
    X = full_df_with_labels.drop(columns=['label'])
    y = full_df_with_labels['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    train_df_final = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df_final = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    train_df_final.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, header=True)
    test_df_final.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, header=True)
    print(f"[+] Successfully wrote train.csv and test.csv to '{OUTPUT_DIR}'")
    print(f"Total blended samples (original + synthetic): {len(full_df_with_labels)}")
    print(f"Train set size: {len(train_df_final)} (Attackers: {train_df_final['label'].sum()})")
    print(f"Test set size: {len(test_df_final)} (Attackers: {test_df_final['label'].sum()})")
    print("\nClass Distribution in Combined Data:")
    print(full_df_with_labels['label'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
