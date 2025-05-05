import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Config
INPUT_CSV = "../GAN/output.csv"   # Change as needed
OUTPUT_DIR = "output"
ATTACKER_COUNT = 100
BOOL_COLS = ['is_rooted', 'vpn_tor_usage']
NUMERIC_COLS = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'gps_latitude',
    'gps_longitude', 'location_conf_radius', 'location_visit_count', 'shift_profile_code',
    'session_start_epoch', 'session_duration_mins', 'time_since_last_login_mins', 'day_type_code',
    'ip_address_as_int', 'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score', 'system_mode_code'
]

def bool_to_int(x):
    if isinstance(x, bool): return int(x)
    if isinstance(x, str): return 1 if x.lower() == "true" else 0
    return int(x)

def normalize(df):
    df = df.copy()
    for col in BOOL_COLS:
        df[col] = df[col].apply(bool_to_int)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[NUMERIC_COLS + BOOL_COLS]), columns=NUMERIC_COLS + BOOL_COLS)
    return df_scaled, scaler

def generate_attackers(user_df, count):
    attackers = user_df.sample(n=count, random_state=42).copy()
    noise = np.random.normal(0, 0.03, attackers.shape)
    attackers += noise
    attackers = np.clip(attackers, 0, 1)
    attacker_df = pd.DataFrame(attackers, columns=user_df.columns)
    attacker_df['label'] = 1
    return attacker_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df_scaled, _ = normalize(df)

    # Assign label 0 to all users
    df_scaled['label'] = 0

    # Generate attackers with label 1
    attacker_df = generate_attackers(df_scaled.drop(columns=['label']), ATTACKER_COUNT)

    # Combine and shuffle
    full_df = pd.concat([df_scaled, attacker_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Train/test split
    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42)

    # Save without headers, no index
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, header=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, header=False)
    print("[+] Wrote train.csv and test.csv to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
