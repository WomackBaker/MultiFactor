import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from typing import List, Tuple, Dict

# --- Configuration ---
INPUT_CSV_PATH = "../GAN/output_with_trust_scores.csv" # Changed to the new output name
OUTPUT_DIR = "output"
ATTACKER_COUNT = 1000 # Number of synthetic attacker samples to generate

BOOLEAN_COLS: List[str] = ['is_rooted', 'vpn_tor_usage']

NUMERIC_COLS: List[str] = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code',
    'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
    'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
    'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
    'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score', 'system_mode_code'
]

# --- Helper Functions ---

def _bool_to_int(x) -> int:
    """Converts boolean-like values to integers (0 or 1)."""
    if isinstance(x, bool): return int(x)
    if isinstance(x, str): return 1 if x.lower() == "true" else 0
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0 # Default to 0 if conversion fails

def normalize_features(df: pd.DataFrame, features_to_scale: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Applies Min-Max scaling to specified feature columns.
    Converts boolean-like columns to integers before scaling if they are in BOOLEAN_COLS.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_to_scale (List[str]): A list of column names to be scaled.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: A tuple containing the scaled DataFrame
                                           and the fitted MinMaxScaler.
    """
    df_copy = df.copy()

    for col in BOOLEAN_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(_bool_to_int)

    # Ensure only columns present in the DataFrame are selected for scaling
    cols_to_scale_existing = [col for col in features_to_scale if col in df_copy.columns]

    scaler = MinMaxScaler()
    df_scaled_values = scaler.fit_transform(df_copy[cols_to_scale_existing])
    df_scaled = pd.DataFrame(df_scaled_values, columns=cols_to_scale_existing, index=df_copy.index)

    return df_scaled, scaler

def generate_synthetic_data(base_df: pd.DataFrame, count: int) -> pd.DataFrame:
    """
    Generates synthetic data by sampling from the base DataFrame
    and adding Gaussian noise. These are treated as "attacker-like" features.

    Args:
        base_df (pd.DataFrame): The DataFrame of original features (already scaled).
        count (int): The number of synthetic samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic samples.
    """
    synthetic_samples = base_df.sample(n=count, random_state=42, replace=True).copy()
    noise = np.random.normal(0, 0.03, synthetic_samples.shape)
    synthetic_samples += noise
    synthetic_samples = np.clip(synthetic_samples, 0, 1) # Clip to maintain [0,1] range
    
    return pd.DataFrame(synthetic_samples, columns=base_df.columns)


# --- Main Execution ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please check the path.")
        return

    # Dynamically identify all feature columns, including trust scores
    # We exclude 'scope_code' from both original features and trust scores if it were to exist.
    # The 'label' column is also excluded here as it's not a feature for scaling.
    all_feature_cols = [col for col in df.columns if col != 'scope_code' and not col.endswith('_trust')]
    
    # Add trust score columns dynamically
    trust_score_cols = [col for col in df.columns if col.endswith('_trust')]
    
    # Combine all columns that need to be part of the feature set for the ML model
    # Maintain original feature order, then interleaved trust scores
    ml_feature_cols = []
    for col in NUMERIC_COLS + BOOLEAN_COLS: # Iterate through the base feature names
        if col != 'scope_code' and col in df.columns: # Check if base feature exists in DF
            ml_feature_cols.append(col)
            if f'{col}_trust' in df.columns: # Check if its trust score column exists
                ml_feature_cols.append(f'{col}_trust')

    # Ensure all selected columns exist in the DataFrame
    missing_cols_in_df = [col for col in ml_feature_cols if col not in df.columns]
    if missing_cols_in_df:
        print(f"Error: Missing expected feature columns in '{INPUT_CSV_PATH}': {missing_cols_in_df}")
        return

    # Select only the relevant feature columns from the original DataFrame for scaling
    # Now, df_features_original will include both original features and their trust scores
    df_features_original = df[ml_feature_cols].copy()

    # Normalize the features of the original dataset, including trust scores
    # Pass the list of all columns that need scaling
    df_scaled_features_original, _ = normalize_features(df_features_original, ml_feature_cols)

    # Assign label 0 (normal user) to all original (now scaled) data points
    df_scaled_features_original['label'] = 0

    # Generate synthetic attacker-like data (features only)
    # The base_df for generation must also contain the trust score columns if they are to be generated
    synthetic_features = generate_synthetic_data(df_scaled_features_original.drop(columns=['label']), ATTACKER_COUNT)

    # Assign label 1 (attacker) to the synthetic data
    synthetic_features['label'] = 1

    # Combine the original scaled features with their labels, and the synthetic features with their labels
    full_df_with_labels = pd.concat([df_scaled_features_original, synthetic_features], ignore_index=True)

    # Shuffle the entire combined dataset to mix normal and attacker samples
    full_df_with_labels = full_df_with_labels.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features (X) and labels (y) for splitting
    X = full_df_with_labels.drop(columns=['label'])
    y = full_df_with_labels['label']

    # Split the combined dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y # Stratify on the actual labels (0s and 1s)
    )

    # Recombine features and labels for saving to CSV, with 'label' as the last column
    train_df_final = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df_final = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Save the training and testing datasets to CSV files with headers
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