import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import sys

# --- Configuration Constants ---
LATENT_DIM      = 16
HIDDEN_DIM      = 64
NUM_EPOCHS      = 1000
BATCH_SIZE      = 32
LEARNING_RATE   = 0.0002

NUMERIC_COLS = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'gps_latitude',
    'gps_longitude', 'location_conf_radius', 'location_visit_count', 'shift_profile_code',
    'session_start_epoch', 'session_duration_mins', 'time_since_last_login_mins', 'day_type_code',
    'ip_address_as_int', 'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score', 'system_mode_code'
]
BOOL_COLS = ['is_rooted', 'vpn_tor_usage']

# --- Trust Scores for Each Feature (Adjust these values as needed) ---
FEATURE_TRUST_SCORES = {
    'region_tz_code': 0.8,
    'os_code': 0.75,
    'device_type_code': 0.7,
    'manufacturer_code': 0.6,
    'is_rooted': 0.9,
    'gps_latitude': 0.7,
    'gps_longitude': 0.7,
    'location_conf_radius': 0.65,
    'location_visit_count': 0.5,
    'shift_profile_code': 0.6,
    'session_start_epoch': 0.55,
    'session_duration_mins': 0.5,
    'time_since_last_login_mins': 0.4,
    'day_type_code': 0.6,
    'ip_address_as_int': 0.65,
    'ip_reputation_code': 0.75,
    'vpn_tor_usage': 0.9,
    'typing_speed_cpm': 0.85,
    'click_pattern_code': 0.7,
    'role_code': 0.8,
    'scope_code': 0.75,
    'failed_login_attempts': 0.95,
    'historic_risk_score': 0.9,
    'system_mode_code': 0.8
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Preprocessing Functions ---

def bool_to_int(value):
    """Converts boolean-like values to integers (1 for true, 0 for false)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return 1 if value.strip().lower() == 'true' else 0
    return 0

def preprocess_user_data(df: pd.DataFrame):
    """
    Preprocesses the raw user data by mapping categorical values,
    converting booleans, and scaling numeric features.
    """
    cat_maps = {
        'region_tz_code': {1.1: 0, 1.2: 1, 1.3: 2},
        'os_code': {1: 0, 2: 1, 3: 2, 4: 3},
        'device_type_code': {1: 0, 2: 1},
        'manufacturer_code': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        'shift_profile_code': {1: 0, 2: 1, 3: 2},
        'day_type_code': {0: 0, 1: 1},
        'ip_reputation_code': {1: 0, 2: 1, 3: 2},
        'click_pattern_code': {1: 0, 2: 1, 3: 2},
        'role_code': {1: 0, 2: 1, 3: 2},
        'scope_code': {1: 0, 2: 1, 3: 2},
        'system_mode_code': {0: 0, 1: 1},
    }

    for col, mapping in cat_maps.items():
        df[col] = pd.to_numeric(df[col], errors='coerce').map(mapping)

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].apply(bool_to_int)
        else:
            df[col] = 0

    df = df.dropna(subset=NUMERIC_COLS + BOOL_COLS).copy()
    
    # Store region_tz_code separately as it's the categorical output for the GAN
    regions = df['region_tz_code'].astype(int)
    
    # Remove region_tz_code from the main DataFrame for continuous processing
    df_for_scaling = df.drop(columns=['region_tz_code'])

    # Scale the remaining numeric and boolean data
    data = df_for_scaling.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, regions.values, scaler, df_for_scaling.columns.tolist()

# --- GAN Model Definitions ---

class Generator(nn.Module):
    def __init__(self, latent_dim, cont_dim, num_classes=3):
        """
        Initializes the Generator network.
        Args:
            latent_dim (int): Dimension of the input noise vector.
            cont_dim (int): Dimension of the continuous output features.
            num_classes (int): Number of categories for the categorical output.
        """
        super().__init__()
        self.fc1       = nn.Linear(latent_dim, HIDDEN_DIM)
        self.fc2       = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.cont_head = nn.Linear(HIDDEN_DIM, cont_dim)
        self.cat_head  = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, z):
        """
        Forward pass for the Generator.
        Args:
            z (torch.Tensor): Input noise vector.
        Returns:
            tuple: Continuous features and categorical logits.
        """
        h = F.leaky_relu(self.fc1(z), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        cont_out   = self.cont_head(h)
        cat_logits = self.cat_head(h)
        return cont_out, cat_logits


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes the Discriminator network.
        Args:
            input_dim (int): Dimension of the input features (continuous features).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid() # Output a probability (real/fake)
        )

    def forward(self, x):
        """
        Forward pass for the Discriminator.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            torch.Tensor: Probability of the input being real.
        """
        return self.net(x)

# --- GAN Training Function ---

def train_gan_model(real_cont, real_cat, latent_dim, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Trains the Generative Adversarial Network.
    Args:
        real_cont (np.ndarray): Real continuous data.
        real_cat (np.ndarray): Real categorical data.
        latent_dim (int): Dimension of the noise vector for the Generator.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimizers.
    Returns:
        tuple: Trained Generator and Discriminator models.
    """
    real_cont_t = torch.tensor(real_cont, dtype=torch.float, device=device)
    real_cat_t  = torch.tensor(real_cat,  dtype=torch.long,  device=device)

    cont_dim = real_cont.shape[1] # Number of continuous features
    gen = Generator(latent_dim, cont_dim).to(device)
    disc = Discriminator(cont_dim).to(device)

    g_optim = optim.Adam(gen.parameters(), lr=lr)
    d_optim = optim.Adam(disc.parameters(), lr=lr)
    bce = nn.BCELoss() # Binary Cross Entropy for discriminator
    ce  = nn.CrossEntropyLoss() # Cross Entropy for generator's categorical output

    for epoch in range(epochs):
        # --- Train Discriminator ---
        # Get a batch of real continuous data
        idx = torch.randint(0, real_cont_t.size(0), (batch_size,), device=device)
        real_c = real_cont_t[idx]
        
        # Generate fake continuous data
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_c, _ = gen(z) # We only need continuous part for discriminator
        
        # Discriminator's prediction on real and fake data
        d_real = disc(real_c)
        d_fake = disc(fake_c.detach()) # Detach to prevent gradients from flowing to generator
        
        # Calculate discriminator loss
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        
        # Backpropagate and update discriminator
        d_optim.zero_grad()
        loss_d.backward()
        d_optim.step()

        # --- Train Generator ---
        # Generate new fake data
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_c, fake_logits = gen(z)
        
        # Generator's adversarial loss (wants discriminator to classify fake as real)
        adv_loss = bce(disc(fake_c), torch.ones(batch_size, 1, device=device))
        
        # Generator's categorical classification loss (wants to produce correct categories)
        # We use real_cat_t[idx] for the categorical loss here. This encourages the generator
        # to produce diverse categorical outputs that align with the real distribution.
        class_loss = ce(fake_logits, real_cat_t[idx]) 
        
        # Total generator loss
        loss_g = adv_loss + 0.1 * class_loss # 0.1 is a weighting factor for class_loss
        
        # Backpropagate and update generator
        g_optim.zero_grad()
        loss_g.backward()
        g_optim.step()

    return gen, disc

# --- Synthetic Data Generation Function ---

def generate_synthetic_users(generator, scaler, cont_cols, num_samples=1):
    """
    Generates synthetic user data using the trained Generator.
    Args:
        generator (nn.Module): Trained Generator model.
        scaler (MinMaxScaler): Scaler used for original data to inverse transform.
        cont_cols (list): List of original continuous column names.
        num_samples (int): Number of synthetic users to generate.
    Returns:
        list: A list of dictionaries, where each dictionary represents a synthetic user
              with features and their corresponding trust scores.
    """
    generator.eval() # Set generator to evaluation mode
    z = torch.randn(num_samples, LATENT_DIM, device=device)
    
    with torch.no_grad(): # No gradient calculations needed for inference
        cont_out, logits = generator(z)
        cont_np = cont_out.cpu().numpy() # Move to CPU and convert to NumPy array
        # Get the predicted categorical index by taking the argmax of the softmax output
        cat_idx = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()

    # Inverse transform the continuous data to original scale
    cont_denorm = scaler.inverse_transform(cont_np)

    # Reverse mappings for categorical columns
    reverse_maps = {
        'region_tz_code': {0: 1.1, 1: 1.2, 2: 1.3},
        'os_code': {0: 1, 1: 2, 2: 3, 3: 4},
        'device_type_code': {0: 1, 1: 2},
        'manufacturer_code': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        'shift_profile_code': {0: 1, 1: 2, 2: 3},
        'day_type_code': {0: 0, 1: 1},
        'ip_reputation_code': {0: 1, 1: 2, 2: 3},
        'click_pattern_code': {0: 1, 1: 2, 2: 3},
        'role_code': {0: 1, 1: 2, 2: 3},
        'scope_code': {0: 1, 1: 2, 2: 3},
        'system_mode_code': {0: 0, 1: 1},
    }

    decoded_rows = []
    for i, row in enumerate(cont_denorm):
        decoded_row = {}
        original_features = {} # Temporarily store original feature values for easier processing
        
        # Handle 'region_tz_code' which was the categorical output
        original_features['region_tz_code'] = reverse_maps['region_tz_code'][cat_idx[i]]

        # Populate original_features with other continuous/boolean columns
        k = 0 # Index for cont_cols, tracking which column in cont_denorm we are processing
        for col in cont_cols:
            if col != 'region_tz_code': # Skip region_tz_code as it's handled separately
                if col in reverse_maps:
                    # Map numerical code back to original value (e.g., 0 to 1 for os_code)
                    original_features[col] = reverse_maps[col].get(int(round(row[k])), row[k])
                elif col in BOOL_COLS:
                    # Convert scaled boolean back to 0 or 1
                    original_features[col] = int(round(row[k]))
                else:
                    # For other numeric columns, round to 2 decimal places
                    original_features[col] = round(row[k], 2)
                k += 1 # Move to the next index in the denormalized row

        # Define the base order of columns for the output CSV
        # This list determines the order of features *before* trust scores are interleaved.
        base_output_columns = [
            'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'is_rooted',
            'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
            'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
            'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
            'ip_reputation_code', 'vpn_tor_usage', 'typing_speed_cpm', 'click_pattern_code',
            'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score',
            'system_mode_code'
        ]

        # Now, add features and their trust scores to decoded_row in the desired interleaved order
        for col in base_output_columns:
            decoded_row[col] = original_features.get(col)
            # Add trust score if available for the feature in our predefined scores
            if col in FEATURE_TRUST_SCORES:
                decoded_row[f'{col}_trust'] = FEATURE_TRUST_SCORES[col]
        
        decoded_rows.append(decoded_row)

    return decoded_rows

# --- Main Execution Block ---

if __name__ == "__main__":
    sample_num = int(sys.argv[1])

    if (not sample_num) or int(sample_num) <= 0:
        print("Usage: python gan.py <number_of_samples>")
        exit(1)

    file_path = "../GenerateData/sample_data.csv"
    output_path = "output_with_trust_scores.csv" # Changed output filename

    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'. Please check the file path.")
        exit(1)

    # Load and preprocess the data
    df = pd.read_csv(file_path)
    try:
        real_cont, real_cat, scaler, cont_cols = preprocess_user_data(df)
        print(f"Successfully preprocessed data. Continuous features shape: {real_cont.shape}, Categorical features shape: {real_cat.shape}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        exit(1)

    # Train the GAN model
    print("Starting GAN training...")
    gen, disc = train_gan_model(real_cont, real_cat, LATENT_DIM)
    print("GAN training complete.")

    # Generate synthetic users with trust scores
    print("Generating synthetic users...")
    synthetic_users = generate_synthetic_users(gen, scaler, cont_cols, sample_num)
    print(f"Generated {len(synthetic_users)} synthetic users.")

    # Define the exact column order for the output CSV, including interleaved trust scores
    final_column_order = []
    # This list represents the features in the order they should appear in the output CSV
    # Each feature will be followed by its trust score column if defined.
    base_features_for_ordering = [
        'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'is_rooted',
        'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
        'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
        'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
        'ip_reputation_code', 'vpn_tor_usage', 'typing_speed_cpm', 'click_pattern_code',
        'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score',
        'system_mode_code'
    ]

    for col in base_features_for_ordering:
        final_column_order.append(col)
        if col in FEATURE_TRUST_SCORES:
            final_column_order.append(f'{col}_trust')

    # Create a DataFrame from the generated synthetic users
    df_synthetic = pd.DataFrame(synthetic_users)
    
    # Reindex the DataFrame to ensure columns are in the specified order.
    # If a column in final_column_order doesn't exist in df_synthetic, it will be added with NaN.
    # This helps ensure consistency even if a trust score wasn't defined for a feature.
    df_synthetic = df_synthetic.reindex(columns=final_column_order)
    
    # Save the synthetic data with trust scores to a CSV file
    df_synthetic.to_csv(output_path, index=False)
    print(f"Synthetic data with trust scores saved to '{output_path}'")