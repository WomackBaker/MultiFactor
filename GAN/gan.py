import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

LATENT_DIM = 16
HIDDEN_DIM = 64
NUM_EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.0002

NUMERIC_COLS = [
    'latitude',
    'longitude',
    'availableMemory',
    'rssi',
    'Processors',
    'Battery',
    'screenWidth',
    'screenLength',
    'screenDensity'
]

BOOL_COLS = [
    'accel',  # If you're treating accel as bool
    'hasTouchScreen',
    'hasCamera',
    'hasFrontCamera',
    'hasMicrophone',
    'hasTemperatureSensor'
]

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def bool_to_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return 1 if value.strip().lower() == 'true' else 0
    return 0

def load_and_combine_csv_files(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except:
            pass
    if not dataframes:
        return None
    return pd.concat(dataframes, ignore_index=True)

def preprocess_user_data(df):
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].apply(bool_to_int)
        else:
            df[col] = 0
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    df.dropna(subset=NUMERIC_COLS + BOOL_COLS, inplace=True)
    final_cols = NUMERIC_COLS + BOOL_COLS
    data_array = df[final_cols].values
    scaler = MinMaxScaler()
    data_array_scaled = scaler.fit_transform(data_array)
    return data_array_scaled, scaler, final_cols

def train_gan_model(real_data, latent_dim, epochs=1000, batch_size=32, lr=0.0002):
    real_data_tensor = torch.tensor(real_data, dtype=torch.float)
    generator = Generator(latent_dim, real_data.shape[1])
    discriminator = Discriminator(real_data.shape[1])
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        idx = torch.randint(0, real_data_tensor.size(0), (batch_size,))
        real_batch = real_data_tensor[idx]
        z = torch.randn(batch_size, latent_dim)
        fake_batch = generator(z).detach()
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))
        real_preds = discriminator(real_batch)
        fake_preds = discriminator(fake_batch)
        d_real_loss = criterion(real_preds, real_labels)
        d_fake_loss = criterion(fake_preds, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        z = torch.randn(batch_size, latent_dim)
        generated_data = generator(z)
        gen_preds = discriminator(generated_data)
        g_loss = criterion(gen_preds, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    return generator, discriminator

def generate_synthetic_users(generator, scaler, num_samples, numeric_cols, bool_cols):
    generator.eval()
    z = torch.randn(num_samples, LATENT_DIM)
    with torch.no_grad():
        synthetic_scaled = generator(z).numpy()
    synthetic_denorm = scaler.inverse_transform(synthetic_scaled)
    num_numeric = len(numeric_cols)
    total_cols = len(numeric_cols) + len(bool_cols)
    for row in synthetic_denorm:
        for i in range(num_numeric, total_cols):
            row[i] = 1 if row[i] >= 0.5 else 0
    return synthetic_denorm

if __name__ == "__main__":
    parent_folder = "./100data"
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for user_folder in subfolders:
        print(f"\nUSER: {os.path.basename(user_folder)}")
        df_combined = load_and_combine_csv_files(user_folder)
        if df_combined is None or df_combined.empty:
            print("No valid CSV data found, skipping.")
            continue
        try:
            user_data_scaled, scaler_obj, final_cols = preprocess_user_data(df_combined)
        except:
            print("Unable to preprocess data, skipping.")
            continue
        generator, discriminator = train_gan_model(
            real_data=user_data_scaled,
            latent_dim=LATENT_DIM,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        synthetic_rows = generate_synthetic_users(
            generator=generator,
            scaler=scaler_obj,
            num_samples=5,
            numeric_cols=NUMERIC_COLS,
            bool_cols=BOOL_COLS
        )
        for row in synthetic_rows:
            for col, val in zip(final_cols, row):
                print(f"{col}: {val}")
            print()
