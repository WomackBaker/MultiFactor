import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

LATENT_DIM    = 16
HIDDEN_DIM    = 64
NUM_EPOCHS    = 1000
BATCH_SIZE    = 32
LEARNING_RATE = 0.0002

NUMERIC_COLS = [
    'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'gps_latitude',
    'gps_longitude', 'location_conf_radius', 'location_visit_count', 'shift_profile_code',
    'session_start_epoch', 'session_duration_mins', 'time_since_last_login_mins', 'day_type_code',
    'ip_address_as_int', 'ip_reputation_code', 'typing_speed_cpm', 'click_pattern_code',
    'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score', 'system_mode_code'
]
BOOL_COLS = ['is_rooted', 'vpn_tor_usage']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bool_to_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return 1 if value.strip().lower() == 'true' else 0
    return 0


def preprocess_user_data(df: pd.DataFrame):
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
    regions = df['region_tz_code'].astype(int)
    df = df.drop(columns=['region_tz_code'])

    data = df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, regions.values, scaler, df.columns.tolist()


class Generator(nn.Module):
    def __init__(self, latent_dim, cont_dim, num_classes=3):
        super().__init__()
        self.fc1      = nn.Linear(latent_dim, HIDDEN_DIM)
        self.fc2      = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.cont_head = nn.Linear(HIDDEN_DIM, cont_dim)
        self.cat_head  = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        cont_out   = self.cont_head(h)
        cat_logits = self.cat_head(h)
        return cont_out, cat_logits


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


def train_gan_model(real_cont, real_cat, latent_dim, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    real_cont_t = torch.tensor(real_cont, dtype=torch.float, device=device)
    real_cat_t  = torch.tensor(real_cat,  dtype=torch.long,  device=device)

    cont_dim = real_cont.shape[1]
    gen = Generator(latent_dim, cont_dim).to(device)
    disc = Discriminator(cont_dim).to(device)

    g_optim = optim.Adam(gen.parameters(), lr=lr)
    d_optim = optim.Adam(disc.parameters(), lr=lr)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        idx = torch.randint(0, real_cont_t.size(0), (batch_size,), device=device)
        real_c = real_cont_t[idx]
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_c, _ = gen(z)
        d_real = disc(real_c)
        d_fake = disc(fake_c.detach())
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        d_optim.zero_grad()
        loss_d.backward()
        d_optim.step()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_c, fake_logits = gen(z)
        adv_loss = bce(disc(fake_c), torch.ones(batch_size, 1, device=device))
        class_loss = ce(fake_logits, real_cat_t[idx])
        loss_g = adv_loss + 0.1 * class_loss
        g_optim.zero_grad()
        loss_g.backward()
        g_optim.step()

    return gen, disc


def generate_synthetic_users(generator, scaler, cont_cols, num_samples=1):
    generator.eval()
    z = torch.randn(num_samples, LATENT_DIM, device=device)
    with torch.no_grad():
        cont_out, logits = generator(z)
        cont_np = cont_out.cpu().numpy()
        cat_idx = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()

    cont_denorm = scaler.inverse_transform(cont_np)

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
        for j, col in enumerate(cont_cols):
            if col in reverse_maps:
                decoded_row[col] = reverse_maps[col].get(int(round(row[j])), row[j])
            elif col in BOOL_COLS:
                decoded_row[col] = int(round(row[j]))
            else:
                decoded_row[col] = round(row[j], 2)
        decoded_row['region_tz_code'] = reverse_maps['region_tz_code'][cat_idx[i]]
        decoded_rows.append(decoded_row)

    return decoded_rows


if __name__ == "__main__":
    file_path = "../GenerateData/sample_data.csv"
    output_path = "output.csv"

    if not os.path.exists(file_path):
        print("File not found. Please check the file path.")
        exit(1)

    df = pd.read_csv(file_path)
    try:
        real_cont, real_cat, scaler, cont_cols = preprocess_user_data(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        exit(1)

    gen, disc = train_gan_model(real_cont, real_cat, LATENT_DIM)

    synthetic_users = generate_synthetic_users(gen, scaler, cont_cols, num_samples=10000)

    column_order = [
        'region_tz_code', 'os_code', 'device_type_code', 'manufacturer_code', 'is_rooted',
        'gps_latitude', 'gps_longitude', 'location_conf_radius', 'location_visit_count',
        'shift_profile_code', 'session_start_epoch', 'session_duration_mins',
        'time_since_last_login_mins', 'day_type_code', 'ip_address_as_int',
        'ip_reputation_code', 'vpn_tor_usage', 'typing_speed_cpm', 'click_pattern_code',
        'role_code', 'scope_code', 'failed_login_attempts', 'historic_risk_score',
        'system_mode_code'
    ]

    df_synthetic = pd.DataFrame(synthetic_users)
    df_synthetic = df_synthetic[column_order]
    df_synthetic.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")