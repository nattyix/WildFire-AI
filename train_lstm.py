# train_lstm.py
# Trains an LSTM to forecast 6 hours of weather from 6 hours of history
# Since UCI dataset has no timestamps, we generate synthetic sequences
# based on the real dataset's statistical properties — standard ML practice

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from preprocess import BASE, CSV_PATH

os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
LSTM_PATH   = os.path.join(BASE, "models", "lstm_weather.pth")
SCALER_PATH = os.path.join(BASE, "models", "lstm_scaler.pkl")

# ── FEATURES used by LSTM ─────────────────────────────────────────────────────
# We use the 8 core weather features — not one-hot encoded columns
# The LSTM needs continuous numeric features to learn temporal patterns
WEATHER_FEATURES = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI']
SEQ_LEN          = 6    # 6 hours of history as input
PRED_LEN         = 6    # predict next 6 hours
N_FEATURES       = len(WEATHER_FEATURES)

# ── LOAD REAL DATA STATISTICS ─────────────────────────────────────────────────
print("Loading UCI dataset statistics...")
df = pd.read_csv(CSV_PATH)
df = df[WEATHER_FEATURES].copy()

# Get real mean and std for each feature
# We use these to generate realistic synthetic sequences
means  = df.mean().values    # shape: (8,)
stds   = df.std().values     # shape: (8,)
mins   = df.min().values
maxs   = df.max().values

print(f"Dataset stats loaded for {N_FEATURES} features")
print(f"Temp range: {mins[0]:.1f} - {maxs[0]:.1f} °C")
print(f"Humidity range: {mins[1]:.1f} - {maxs[1]:.1f} %")

# ── GENERATE SYNTHETIC TIME-SERIES ───────────────────────────────────────────
# We create realistic weather sequences by:
# 1. Starting from a random real data point
# 2. Adding smooth temporal evolution (not random noise)
# 3. Ensuring values stay within realistic bounds
# This is called "trajectory simulation" and is used in climate ML research

def generate_sequences(n_sequences=5000, seq_total=SEQ_LEN+PRED_LEN):
    """
    Generate n_sequences of weather trajectories.
    Each sequence has seq_total timesteps.
    First SEQ_LEN = input (known history)
    Last PRED_LEN = target (what LSTM should predict)
    """
    sequences = []

    for _ in range(n_sequences):
        # Start from a random real observation as anchor
        anchor_idx = np.random.randint(len(df))
        anchor     = df.iloc[anchor_idx].values.copy()  # shape: (8,)

        seq = [anchor]

        for t in range(1, seq_total):
            prev = seq[t-1].copy()
            # Smooth evolution: each step changes by a small random amount
            # The 0.05 factor keeps changes realistic hour-to-hour
            delta = np.random.randn(N_FEATURES) * stds * 0.05

            # Temperature tends to follow a daily cycle
            # Add a slight warming trend in the first 6 hours, cooling after
            if t < seq_total // 2:
                delta[0] += 0.3   # temp rising (daytime heating)
                delta[1] -= 0.5   # humidity dropping as temp rises
            else:
                delta[0] -= 0.2   # temp falling
                delta[1] += 0.3   # humidity recovering

            # Wind tends to increase with temperature
            delta[2] += delta[0] * 0.1

            next_step = prev + delta
            # Clip to realistic bounds
            next_step = np.clip(next_step, mins, maxs)
            seq.append(next_step)

        sequences.append(np.array(seq))   # shape: (seq_total, 8)

    return np.array(sequences)   # shape: (n_sequences, seq_total, 8)

print("\nGenerating synthetic weather sequences...")
all_seqs = generate_sequences(n_sequences=8000)
print(f"Generated {len(all_seqs)} sequences of length {SEQ_LEN+PRED_LEN}")

# Split into input (X) and target (y)
X_seqs = all_seqs[:, :SEQ_LEN,  :]   # shape: (8000, 6, 8) — input history
y_seqs = all_seqs[:, SEQ_LEN:,  :]   # shape: (8000, 6, 8) — target forecast

# Scale sequences
# Fit scaler on the anchor points (first timestep of each sequence)
lstm_scaler = StandardScaler()
lstm_scaler.fit(all_seqs.reshape(-1, N_FEATURES))

X_scaled = lstm_scaler.transform(
    X_seqs.reshape(-1, N_FEATURES)).reshape(X_seqs.shape)
y_scaled = lstm_scaler.transform(
    y_seqs.reshape(-1, N_FEATURES)).reshape(y_seqs.shape)

# Train/val split
split     = int(0.85 * len(X_scaled))
X_train   = torch.FloatTensor(X_scaled[:split])
y_train   = torch.FloatTensor(y_scaled[:split])
X_val     = torch.FloatTensor(X_scaled[split:])
y_val     = torch.FloatTensor(y_scaled[split:])

print(f"Train sequences: {len(X_train)} | Val sequences: {len(X_val)}")

# ── DATASET ───────────────────────────────────────────────────────────────────
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(WeatherDataset(X_train, y_train),
                          batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(WeatherDataset(X_val,   y_val),
                          batch_size=64, shuffle=False, num_workers=0)

# ── LSTM MODEL ────────────────────────────────────────────────────────────────
# Encoder-Decoder LSTM:
# Encoder reads the 6-hour history and compresses it into a context vector
# Decoder uses that context to predict the next 6 hours one step at a time
# This is the same architecture used in weather forecasting research

class WeatherLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=128, n_layers=2,
                 pred_len=PRED_LEN):
        super().__init__()
        self.pred_len  = pred_len
        self.n_features= n_features

        # Encoder: reads input sequence
        self.encoder = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = 0.2
        )

        # Decoder: generates predictions one step at a time
        self.decoder = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = 0.2
        )

        # Output projection: hidden state → weather features
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_features)
        )

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        x      : input sequence [batch, seq_len, features]
        target : ground truth future [batch, pred_len, features] (for training)
        teacher_forcing_ratio: probability of using real target as next input
                               during training — helps the model learn faster
        """
        batch_size = x.size(0)

        # Encode history → get context (hidden + cell state)
        _, (hidden, cell) = self.encoder(x)

        # Start decoder with last known timestep
        decoder_input = x[:, -1:, :]   # [batch, 1, features]

        predictions = []
        for t in range(self.pred_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)         # [batch, 1, features]
            predictions.append(pred)

            # Teacher forcing: sometimes use real value as next input
            # This stabilizes training on small datasets
            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred

        return torch.cat(predictions, dim=1)   # [batch, pred_len, features]

model     = WeatherLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"\nLSTM parameters: {total_params:,}")
print(f"Architecture: Encoder-Decoder LSTM | Hidden: 128 | Layers: 2")

# ── TRAINING ──────────────────────────────────────────────────────────────────
EPOCHS   = 30
best_val = float('inf')

print("\n" + "="*55)
print("Training LSTM weather forecaster...")
print("="*55)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # Use teacher forcing during training (ratio decreases over epochs)
        tf_ratio = max(0.1, 0.5 - epoch * 0.015)
        pred     = model(X_batch, y_batch, teacher_forcing_ratio=tf_ratio)
        loss     = criterion(pred, y_batch)
        loss.backward()
        # Gradient clipping prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred      = model(X_batch, teacher_forcing_ratio=0.0)
            val_loss += criterion(pred, y_batch).item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    scheduler.step(avg_val)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
          f"TF ratio: {max(0.1, 0.5-epoch*0.015):.2f}")

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), LSTM_PATH)
        print(f"  ✅ Best model saved (val loss: {avg_val:.4f})")

# ── EVALUATE: test a prediction ───────────────────────────────────────────────
print("\n=== Sample Forecast Test ===")
model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
model.eval()

# Take a real weather sequence as input
sample_raw = df.sample(6).values              # 6 real weather rows
sample_sc  = lstm_scaler.transform(sample_raw)
sample_t   = torch.FloatTensor(sample_sc).unsqueeze(0)  # [1, 6, 8]

with torch.no_grad():
    forecast_sc = model(sample_t, teacher_forcing_ratio=0.0)

forecast = lstm_scaler.inverse_transform(
    forecast_sc.squeeze(0).numpy())  # shape: (6, 8)

print(f"\nInput (last known hour):")
print(f"  Temp={sample_raw[-1,0]:.1f}°C  "
      f"RH={sample_raw[-1,1]:.1f}%  "
      f"Wind={sample_raw[-1,2]:.1f}km/h  "
      f"FFMC={sample_raw[-1,4]:.1f}")
print(f"\n6-Hour Forecast:")
for h, row in enumerate(forecast):
    print(f"  +{h+1}h: Temp={row[0]:.1f}°C  "
          f"RH={row[1]:.1f}%  "
          f"Wind={row[2]:.1f}km/h  "
          f"FFMC={row[4]:.1f}")

# Save scaler
pickle.dump(lstm_scaler, open(SCALER_PATH, 'wb'))
print(f"\nLSTM saved to    : {LSTM_PATH}")
print(f"Scaler saved to  : {SCALER_PATH}")
print("\nNext step: run spread_simulation.py")