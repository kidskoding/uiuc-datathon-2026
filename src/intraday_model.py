import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from src.features import encode_dow

class IntradayNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class IntradayMeanNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

def train(model, X, y, epochs=300, lr=1e-3):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"epoch={epoch}, loss={loss.item():.6f}")
        
    model.eval()
    return model
    
def predict_weights(model, dow_onehot):
    x = torch.tensor(dow_onehot, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
        
    pred = pred.squeeze(0).numpy()
    return pred

def interval_to_slot(interval_str):
    try:
        parts = str(interval_str).split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        
        return hours * 2 + (1 if minutes >= 30 else 0)
    except Exception:
        return -1

def build_training_data(interval_df, metric, normalize):
    df = interval_df.copy()
    
    df['SlotIndex'] = df['Interval'].apply(interval_to_slot)
    df = df[df['SlotIndex'] >= 0]
    
    X_list = []
    y_list = []
    
    for date, day_group in df.groupby("Date"):
        day_group = day_group.sort_values("SlotIndex")
        if len(day_group) != 48:
            continue
        
        values = day_group[metric].to_numpy(dtype=np.float32)
        if np.any(np.isnan(values)):
            continue

        if normalize:
            total = values.sum()
            if total == 0:
                continue
            values = values / total
        
        dow_vec = encode_dow(pd.Timestamp(date).dayofweek)
        
        X_list.append(dow_vec)
        y_list.append(values)
        
    if not X_list:
        return np.empty((0, 7), dtype=np.float32), np.empty((0, 48), dtype=np.float32)
    
    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    return X, y

def build_training_data_cv(interval_df):
    return build_training_data(interval_df, metric="CV", normalize=True)

def build_training_data_mean(interval_df, metric):
    return build_training_data(interval_df, metric=metric, normalize=False)