# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:47:29 2025

@author: Altinses, M.Sc.

To-Do:
    - Nothing...
"""

# %% imports

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_generation import generate_synthetic_data
from lstm_model import LSTMModel
from evaluate import evaluate_model
from config_plots import configure_plt

# %% config

np.random.seed(42)
torch.manual_seed(42)

configure_plt()

# %% generate data

print("Generiere synthetische Sensordaten...")
X, y = generate_synthetic_data()
print(f"Daten Shape: {X.shape}, Labels Shape: {y.shape}")


#%% Train-Test-Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# time series -> features for random forest
def create_rf_features(X):
    """Erstelle Features für Random Forest aus Zeitreihendaten"""
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    maxs = np.max(X, axis=1)
    mins = np.min(X, axis=1)
    
    # Kombiniere alle Features
    X_rf = np.concatenate([means, stds, maxs, mins], axis=1)
    return X_rf

print("Erstelle Features für Random Forest...")

X_train_rf = create_rf_features(X_train)
X_test_rf = create_rf_features(X_test)

# Scale Features
scaler = StandardScaler()
X_train_rf = scaler.fit_transform(X_train_rf)
X_test_rf = scaler.transform(X_test_rf)

X_train_lstm = torch.FloatTensor(X_train)
X_test_lstm = torch.FloatTensor(X_test)
y_train_lstm = torch.FloatTensor(y_train).unsqueeze(1)  # Für BCELoss benötigte Form
y_test_lstm = torch.FloatTensor(y_test).unsqueeze(1)

batch_size = 64
train_dataset = TensorDataset(X_train_lstm, y_train_lstm)
test_dataset = TensorDataset(X_test_lstm, y_test_lstm)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% train model 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende {device} für Training")

# LSTM Hyperparameter
input_size = X_train.shape[2]  # Anzahl Sensoren
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 20

# Initialisiere LSTM Modell
lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Trainiere LSTM Modell
print("\nTrainiere LSTM Modell...")
lstm_train_losses = []
lstm_val_losses = []

for epoch in range(num_epochs):
    lstm_model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Durchschnittlicher Trainingsverlust
    train_loss /= len(train_loader)
    lstm_train_losses.append(train_loss)
    
    # Validierung
    lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = lstm_model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    
    val_loss /= len(test_loader)
    lstm_val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Trainiere Random Forest
print("\nTrainiere Random Forest Modell...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train)


# %% evaluate

lstm_metrics = evaluate_model(lstm_model, X_test_lstm, y_test.squeeze(), 'lstm', device)
rf_metrics = evaluate_model(rf_model, X_test_rf, y_test, 'rf', device)

# Ergebnisse anzeigen
print("\nEvaluierungsergebnisse:")
print("LSTM Modell:")
for metric, value in lstm_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

print("\nRandom Forest Modell:")
for metric, value in rf_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# %% plots

os.makedirs('pred_maintenance_frames', exist_ok=True)

# Plot der Trainingsverläufe
plt.figure(figsize=(6, 3))
plt.plot(lstm_train_losses, label='Train Loss')
plt.plot(lstm_val_losses, label='Validation Loss')
plt.title('LSTM Training Verlauf')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_loss.png', dpi = 300)
plt.close()

# Funktion zur Visualisierung von Vorhersagen
def visualize_predictions(sample_idx, X_test, y_test, lstm_pred, rf_pred, save_path=None):
    """Visualisiere Sensordaten und Vorhersagen für einen Sample"""
    plt.figure(figsize=(6, 6))
    
    # Sensordaten plotten
    for sensor in range(X_test.shape[2]):
        plt.subplot(3, 1, 1)
        plt.plot(X_test[sample_idx, :, sensor], alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Vorhersagen plotten
    plt.subplot(3, 1, 2)
    plt.bar(['LSTM', 'Random Forest'], 
            [lstm_pred[sample_idx], rf_pred[sample_idx]], 
            color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    
    plt.subplot(3, 1, 3)
    plt.bar(['LSTM', 'Random Forest'], 
            [lstm_pred[sample_idx] > 0.5, rf_pred[sample_idx] > 0.5], 
            color=['blue', 'green'])
    plt.ylabel('Predicted')
    plt.yticks([0, 1], ['No', 'Yes'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    plt.close()



lstm_model.eval()
with torch.no_grad():
    lstm_probs = lstm_model(X_test_lstm.to(device)).cpu().numpy().squeeze()

rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]

sample_indices = np.random.choice(len(X_test), size=50, replace=False)

for i, idx in enumerate(tqdm(sample_indices)):
    save_path = f'pred_maintenance_frames/frame_{i:03d}.png'
    visualize_predictions(idx, X_test, y_test, lstm_probs, rf_probs, save_path)
