# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:48:29 2025

@author: Altinses
"""

# %% imports

import numpy as np

# %%

def generate_synthetic_data(num_samples=10000, sequence_length=60, num_sensors=5):
    """Generiere synthetische Sensordaten für Predictive Maintenance"""
    # Gesamte Datenmenge
    total_samples = num_samples * sequence_length
    
    # Generiere normale Betriebsdaten
    data = np.random.normal(0, 1, (total_samples, num_sensors))
    
    # Generiere Ausfallmuster (10% der Samples sollen Ausfälle sein)
    failure_indices = np.random.choice(num_samples, size=num_samples//10, replace=False)
    
    # Erzeuge Zeitreihen mit Ausfallmustern
    labels = np.zeros(num_samples)
    for idx in failure_indices:
        # Setze Label für Ausfall
        labels[idx] = 1
        
        # Erzeuge Anomalien in den Sensordaten vor dem Ausfall
        start_anomaly = max(0, idx - np.random.randint(5, 20))
        
        for t in range(start_anomaly, idx):
            # Erhöhe die Varianz und den Mittelwert der Sensoren
            anomaly_factor = 1 + (idx - t) * 1.1  # Linear ansteigende Anomalie
            data[t*sequence_length:(t+1)*sequence_length] += np.random.normal(
                anomaly_factor*0.5, anomaly_factor, (sequence_length, num_sensors))
    
    # Reshape zu (num_samples, sequence_length, num_sensors)
    data = data.reshape(num_samples, sequence_length, num_sensors)
    
    return data, labels


# %% test

if __name__ == '__main__':
    data = generate_synthetic_data()