# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:48:29 2025

@author: Altinses
"""

# %% imports

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%

def evaluate_model(model, X_test, y_test, model_type='lstm', device = 'cpu'):
    """Evaluierte Modellleistung"""
    if model_type == 'lstm':
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            y_pred = model(X_test)
            y_pred = (y_pred > 0.5).float().cpu().numpy()
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# %% test

if __name__ == '__main__':
    data = 0.0