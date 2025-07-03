# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:47:29 2025

@author: Altinses, M.Sc.

To-Do:
    - Nothing...
"""

# %% imports

import torch

# %%

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Initialisiere versteckte Zustände
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Nur den letzten Zeitschritt nehmen
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
    
    
# %% test

if __name__ == '__main__':
    model = LSTMModel(100, 100, 5, 20)