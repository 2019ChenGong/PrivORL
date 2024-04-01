import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TargetNetwork(nn.Module):
    def __init__(self, input_dim):
        super(TargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.)
    
    def forward(self, x):
        return self.network(x)


class PredictionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PredictionNetwork, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        return self.network(x)


# rnd
class Rnd(nn.Module):
    def __init__(self, input_dim, device):
        super(Rnd, self).__init__()
        self.target_net = TargetNetwork(input_dim=input_dim).to(device)
        self.prediction_net = PredictionNetwork(input_dim=input_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=0.001)

    def forward(self, x):
        target_out = self.target_net(x)
        prediction_out = self.prediction_net(x)
        rnd_loss = F.mse_loss(prediction_out, target_out)
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        
        return rnd_loss
