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
        self.rnd_optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=0.0003)

    def forward(self, x):
        """
        x: A single sample with shape [1, self.horizon, input_dim].
        """
        # Flatten the input to shape [self.horizon, input_dim] for both target and prediction networks
        x = x.squeeze(0)  # Remove the batch dimension, x will have shape [self.horizon, input_dim]
        target_out = self.target_net(x)
        prediction_out = self.prediction_net(x)
        rnd_loss = F.mse_loss(prediction_out, target_out)
        return rnd_loss

    def train_step(self, rnd_loss_total):
        """
        Perform a single RND training step using the total loss.

        rnd_loss_total: The total loss computed by summing the RND losses for all samples in a batch.
        """
        self.rnd_optimizer.zero_grad()
        rnd_loss_total.backward()  # Backpropagate the total loss
        self.rnd_optimizer.step()
        return rnd_loss_total