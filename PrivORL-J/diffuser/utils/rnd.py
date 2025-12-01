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
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: A single sample with shape [1, self.horizon, input_dim] or batch of samples [batch_size, self.horizon, input_dim].
        Each forward call will update the prediction network.
        """
        # Store original shape for handling both single sample and batch
        original_shape = x.shape

        # Flatten the input: if batch, shape becomes [batch_size * horizon, input_dim]
        if len(x.shape) == 3:
            batch_size, horizon, input_dim = x.shape
            x = x.reshape(-1, input_dim)
        elif len(x.shape) == 2:
            # Already flattened [horizon, input_dim]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")

        # Forward pass through both networks
        target_out = self.target_net(x)
        prediction_out = self.prediction_net(x)

        # Compute RND loss
        rnd_loss = F.mse_loss(prediction_out, target_out)

        # Update prediction network immediately on each forward
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        # Return detached loss for logging/selection purposes
        return rnd_loss.detach()

    def train_step(self, rnd_loss_total):
        """
        This method is kept for backward compatibility but is now deprecated.
        The training step is already performed in forward().

        rnd_loss_total: The total loss (already detached from forward).
        """
        # No-op: training already happened in forward()
        return rnd_loss_total