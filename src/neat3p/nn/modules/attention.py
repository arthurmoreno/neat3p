import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, input_dim, device):
        super(SimpleAttention, self).__init__()
        self.device = device
        self.query = nn.Linear(input_dim, input_dim).to(self.device)
        self.key = nn.Linear(input_dim, input_dim).to(self.device)
        self.value = nn.Linear(input_dim, input_dim).to(self.device)
        self.scale = input_dim**0.5

    def forward(self, x):
        x = x.to(self.device)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


class FeatureAttention(nn.Module):
    def __init__(self, input_dim, device):
        super(FeatureAttention, self).__init__()
        self.device = device
        self.attn_fc = nn.Linear(input_dim, input_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        attn_scores = self.attn_fc(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        filtered = x * attn_weights
        return filtered


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, device: str = "cuda"):
        super(ChannelAttention, self).__init__()
        self.device = torch.device(device)
        self.attn_fc = nn.Linear(num_channels, num_channels).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channels)
        attn_scores = self.attn_fc(x_pool)
        attn_weights = F.softmax(attn_scores, 1)
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)
        return x * attn_weights
