import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class CrossAttention(nn.Module): 
    def __init__(self): 
        self.q_proj = nn.Conv2d(in_channel, hidden_dim, 3) 
        self.k_proj = nn.Conv2d(in_channel, hidden_dim, 3)
        self.v_proj = nn.Conv2d(in_channel, hidden_dim, 3)
        self.scale = hidden_dim ** -0.5
        self.out = nn.Conv2d(hidden_dim, in_channel)

    def forward(self, feat_A, feat_B):
        Q = self.q_proj(feat_A)
        K = self.q_proj(feat_B)
        V = self.q_proj(feat_B)