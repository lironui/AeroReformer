import torch
import torch.nn as nn
# -------------------------------
# L2 Normalization Function
# -------------------------------
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / (torch.norm(x, p=2, dim=-2) + 1e-6))


# -------------------------------
# Spatial Attention Module
# -------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_places, eps=1e-6):
        super(SpatialAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        Q = self.query_conv(x).view(B, -1, H * W)
        K = self.key_conv(x).view(B, -1, H * W)
        V = self.value_conv(x).view(B, -1, H * W)

        Q = self.l2_norm(Q).permute(0, 2, 1)  # [B, N, C']
        K = self.l2_norm(K)                  # [B, C', N]

        tailor_sum = 1 / (H * W + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1)) + self.eps)
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, C, H * W)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(B, C, H, W)

        return (x + self.gamma * weight_value).contiguous()


# -------------------------------
# Channel Attention Module
# -------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        Q = x.view(B, C, -1)
        K = x.view(B, C, -1)
        V = x.view(B, C, -1)

        Q = self.l2_norm(Q)  # [B, C, N]
        K = self.l2_norm(K).permute(0, 2, 1)  # [B, N, C]

        tailor_sum = 1 / (H * W + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2)) + self.eps)
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, C, H * W)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)
        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(B, C, H, W)

        return (x + self.gamma * weight_value).contiguous()
