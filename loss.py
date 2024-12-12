import torch
import torch.nn as nn
from torch.nn import functional as F

def normalize_heatmap(heatmap):
    min_val = heatmap.min()
    max_val = heatmap.max()
    return (heatmap - min_val) / (max_val - min_val + 1e-8)  # Adding a small value for numerical stability

def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss, self).__init__()

    def forward(self, heatmaps):
        # Ensure the heatmaps are in tensor form
        if isinstance(heatmaps, list):
            heatmaps = torch.stack(heatmaps, dim=0)

        # Normalize each heatmap in the batch
        heatmaps = torch.stack([normalize_heatmap(h) for h in heatmaps], dim=0)

        l1_loss = torch.abs(heatmaps).sum()  # L1 norm encourages sparsity
        return l1_loss


class BatchOrthogonalLoss(nn.Module):
    def __init__(self):
        super(BatchOrthogonalLoss, self).__init__()

    def forward(self, heatmaps):
        # Normalize the heatmaps along the last dimension
        norm = torch.norm(heatmaps, p=2, dim=-1, keepdim=True)
        normalized_heatmaps = heatmaps / norm

        # Compute the dot products using batched matrix multiplication
        # Result shape [b, n, n]
        dot_products = torch.bmm(normalized_heatmaps, normalized_heatmaps.transpose(1, 2))
        
        # Zero out the diagonal (self-dot products) since we don't penalize them
        mask = torch.eye(dot_products.size(1), device=dot_products.device).bool()
        dot_products[:, mask] = 0

        # Square the off-diagonal elements and sum them up
        # We use dot_products * (1 - mask) to ignore diagonal elements
        loss = (dot_products ** 2).sum() - (dot_products.diagonal(dim1=1, dim2=2) ** 2).sum()
        return loss