import torch


def mixup_data(x, beta=0.5):
    batch_size = x.size(0)
    shuffled_ids = torch.randperm(batch_size, device=x.device)
    lam = beta if 0.0 <= beta <= 1.0 else 0.5
    mixed = lam * x + (1.0 - lam) * x[shuffled_ids]
    feat_masks = torch.ones_like(x, dtype=torch.bool)
    return mixed, feat_masks, shuffled_ids


def batch_feat_shuffle(x, beta=0.5):
    batch_size, n_features = x.size(0), x.size(1)
    shuffled_ids = torch.randperm(batch_size, device=x.device)
    feat_masks = torch.rand(batch_size, n_features, device=x.device) < beta
    mixed = x.clone()
    mixed[feat_masks] = x[shuffled_ids][feat_masks]
    return mixed, feat_masks, shuffled_ids


def batch_dim_shuffle(x, beta=0.5):
    batch_size, _, n_dims = x.shape
    shuffled_ids = torch.randperm(batch_size, device=x.device)
    dim_masks = torch.rand(batch_size, 1, n_dims, device=x.device) < beta
    mixed = x.clone()
    mixed[dim_masks.expand_as(x)] = x[shuffled_ids][dim_masks.expand_as(x)]
    return mixed, dim_masks.squeeze(1), shuffled_ids
