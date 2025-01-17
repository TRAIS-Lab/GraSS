import torch

from scipy.stats import spearmanr

def compute_pairwise_distance_metrics(grad_t, grad_p):
    """
    Computes relative error, RMSE, and stress between pairwise distances of
    original and projected datasets.

    Arguments:
    grad_t -- tensor of original data (batch of gradients)
    grad_p -- tensor of projected data (batch of projected gradients)

    Returns:
    relative_error -- average relative error between original and projected pairwise distances
    rmse -- root mean squared error between original and projected pairwise distances
    stress -- stress function measuring global distance preservation
    """

    # Compute pairwise distances
    original_distances = torch.cdist(grad_t, grad_t, p=2)
    projected_distances = torch.cdist(grad_p, grad_p, p=2)

    # Avoid division by zero for any zero distances in the original data
    mask = original_distances > 1e-8  # Mask to filter out zero distances

    # Compute Relative Error
    relative_errors = torch.abs((original_distances[mask] - projected_distances[mask]) / original_distances[mask])
    average_relative_error = torch.mean(relative_errors).item()

    # Compute RMSE (Root Mean Squared Error)
    # mse = torch.mean((original_distances[mask] - projected_distances[mask]) ** 2)
    # rmse = torch.sqrt(mse).item()

    # Compute Stress
    # stress = torch.sqrt(torch.sum((original_distances[mask] - projected_distances[mask]) ** 2) /
    #                     torch.sum(original_distances[mask] ** 2)).item()

    return average_relative_error#, rmse, stress

def compute_pairwise_inner_product_rank_correlation(grad_t, grad_p):
    """
    Computes Spearman's rank correlation coefficient for pairwise inner products
    between original and projected datasets.

    Arguments:
    grad_t -- tensor of original data (batch of gradients)
    grad_p -- tensor of projected data (batch of projected gradients)

    Returns:
    spearman_corr -- Spearman's rank correlation coefficient
    """

    # Compute pairwise inner products (original and projected)
    original_inner_products = torch.matmul(grad_t, grad_t.T).flatten()
    projected_inner_products = torch.matmul(grad_p, grad_p.T).flatten()

    # Convert tensors to numpy arrays for rank calculation
    original_inner_products_np = original_inner_products.cpu().numpy()
    projected_inner_products_np = projected_inner_products.cpu().numpy()

    # Compute Spearman's rank correlation
    spearman_corr, _ = spearmanr(original_inner_products_np, projected_inner_products_np)

    return spearman_corr

# def SJLT(batch_vec, proj_dim, rand_indices, rand_signs, c=2, blow_up=1):
#     """
#     Forward SJLT implementation optimized for sparse inputs

#     Args:
#         batch_vec (torch.Tensor): Input tensor of shape [batch_size, original_dim]
#         proj_dim (int): Target projection dimension
#         rand_indices (torch.Tensor): Random indices of shape [original_dim, c], values in [0, proj_dim * blow_up)
#         rand_signs (torch.Tensor): Random signs of shape [original_dim, c], values in {-1, 1}
#         c (int): Sparsity parameter. Default: 2
#         blow_up (int): Intermediate dimension multiplier. Default: 1

#     Returns:
#         torch.Tensor: Projected tensor of shape [batch_size, proj_dim]
#     """
#     batch_size, original_dim = batch_vec.size()

#     batch_vec_p = torch.zeros(batch_size, proj_dim * blow_up, device=batch_vec.device)

#     for i in range(batch_size):
#         vec = batch_vec[i]
#         non_zero_indices = torch.nonzero(vec).squeeze()
#         if non_zero_indices.numel() == 0:
#             continue

#         # Multiply the non-zero elements of batch_vec by their corresponding random signs
#         scaled_vals = vec[non_zero_indices].unsqueeze(1) * rand_signs[non_zero_indices] # Shape (num_non_zero, c_int)

#         # Perform vectorized index addition (summing over c_int for each non-zero element)
#         batch_vec_p[i].index_add_(0, rand_indices[non_zero_indices].flatten(), scaled_vals.flatten())

#     batch_vec_p = batch_vec_p.view(batch_size, proj_dim, blow_up)
#     batch_vec_p = batch_vec_p.sum(dim=2)

#     return batch_vec_p / (c ** 0.5)


def SJLT(vecs, proj_dim, seed=0, rand_indices=None, rand_signs=None, c=2, blow_up=1):
    """
    Forward and batched SJLT implementation optimized for sparse inputs

    Args:
        vecs (torch.Tensor): Input tensor of shape [batch_size, original_dim]
        proj_dim (int): Target projection dimension
        seed (int): Random seed for reproducibility. Default: 0
        rand_indices (torch.Tensor): Random indices of shape [original_dim, c], values in [0, proj_dim * blow_up)
        rand_signs (torch.Tensor): Random signs of shape [original_dim, c], values in {-1, 1}
        c (int): Sparsity parameter. Default: 2
        blow_up (int): Intermediate dimension multiplier. Default: 1

    Returns:
        torch.Tensor: Projected tensor of shape [batch_size, proj_dim]
    """
    torch.manual_seed(seed)

    batch_size, original_dim = vecs.size()
    device = vecs.device

    if rand_indices is None:
        rand_indices = torch.randint(proj_dim * blow_up, (original_dim, c), device=device)
    if rand_signs is None:
        rand_signs = torch.randint(0, 2, (original_dim, c), device=device) * 2 - 1

    # Get indices of all non-zero elements across the batch
    batch_idx, input_idx = torch.nonzero(vecs, as_tuple=True)

    if input_idx.numel() == 0:
        return torch.zeros(batch_size, proj_dim, device=device)

    values = vecs[batch_idx, input_idx]
    scaled_vals = values.unsqueeze(1) * rand_signs[input_idx]
    output_indices = rand_indices[input_idx]

    vecs_p = torch.zeros(batch_size, proj_dim * blow_up, device=device)

    # Create virtual repeated indices using broadcasting
    final_indices = (batch_idx.view(-1, 1) * (proj_dim * blow_up) + output_indices).flatten()

    vecs_p.view(-1).index_add_(0, final_indices, scaled_vals.flatten())

    vecs_p = vecs_p.view(batch_size, proj_dim, blow_up)
    vecs_p = vecs_p.sum(dim=2)

    return vecs_p / (c ** 0.5)

def SJLT_reverse(batch_vec, proj_dim, pos_indices=None, neg_indices=None, c=2):
    """
    Backward SJLT implementation that processes entire batch at once.

    Args:
        batch_vec (torch.Tensor): Input tensor of shape [batch_size, original_dim]
        proj_dim (int): Target projection dimension
        pos_indices (torch.Tensor): Precomputed positive contribution indices of shape [proj_dim, max_pos_len]
        neg_indices (torch.Tensor): Precomputed negative contribution indices of shape [proj_dim, max_neg_len]
        c (int): Sparsity parameter

    Return:
        torch.Tensor: Projected tensor of shape [batch_size, proj_dim]
    """
    batch_size, original_dim = batch_vec.size()
    max_pos_len = pos_indices.size(1) if pos_indices is not None else 0
    max_neg_len = neg_indices.size(1) if neg_indices is not None else 0

    # Initialize output projected vector
    batch_vec_p = torch.zeros(batch_size, proj_dim, device=batch_vec.device)

    # Create masks for valid indices once
    pos_mask = (pos_indices != -1).float()  # [proj_dim, max_pos_len]
    neg_mask = (neg_indices != -1).float()  # [proj_dim, max_neg_len]

    # Process positive contributions for entire batch at once
    if max_pos_len > 0:
        # Gather and reshape: [batch_size, proj_dim, max_pos_len]
        pos_values = batch_vec[:, pos_indices.clamp(min=0)]
        # Apply mask and sum: [batch_size, proj_dim]
        batch_vec_p += (pos_values * pos_mask).sum(dim=2)

    # Process negative contributions for entire batch at once
    if max_neg_len > 0:
        # Gather and reshape: [batch_size, proj_dim, max_neg_len]
        neg_values = batch_vec[:, neg_indices.clamp(min=0)]
        # Apply mask and sum: [batch_size, proj_dim]
        batch_vec_p -= (neg_values * neg_mask).sum(dim=2)

    return batch_vec_p / (c ** 0.5)

def backward_SJLT_indices(original_dim, proj_dim, c, device, rand_indices=None, rand_signs=None):
    """
    Vectorized computation of SJLT contribution indices.

    Args:
        original_dim (int): Original input dimension
        proj_dim (int): Target projection dimension
        c (int): Sparsity parameter
        device (torch.device): Device to create tensors on
        rand_indices (torch.Tensor): Optional precomputed random indices [original_dim, c]
        rand_signs (torch.Tensor): Optional precomputed random signs [original_dim, c]
    """
    # Generate random indices and signs if not provided
    if rand_indices is None:
        rand_indices = torch.randint(proj_dim, (original_dim, c), device=device)
    if rand_signs is None:
        rand_signs = torch.randint(0, 2, (original_dim, c), device=device) * 2 - 1

    # Create index pairs for sorting
    input_indices = torch.arange(original_dim, device=device).repeat_interleave(c)
    output_indices = rand_indices.reshape(-1)
    signs = rand_signs.reshape(-1)

    # Split positive and negative contributions
    pos_mask = signs == 1
    neg_mask = ~pos_mask

    # Handle positive contributions
    pos_input_indices = input_indices[pos_mask]
    pos_output_indices = output_indices[pos_mask]

    # Handle negative contributions
    neg_input_indices = input_indices[neg_mask]
    neg_output_indices = output_indices[neg_mask]

    # Sort by output indices for both positive and negative contributions
    pos_sort_idx = torch.argsort(pos_output_indices)
    neg_sort_idx = torch.argsort(neg_output_indices)

    pos_input_sorted = pos_input_indices[pos_sort_idx]
    pos_output_sorted = pos_output_indices[pos_sort_idx]

    neg_input_sorted = neg_input_indices[neg_sort_idx]
    neg_output_sorted = neg_output_indices[neg_sort_idx]

    # Find unique output indices and their counts
    pos_unique, pos_counts = torch.unique_consecutive(pos_output_sorted, return_counts=True)
    neg_unique, neg_counts = torch.unique_consecutive(neg_output_sorted, return_counts=True)

    # Calculate maximum lengths
    max_pos_len = pos_counts.max().item() if len(pos_counts) > 0 else 0
    max_neg_len = neg_counts.max().item() if len(neg_counts) > 0 else 0

    # Create output tensors
    pos_indices = torch.full((proj_dim, max_pos_len), -1, device=device)
    neg_indices = torch.full((proj_dim, max_neg_len), -1, device=device)

    # Fill positive indices
    if max_pos_len > 0:
        pos_start_idx = torch.cat([torch.tensor([0], device=device), pos_counts.cumsum(0)[:-1]])
        for i, (output_idx, count, start) in enumerate(zip(pos_unique, pos_counts, pos_start_idx)):
            pos_indices[output_idx, :count] = pos_input_sorted[start:start + count]

    # Fill negative indices
    if max_neg_len > 0:
        neg_start_idx = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
        for i, (output_idx, count, start) in enumerate(zip(neg_unique, neg_counts, neg_start_idx)):
            neg_indices[output_idx, :count] = neg_input_sorted[start:start + count]

    return pos_indices, neg_indices