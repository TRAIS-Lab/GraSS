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