# from layers.linear import GCLinear
import torch
import torch.nn as nn
import torch
import numpy as np
from modelgc import GCLinear
import sys, os
import time


def replace_Linear(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        # print out shows <class 'torch.nn.modules.linear.Linear'>
        if type(layer) == torch.nn.Linear:
            new_layer = GCLinear(in_features=layer.in_features, out_features=layer.out_features)
            new_layer.weight = layer.weight
            new_layer.weight.requires_grad = True
            del layer
            print('Found Linear Layer: {}'.format(layer_str))
            setattr(module, layer_str, new_layer)
            print('Replaced Linear Layer: {}'.format(layer_str))
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_Linear(immediate_child_module)


def replace_embedding(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        # print out shows <class 'torch.nn.modules.linear.Linear'>
        if type(layer) == torch.nn.Embedding:
            new_layer = GCLinear(in_features=layer.in_features, out_features=layer.out_features)
            new_layer.weight = layer.weight
            new_layer.weight.requires_grad = True
            del layer
            print('Found Linear Layer: {}'.format(layer_str))
            setattr(module, layer_str, new_layer)
            print('Replaced Linear Layer: {}'.format(layer_str))
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_Linear(immediate_child_module)


def find_GClayers(module):

    GC_layers = []

    for layer_str in dir(module):
        layer = getattr(module, layer_str)

        if type(layer) in [GCLinear]:
            # print('Found GC Layer: {}'.format(layer_str))
            GC_layers.append( layer )

    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            GC_layers = GC_layers + find_GClayers(immediate_child_module)

    return GC_layers


# if per_token is True, then return the gradient similarity between training loss
# and per-token validation loss.
def compute_TracIN_GC_per_iter_cover(model, device, train_data, val_data, optimizer,
                               trainable_layers,
                               return_tracin_and_similarity=True, return_val_similarity=True):

    per_val=False

    X, Y = train_data
    X_val, Y_val = val_data
    batch_size = X.shape[0]
    n_val = X_val.shape[0]

    optimizer.zero_grad()

    dLdZ_a_val_lst = []

    val_logits, val_loss = model(X_val, Y_val, return_per_token=True)

    # # pick val_loss subset with loss smaller than 10
    # print('val_loss fewer loss: ', val_loss.shape)

    # scale each loss by exp(-loss), but exp(-loss) is treated as constant
    val_loss_value = val_loss.detach().cpu().numpy()
    val_loss = val_loss * torch.exp(-torch.from_numpy(val_loss_value)).to(device)

    val_loss = val_loss.mean()

    val_pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad_val = torch.autograd.grad(val_loss, val_pre_acts, retain_graph=True)
    assert len(trainable_layers) == len(Z_grad_val)
    for layer, zgrad_val in zip(trainable_layers, Z_grad_val):
        decompose_result = layer.pe_grad_gradcomp(zgrad_val, per_sample=True)
        dLdZ_a_val_lst = update_list(dLdZ_a_val_lst, decompose_result)

    optimizer.zero_grad()

    # Compute individual training loss
    train_logits, train_loss = model(X, Y)
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad = torch.autograd.grad(train_loss, pre_acts, retain_graph=False)

    dLdZ_a_train_lst = []
    for layer, zgrad in zip(trainable_layers, Z_grad):
        decompose_result = layer.pe_grad_gradcomp(zgrad, per_sample=True)
        dLdZ_a_train_lst = update_list(dLdZ_a_train_lst, decompose_result)

    # Compute TracIN score
    tracin_local_score = np.zeros( (batch_size, n_val) ) if per_val else np.zeros(batch_size)

    if return_tracin_and_similarity:
        similarity_local_score = np.zeros( (batch_size, batch_size) )

    if return_val_similarity:
        val_similarity_score = np.zeros( (n_val, n_val) )

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)
    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        dLdZ = dLdZ.detach()
        a = a.detach()

        dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

        if per_val:
            tracin_local_score += ((dot_prod).float()).cpu().detach().numpy()
        else:
            tracin_local_score += ((dot_prod).mean(dim=1).float()).cpu().detach().numpy()

        if return_tracin_and_similarity:
            dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += ((dot_prod).float()).cpu().detach().numpy()

        if return_val_similarity:
            dot_prod = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
            val_similarity_score += ((dot_prod).float()).cpu().detach().numpy()

    if return_val_similarity:
        return tracin_local_score, similarity_local_score, val_similarity_score

    if return_tracin_and_similarity:
        return tracin_local_score, similarity_local_score
    else:
        return tracin_local_score




# if per_token is True, then return the gradient similarity between training loss
# and per-token validation loss.
def compute_TracIN_GC_per_iter_pertoken(model, device, train_data, val_data, optimizer,
                               trainable_layers,
                               return_tracin_and_similarity=True, return_val_similarity=True,
                               token_id=0):

    per_val=False

    X, Y = train_data
    X_val, Y_val = val_data
    batch_size = X.shape[0]
    n_val = X_val.shape[0]

    optimizer.zero_grad()

    dLdZ_a_val_lst = []
    val_logits, val_loss = model(X_val, Y_val, return_per_token=True)

    # val_loss_value = val_loss.detach().cpu().numpy()
    # val_loss = val_loss * torch.exp(-torch.from_numpy(val_loss_value)).to(device)

    val_loss_value = val_loss[:, token_id].item()
    print('TokenLoss={}'.format(val_loss_value))

    val_loss = val_loss[:, token_id]

    val_pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad_val = torch.autograd.grad(val_loss, val_pre_acts, retain_graph=True)
    assert len(trainable_layers) == len(Z_grad_val)
    for layer, zgrad_val in zip(trainable_layers, Z_grad_val):
        decompose_result = layer.pe_grad_gradcomp(zgrad_val, per_sample=True)
        dLdZ_a_val_lst = update_list(dLdZ_a_val_lst, decompose_result)

    optimizer.zero_grad()

    # Compute individual training loss
    train_logits, train_loss = model(X, Y)
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad = torch.autograd.grad(train_loss, pre_acts, retain_graph=False)

    dLdZ_a_train_lst = []
    for layer, zgrad in zip(trainable_layers, Z_grad):
        decompose_result = layer.pe_grad_gradcomp(zgrad, per_sample=True)
        dLdZ_a_train_lst = update_list(dLdZ_a_train_lst, decompose_result)

    # Compute TracIN score
    tracin_local_score = np.zeros( (batch_size, n_val) ) if per_val else np.zeros(batch_size)

    if return_tracin_and_similarity:
        similarity_local_score = np.zeros( (batch_size, batch_size) )

    if return_val_similarity:
        val_similarity_score = np.zeros( (n_val, n_val) )

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)
    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        dLdZ = dLdZ.detach()
        a = a.detach()

        dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

        if per_val:
            tracin_local_score += ((dot_prod).float()).cpu().detach().numpy()
        else:
            tracin_local_score += ((dot_prod).mean(dim=1).float()).cpu().detach().numpy()

        if return_tracin_and_similarity:
            dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += ((dot_prod).float()).cpu().detach().numpy()

        if return_val_similarity:
            dot_prod = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
            val_similarity_score += ((dot_prod).float()).cpu().detach().numpy()

    if return_val_similarity:
        return tracin_local_score, similarity_local_score, val_similarity_score

    if return_tracin_and_similarity:
        return tracin_local_score, similarity_local_score, val_loss_value
    else:
        return tracin_local_score, val_loss_value




def compute_value_per_iter_inrun(model, device, train_data, val_data, optimizer, trainable_layers,
                                 return_tracin_and_similarity=True, return_val_similarity=True):

    per_val=False

    X, Y = train_data
    X_val, Y_val = val_data
    batch_size = X.shape[0]
    n_val = X_val.shape[0]

    X_combined = torch.cat((X, X_val), dim=0)
    Y_combined = torch.cat((Y, Y_val), dim=0)

    optimizer.zero_grad()

    full_logits, full_loss = model(X_combined, Y_combined) # Aggregated loss
    full_pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

    dLdZ_a_val_lst = []
    dLdZ_a_train_lst = []
    for layer, zgrad_full in zip(trainable_layers, Z_grad_full):
        decompose_result = layer.pe_grad_gradcomp(zgrad_full, per_sample=True)
        val_1, val_2 = decompose_result

        decompose_result_val = (val_1[batch_size:, :, :], val_2[batch_size:, :, :])
        dLdZ_a_val_lst = update_list(dLdZ_a_val_lst, decompose_result_val)

        decompose_result_train = (val_1[:batch_size, :, :], val_2[:batch_size, :, :])
        dLdZ_a_train_lst = update_list(dLdZ_a_train_lst, decompose_result_train)

    first_order_score = torch.zeros(batch_size, n_val, device='cuda') if per_val else torch.zeros(batch_size, device='cuda')

    if return_tracin_and_similarity:
        second_order_interaction = torch.zeros((batch_size, batch_size), device='cuda')

    if return_val_similarity:
        val_similarity_score = torch.zeros((n_val, n_val), device='cuda')

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)

    with torch.no_grad():
        for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

            dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

            if per_val:
                first_order_score += (dot_prod).float()
            else:
                first_order_score += (dot_prod).mean(dim=1).float()

            if return_tracin_and_similarity:
                dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
                second_order_interaction += dot_prod

            if return_val_similarity:
                dot_prod = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
                val_similarity_score += dot_prod

    if return_val_similarity:
        return first_order_score, second_order_interaction, val_similarity_score

    if return_tracin_and_similarity:
        return first_order_score, second_order_interaction
    else:
        return first_order_score





def compute_TracIN_GC_per_iter(model, device, train_data, val_data, optimizer,
                               trainable_layers,
                               return_tracin_and_similarity=True, return_val_similarity=True):

    per_val=False

    X, Y = train_data
    X_val, Y_val = val_data
    batch_size = X.shape[0]
    n_val = X_val.shape[0]

    optimizer.zero_grad()

    dLdZ_a_val_lst = []
    val_logits, val_loss = model(X_val, Y_val)
    val_pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad_val = torch.autograd.grad(val_loss, val_pre_acts, retain_graph=True)
    assert len(trainable_layers) == len(Z_grad_val)
    for layer, zgrad_val in zip(trainable_layers, Z_grad_val):
        decompose_result = layer.pe_grad_gradcomp(zgrad_val, per_sample=True)
        dLdZ_a_val_lst = update_list(dLdZ_a_val_lst, decompose_result)

    optimizer.zero_grad()

    # Compute individual training loss
    train_logits, train_loss = model(X, Y)
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad = torch.autograd.grad(train_loss, pre_acts, retain_graph=False)

    dLdZ_a_train_lst = []
    for layer, zgrad in zip(trainable_layers, Z_grad):
        decompose_result = layer.pe_grad_gradcomp(zgrad, per_sample=True)
        dLdZ_a_train_lst = update_list(dLdZ_a_train_lst, decompose_result)

    # Compute TracIN score
    tracin_local_score = np.zeros( (batch_size, n_val) ) if per_val else np.zeros(batch_size)

    if return_tracin_and_similarity:
        similarity_local_score = np.zeros( (batch_size, batch_size) )

    if return_val_similarity:
        val_similarity_score = np.zeros( (n_val, n_val) )

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)
    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        dLdZ = dLdZ.detach()
        a = a.detach()

        dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

        if per_val:
            tracin_local_score += ((dot_prod).float()).cpu().detach().numpy()
        else:
            tracin_local_score += ((dot_prod).mean(dim=1).float()).cpu().detach().numpy()

        if return_tracin_and_similarity:
            dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += ((dot_prod).float()).cpu().detach().numpy()

        if return_val_similarity:
            dot_prod = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
            val_similarity_score += ((dot_prod).float()).cpu().detach().numpy()

    if return_val_similarity:
        return tracin_local_score, similarity_local_score, val_similarity_score

    if return_tracin_and_similarity:
        return tracin_local_score, similarity_local_score
    else:
        return tracin_local_score



def update_list(original, input_element):
    # Check if the input is a list
    if isinstance(input_element, list):
        # Concatenate with the original list
        return original + input_element
    else:
        # Append to the original list
        original.append(input_element)
        return original


def grad_dotprod(A1, B1, A2, B2) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        return grad_dotprod_non_sequential(A1, B1, A2, B2)
    elif A1.dim() == 3 and B1.dim() == 3:
        return grad_dotprod_sequential(A1, B1, A2, B2)
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")


def grad_dotprod_non_sequential(A1, B1, A2, B2):

    dot_prod_1 = torch.matmul(A1, A2.T)
    dot_prod_2 = torch.matmul(B1, B2.T)
    dot_prod = dot_prod_1*dot_prod_2

    return dot_prod


def grad_dotprod_sequential(A1, B1, A2, B2):

    (b, t, p), (_, _, d) = A1.size(), B1.size()
    nval, _, _ = A2.size()

    if 2*b*nval*t**2 < (b+nval)*p*d:

        #transpose_start = time.time()
        A2, B2 = A2.transpose(-1, -2), B2.transpose(-1, -2)
        #transpose_end = time.time()
        #print('Time for transpose: {}'.format(transpose_end - transpose_start))

        A1_expanded = A1.unsqueeze(1)
        A2_expanded = A2.unsqueeze(0)
        B1_expanded = B1.unsqueeze(1)
        B2_expanded = B2.unsqueeze(0)

        # expand_end = time.time()
        #print('Time for expand: {}'.format(expand_end - transpose_end))

        # Memory consumption: 2*b*nval*T^2
        # A_dotprod = torch.matmul(A1_expanded, A2_expanded) # Shape: [b, nval, T, T]
        # B_dotprod = torch.matmul(B1_expanded, B2_expanded) # Shape: [b, nval, T, T]
        A_dotprod = _chunked_matmul(A1_expanded, A2_expanded, chunk_size=4096)
        B_dotprod = _chunked_matmul(B1_expanded, B2_expanded, chunk_size=4096)

        # chunk_end = time.time()
        # print('Time for chunked matmul: {}'.format(chunk_end - expand_end))

        result = (A_dotprod * B_dotprod).sum(dim=(2, 3))
        #result_end = time.time()
        #print('Time for sum: {}'.format(result_end - chunk_end))

        return result

    else:

        # [b, p, T] * [b, T, d]
        A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1) # Shape: [b, p*d]
        B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1) # Shape: [nval, p*d]

        return torch.matmul(A, B.T)


def _chunked_matmul(A1, A2, chunk_size=128):
    """
    Performs matrix multiplication in chunks for memory efficiency.

    Parameters:
    A1 (torch.Tensor): The first tensor with shape [n1, c1, h1, w1]
    A2 (torch.Tensor): The second tensor with shape [n2, c2, w2, h2]
    chunk_size (int): The size of each chunk to be multiplied

    Returns:
    torch.Tensor: The result of the matrix multiplication with shape [n1, c2, h1, h2]
    """
    # Validate input shapes
    if A1.shape[-1] != A2.shape[-2]:
        raise ValueError(f"Inner dimensions must match for matrix multiplication, got {A1.shape[-1]} and {A2.shape[-2]}")

    # Determine output shape
    n1, c1, h1, w1 = A1.shape
    n2, c2, w2, h2 = A2.shape

    if w1 != w2:
        raise ValueError(f"Inner matrix dimensions must agree, got {w1} and {w2}")

    # Prepare the result tensor on the same device as the inputs
    result = torch.zeros(n1, c2, h1, h2, device=A1.device, dtype=A1.dtype)

    # Perform the multiplication in chunks
    for start in range(0, w1, chunk_size):
        end = min(start + chunk_size, w1)
        A1_chunk = A1[:, :, :, start:end]  # [8, 1, 1024, chunk_size]
        A2_chunk = A2[:, :, start:end, :]  # [1, 8, chunk_size, 1024]

        # Multiply the chunks
        result += torch.matmul(A1_chunk, A2_chunk)

    return result


def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        # Select the index with the highest score
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to a very large negative value
        # to ensure it's not selected again
        scores[idx_max] = -np.inf

    return selected_indices




