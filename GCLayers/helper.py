import torch
from .linear import GCLinear, GCEmbedding
from .layer_norm import GCLayerNorm

def find_GClayers(model):
    GC_layers = []

    for module in model.modules():
        if isinstance(module, GCLinear) or isinstance(module, GCLayerNorm) or isinstance(module, GCEmbedding):
            GC_layers.append(module)

    return GC_layers

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

def Ghost_Inner_Product(model, train_dataloader, test_dataloader, trainable_layers, projector_kwargs=None, device='cuda'):
    """
    Adapted version of Ghost_Inner_Product to work with DataLoader format
    """
    # Collect all data from dataloaders
    train_input_ids, train_attention_masks, train_labels = [], [], []
    eval_input_ids, eval_attention_masks, eval_labels = [], [], []

    # Gather training data
    for batch in train_dataloader:
        train_input_ids.append(batch["input_ids"])
        train_attention_masks.append(batch["attention_mask"])
        train_labels.append(batch["labels"])

    # Gather evaluation data
    for batch in test_dataloader:
        eval_input_ids.append(batch["input_ids"])
        eval_attention_masks.append(batch["attention_mask"])
        eval_labels.append(batch["labels"])

    # Concatenate all batches
    train_input_ids = torch.cat(train_input_ids, dim=0).to(device)
    train_attention_masks = torch.cat(train_attention_masks, dim=0).to(device)
    train_labels = torch.cat(train_labels, dim=0).to(device)

    eval_input_ids = torch.cat(eval_input_ids, dim=0).to(device)
    eval_attention_masks = torch.cat(eval_attention_masks, dim=0).to(device)
    eval_labels = torch.cat(eval_labels, dim=0).to(device)

    batch_size = train_input_ids.shape[0]
    n_val = eval_input_ids.shape[0]

    # Combine all data
    combined_input_ids = torch.cat((train_input_ids, eval_input_ids), dim=0)
    combined_attention_masks = torch.cat((train_attention_masks, eval_attention_masks), dim=0)
    combined_labels = torch.cat((train_labels, eval_labels), dim=0)

    # Forward pass with all data
    outputs = model(
        input_ids=combined_input_ids,
        attention_mask=combined_attention_masks,
        labels=combined_labels
    )
    full_loss = outputs.loss

    # Get pre-activations from trainable layers
    full_pre_acts = [layer.pre_activation for layer in trainable_layers]

    # Calculate gradients
    Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

    dLdZ_a_val_lst = []
    dLdZ_a_train_lst = []


    if projector_kwargs is not None:
        threshold = projector_kwargs.get("threshold", None)
        projector_kwargs.pop("threshold")


    for layer, zgrad_full in zip(trainable_layers, Z_grad_full):
        decompose_result = layer.pe_grad_gradcomp(zgrad_full, per_sample=True)
        val_1, val_2 = decompose_result

        # Split for validation and training
        decompose_result_val = (val_1[batch_size:, :, :], val_2[batch_size:, :, :])
        dLdZ_a_val_lst.append(decompose_result_val)

        decompose_result_train = (val_1[:batch_size, :, :], val_2[:batch_size, :, :])
        dLdZ_a_train_lst.append(decompose_result_train)

    # Initialize scores
    grad_dot = torch.zeros(batch_size, n_val, device=device)
    # second_order_interaction = torch.zeros((batch_size, batch_size), device=device) if return_tracin_and_similarity else None
    # val_similarity_score = torch.zeros((n_val, n_val), device=device) if return_val_similarity else None

    # Calculate scores
    with torch.no_grad():
        for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):
            dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)
            grad_dot += dot_prod

            # dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            # second_order_interaction += dot_prod

            # dot_prod = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
            # val_similarity_score += dot_prod

    return grad_dot