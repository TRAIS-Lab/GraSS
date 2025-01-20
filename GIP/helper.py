import torch
from .layers.linear import GIPLinear, GIPEmbedding
from .layers.layer_norm import GIPLayerNorm
from tqdm import tqdm
from _dattri.func.projection import random_project

def find_GIPlayers(model):
    GIP_layers = []

    for module in model.modules():
        if isinstance(module, GIPLinear) or isinstance(module, GIPLayerNorm) or isinstance(module, GIPEmbedding):
            GIP_layers.append(module)

    return GIP_layers

def grad_dotprod(A1, B1, A2, B2) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a GIPlinear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        dot_prod_1 = torch.matmul(A1, A2.T)
        dot_prod_2 = torch.matmul(B1, B2.T)
        dot_prod = dot_prod_1*dot_prod_2

        return dot_prod
    elif A1.dim() == 3 and B1.dim() == 3:
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
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")

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

def Ghost_Inner_Product(model, train_dataloader, test_dataloader, layer_name, projector_kwargs=None, lr=1e-3, batch=False, device='cuda'):
    """
    Adapted version of Ghost_Inner_Product to work with DataLoader format
    """
    if projector_kwargs is not None:
        threshold = projector_kwargs.get("threshold", None)
        seed = projector_kwargs.get("proj_seed", 0)
        projector_kwargs.pop("threshold")
        projector_kwargs.pop("proj_seed")

    if batch:
        # Process in batches
        grad_dot = None  # Accumulate gradient dot products batch-by-batch

        for train_batch in tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ):
            train_input_ids = train_batch["input_ids"].to(device)
            train_attention_masks = train_batch["attention_mask"].to(device)
            train_labels = train_batch["labels"].to(device)

            for eval_batch in tqdm(
                    test_dataloader,
                    desc="calculating gradient of evaluation set...",
                    leave=False,
                ):
                eval_input_ids = eval_batch["input_ids"].to(device)
                eval_attention_masks = eval_batch["attention_mask"].to(device)
                eval_labels = eval_batch["labels"].to(device)

                # Combine train and eval data for this batch
                combined_input_ids = torch.cat((train_input_ids, eval_input_ids), dim=0)
                combined_attention_masks = torch.cat((train_attention_masks, eval_attention_masks), dim=0)
                combined_labels = torch.cat((train_labels, eval_labels), dim=0)

                # Forward pass
                outputs = model(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_masks,
                    labels=combined_labels
                )
                full_loss = outputs.loss

                # Get pre-activations from trainable layers
                full_pre_acts = [layer.pre_activation for layer in layer_name]

                # Calculate gradients
                Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

                # Calculate scores for this batch
                batch_grad_dot = torch.zeros(train_input_ids.size(0), eval_input_ids.size(0), device=device)
                with torch.no_grad():
                    for layer_id, (layer, z_grad_full) in enumerate(zip(layer_name, Z_grad_full)):
                        val_1, val_2 = layer.pe_grad_gradcomp(z_grad_full, per_sample=True)
                        if projector_kwargs is not None:
                            # Apply threshold is specified
                            if threshold is not None:
                                val_1 = torch.where(val_1.abs() > threshold, val_1, torch.zeros_like(val_1))
                                val_2 = torch.where(val_2.abs() > threshold, val_2, torch.zeros_like(val_2))

                            # input projector
                            random_project_1 = random_project(
                                val_1,
                                val_1.shape[0],
                                proj_seed=seed + int(1e4) * layer_id,
                                **projector_kwargs,
                            )
                            # output_grad projector
                            random_project_2 = random_project(
                                val_2,
                                val_2.shape[0],
                                proj_seed=seed + int(1e4) * layer_id + 1,
                                **projector_kwargs,
                            )

                            val_1 = random_project_1(val_1)
                            val_2 = random_project_2(val_2)
                        dLdZ_train, z_train = val_1[:train_input_ids.size(0)], val_2[:train_input_ids.size(0)]
                        dLdZ_val, z_val = val_1[train_input_ids.size(0):], val_2[train_input_ids.size(0):]
                        result = grad_dotprod(dLdZ_train, z_train, dLdZ_val, z_val) * lr

                        batch_grad_dot += result

                # Accumulate results
                if grad_dot is None:
                    grad_dot = batch_grad_dot
                else:
                    grad_dot += batch_grad_dot

        return grad_dot

    else:
        # Process all data at once
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
        full_pre_acts = [layer.pre_activation for layer in layer_name]

        # Calculate gradients
        Z_grad_full = torch.autograd.grad(full_loss, full_pre_acts, retain_graph=True)

        grad_dot = torch.zeros(batch_size, n_val, device=device)

        # Calculate scores
        with torch.no_grad():
            for layer_id, (layer, z_grad_full) in enumerate(zip(layer_name, Z_grad_full)):
                val_1, val_2 = layer.pe_grad_gradcomp(z_grad_full, per_sample=True)
                if projector_kwargs is not None:
                    # Apply threshold is specified
                    if threshold is not None:
                        val_1 = torch.where(val_1.abs() > threshold, val_1, torch.zeros_like(val_1))
                        val_2 = torch.where(val_2.abs() > threshold, val_2, torch.zeros_like(val_2))

                    # input projector
                    random_project_1 = random_project(
                        val_1,
                        val_1.shape[0],
                        proj_seed=seed + int(1e4) * layer_id,
                        **projector_kwargs,
                    )
                    # output_grad projector
                    random_project_2 = random_project(
                        val_2,
                        val_2.shape[0],
                        proj_seed= seed + int(1e4) * layer_id + 1,
                        **projector_kwargs,
                    )

                    val_1 = random_project_1(val_1)
                    val_2 = random_project_2(val_2)
                dLdZ_train, z_train = val_1[:batch_size], val_2[:batch_size]
                dLdZ_val, z_val = val_1[batch_size:], val_2[batch_size:]
                result = grad_dotprod(dLdZ_train, z_train, dLdZ_val, z_val) * lr
                grad_dot += result
        return grad_dot