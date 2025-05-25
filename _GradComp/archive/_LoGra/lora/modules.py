import torch
import torch.nn as nn
import math

class LoraModule(nn.Module):
    """
    A LoRA module implementation for gradient projection.

    This module adds a three-layer LoRA structure to an existing module:
    1. An encoder that projects the input
    2. A bottleneck layer (initialized to zero)
    3. A decoder that projects back

    The bottleneck ensures the forward and backward passes remain unchanged,
    while allowing access to the projected gradients.
    """

    def __init__(
            self,
            base_module: nn.Module,
            rank: int = 64,
            init_method: str = "random"
        ):
        """
        Initialize the LoRA module.

        Args:
            base_module: The module to wrap with LoRA
            rank: The rank of the LoRA decomposition
            init_method: Initialization method ("random" or "pca")
        """
        super().__init__()

        self.base_module = base_module
        self.rank = rank
        self.init_method = init_method
        self.device = next(base_module.parameters()).device

        # Determine dimensions based on module type
        if isinstance(base_module, nn.Linear):
            in_features = base_module.in_features
            out_features = base_module.out_features

            self.has_bias = base_module.bias is not None # Previously, there's a bug here: bias is not taken into account

            # Create LoRA layers
            self.lora_A = nn.Linear(in_features, rank, bias=self.has_bias)  # Encoder
            self.lora_B = nn.Linear(rank, rank, bias=False)                 # Bottleneck
            self.lora_C = nn.Linear(rank, out_features, bias=False)         # Decoder

            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            if self.has_bias:
                fan_in = self.lora_A.weight.shape[1]  # Get fan_in
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.lora_A.bias, -bound, bound)
            nn.init.zeros_(self.lora_B.weight)
            nn.init.kaiming_uniform_(self.lora_C.weight, a=math.sqrt(5))

        elif isinstance(base_module, nn.Conv2d):
            in_channels = base_module.in_channels
            out_channels = base_module.out_channels
            kernel_size = base_module.kernel_size
            stride = base_module.stride
            padding = base_module.padding

            # Create LoRA layers
            self.lora_A = nn.Conv2d(
                in_channels, rank, kernel_size, stride, padding, bias=False
            )
            self.lora_B = nn.Conv2d(rank, rank, 1, bias=False)
            self.lora_C = nn.Conv2d(rank, out_channels, 1, bias=False)

            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)  # Initialize bottleneck to zero
            nn.init.kaiming_uniform_(self.lora_C.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"Unsupported module type: {type(base_module)}")

        # Move LoRA layers to the same device as base module
        self.lora_A = self.lora_A.to(self.device)
        self.lora_B = self.lora_B.to(self.device)
        self.lora_C = self.lora_C.to(self.device)

        # Store intermediate activations
        self.lora_B_output = None
        self.projected_input = None
        self.projected_grad_pre_activation = None
        self.projected_grad = None

    def forward(self, x):
        """
        Forward pass through the LoRA module.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Base module forward pass
        base_output = self.base_module(x)

        # LoRA forward pass
        self.projected_input = self.lora_A(x)
        self.lora_B_output = self.lora_B(self.projected_input)
        self.lora_B_output.retain_grad()
        lora_C_output = self.lora_C(self.lora_B_output)

        return base_output + lora_C_output

    def update_projected_grad(self):
        """Update the projected gradient using the output of lora_B."""
        if self.lora_B_output.grad is not None:
            self.projected_grad_pre_activation = self.lora_B_output.grad
            bsz = self.projected_input.shape[0]
            self.projected_grad = torch.einsum('ijk,ijl->ikl', self.projected_grad_pre_activation, self.projected_input).reshape(bsz, -1)

    def init_pca(self, covariance_data):
        """
        Initialize LoRA weights using PCA.

        Args:
            covariance_data: Covariance data for PCA initialization
        """
        if "forward" not in covariance_data or "backward" not in covariance_data:
            raise ValueError("Covariance data must contain 'forward' and 'backward' keys")

        # Compute SVD of forward covariance
        fwd_cov = covariance_data["forward"]
        U_fwd, S_fwd, _ = torch.svd(fwd_cov)
        top_fwd_vectors = U_fwd[:, :self.rank]

        # Compute SVD of backward covariance
        bwd_cov = covariance_data["backward"]
        U_bwd, S_bwd, _ = torch.svd(bwd_cov)
        top_bwd_vectors = U_bwd[:, :self.rank]

        # Update LoRA weights
        if isinstance(self.base_module, nn.Linear):
            self.lora_A.weight.data.copy_(top_fwd_vectors.t())
            self.lora_C.weight.data.copy_(top_bwd_vectors)
        elif isinstance(self.base_module, nn.Conv2d):
            # Reshape for convolutional layers
            self.lora_A.weight.data.copy_(
                top_fwd_vectors.t().view(self.lora_A.weight.shape)
            )
            self.lora_C.weight.data.copy_(
                top_bwd_vectors.view(self.lora_C.weight.shape)
            )