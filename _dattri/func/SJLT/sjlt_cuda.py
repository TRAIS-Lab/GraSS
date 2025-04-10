# sjlt_cuda.py
import os
import subprocess
import torch

# Check if the CUDA extension is already compiled
try:
    import sjlt_cuda_ext
except ImportError:
    # If not, compile it now
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Compiling CUDA extension in {current_dir}")
    subprocess.check_call(['pip', 'install', '-e', current_dir])
    import sjlt_cuda_ext

class SJLTProjection(torch.nn.Module):
    """Sparse Johnson-Lindenstrauss Transform implemented with CUDA kernels"""

    def __init__(self, original_dim, proj_dim, c=1, threads=1024, fixed_blocks=84, device='cuda'):
        """
        Initialize SJLT projection

        Args:
            original_dim: Original dimension of the input vectors
            proj_dim: Target projection dimension
            c: Number of non-zeros per column (sparsity parameter)
            device: Device to run the computation on
        """
        super(SJLTProjection, self).__init__()

        self.original_dim = original_dim
        self.proj_dim = proj_dim
        self.c = c
        self.threads = threads
        self.fixed_blocks = fixed_blocks
        self.device = device

        # Generate random indices and signs (these are fixed for the projection)
        self.register_buffer(
            'rand_indices',
            torch.randint(proj_dim, (original_dim, c), device=device)
        )

        self.register_buffer(
            'rand_signs',
            (torch.randint(0, 2, (original_dim, c), device=device) * 2 - 1).to(torch.int8)
        )

    def forward(self, x):
        """
        Apply SJLT projection to input tensor

        Args:
            x: Input tensor of shape [batch_size, original_dim]

        Returns:
            Projected tensor of shape [batch_size, proj_dim]
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Apply SJLT projection using CUDA kernel
        output = sjlt_cuda_ext.sjlt_projection_cuda(
            x,
            self.rand_indices,
            self.rand_signs,
            self.proj_dim,
            self.c,
            self.threads,
            self.fixed_blocks,
        )[0]

        return output