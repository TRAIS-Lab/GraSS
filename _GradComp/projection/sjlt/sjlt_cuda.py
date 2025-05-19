"""
SJLT CUDA implementation for sparse projections.
"""

import torch
import math
import os
import subprocess
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Check if the CUDA extension is already compiled
try:
    import sjlt_cuda_ext
    logger.info("SJLT CUDA extension loaded successfully")
except ImportError:
    # If not, compile it now
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Compiling SJLT CUDA extension in {current_dir}")
    try:
        subprocess.check_call(['pip', 'install', '-e', current_dir])
        import sjlt_cuda_ext
        logger.info("SJLT CUDA extension compiled and loaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile SJLT CUDA extension: {e}")
        logger.warning("SJLT projections will not be available")
    except ImportError as e:
        logger.error(f"Failed to import SJLT CUDA extension after compilation: {e}")
        logger.warning("SJLT projections will not be available")

class SJLTProjection(torch.nn.Module):
    """Sparse Johnson-Lindenstrauss Transform implemented with CUDA kernels"""
    def __init__(self, original_dim, proj_dim, c=1, threads=1024, fixed_blocks=84, device='cuda'):
        """
        Initialize SJLT projection
        Args:
            original_dim: Original dimension of the input vectors
            proj_dim: Target projection dimension
            c: Number of non-zeros per column (sparsity parameter)
            threads: Number of CUDA threads per block
            fixed_blocks: Number of CUDA blocks to use
            device: Device to run the computation on (e.g., 'cuda:0', 'cuda:1')
        """
        super(SJLTProjection, self).__init__()
        self.original_dim = original_dim
        self.proj_dim = proj_dim
        self.c = c

        # Ensure threads is a multiple of 32 (warp size) for optimal performance
        self.threads = (threads // 32) * 32

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

        logger.debug(f"Initialized SJLT projection with dimensions {original_dim} -> {proj_dim}")

    def forward(self, x):
        """
        Apply SJLT projection to input tensor
        Args:
            x: Input tensor of shape [batch_size, original_dim]
        Returns:
            Projected tensor of shape [batch_size, proj_dim]
        """
        # Check if SJLT is available
        if 'sjlt_cuda_ext' not in globals():
            raise ImportError("SJLT CUDA extension is not available")

        # Move input to the specified device if necessary
        if x.device != self.rand_indices.device:
            x = x.to(self.device)

        # Ensure indices and signs are on the same device as input
        # This is a safety check in case device context has changed
        rand_indices = self.rand_indices.to(x.device)
        rand_signs = self.rand_signs.to(x.device)

        # Apply SJLT projection using CUDA kernel
        output = sjlt_cuda_ext.sjlt_projection_cuda(
            x,
            rand_indices,
            rand_signs,
            self.proj_dim,
            self.c,
            self.threads,
            self.fixed_blocks,
        )[0]

        return output

    def extra_repr(self):
        """Extra information for string representation"""
        return f'original_dim={self.original_dim}, proj_dim={self.proj_dim}, c={self.c}'