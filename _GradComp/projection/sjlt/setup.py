from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.cuda
import logging

logger = logging.getLogger(__name__)

def get_cuda_arch_flags():
    """Get CUDA architecture flags based on detected GPU capability"""
    # Default architecture flags for older GPUs
    arch_flags = ["-gencode", "arch=compute_60,code=sm_60",
                  "-gencode", "arch=compute_70,code=sm_70"]

    # Check if we have CUDA capability >= 8.0 (Ampere and newer), which has better bfloat16 support
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                # Add Ampere (SM80) or newer architectures
                arch_flags += ["-gencode", "arch=compute_80,code=sm_80"]
            if major >= 9:  # H100/Hopper or newer
                arch_flags += ["-gencode", "arch=compute_90,code=sm_90"]
            logger.info(f"Detected CUDA capability: {major}.0")
        except:
            # If we can't detect the device capability, stick with the defaults
            logger.warning("Could not detect CUDA capability, using default architecture flags")
            pass

    return arch_flags

setup(
    name="sjlt_cuda",
    ext_modules=[
        CUDAExtension(
            name="sjlt_cuda_ext",  # This will be the name of the imported module
            sources=["SJLT_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    # Add extended-lambda flag for PyTorch headers
                    "--extended-lambda",
                    # Enable relaxed constexpr for bfloat16 support
                    "--expt-relaxed-constexpr",
                ] + get_cuda_arch_flags()
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)