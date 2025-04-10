# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sjlt_cuda",
    ext_modules=[
        CUDAExtension(
            name="sjlt_cuda_ext",  # This will be the name of the imported module
            sources=["SJLT_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)