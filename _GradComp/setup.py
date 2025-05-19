"""
GradComp: Efficient gradient computation and influence attribution for PyTorch models
"""

from setuptools import setup, find_packages

setup(
    name="gradcomp",
    version="0.1.0",
    description="Efficient gradient computation and influence attribution for PyTorch models",
    author="GradComp Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)