# GradComp

Gradient computation and processing utilities.

## IO Module

The IO module provides efficient disk and memory-mapped operations:

- `ChunkedDiskIOManager`: Manages chunked gradient, preconditioner, and IFVP storage
- `ChunkedMemoryMapHandler`: Low-level memory-mapped file operations
- `ChunkedGradientDataset` and `ChunkedBatchDataset`: PyTorch datasets for loading chunked data

[Reference](https://github.com/ppmlguy/fastgradclip)