# GraSS 🌿

This is the official implementation of [GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](https://arxiv.org/abs/2505.18976).

## Setup Guide

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n GraSS python=3.10
conda activate GraSS

conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install sjlt --no-build-isolation
pip install fast_jl --no-build-isolation

pip install -r requirements.txt
```

> [!Note]
> The two projection CUDA kernels, [`sjlt`](https://github.com/TRAIS-Lab/sjlt/tree/main) and [`fast_jl`](https://pypi.org/project/fast-jl/), sometimes can be tricky to install due to the CUDA version mismatch. If you encounter any issues during their installation, please refer to [`sjlt`](https://github.com/TRAIS-Lab/sjlt/tree/main)'s repository for troubleshooting.

## File Structure

The repository is organized as follows:

### Libraries

1. `_dattri/`: A local copy of the [dattri](https://github.com/TRAIS-Lab/dattri) library (based on commit [`7576361`](https://github.com/TRAIS-Lab/dattri/commit/7576361284cc4dcce35248bbb4ef0ae69e99041f)) with **GraSS**, **FactGraSS**, and **BlockProjectedIFAttributor** implementations (no separate installation required).
   - Noticeable modifications include `fast_jl` and Selective Mask supports. where the former is now removed from dattri, while the latter is specific to this project.
2. `_SelectiveMask/`: The implementation of **Selective Mask** for learning important gradient dimensions.
3. `_LogIX/`: The [LogIX](https://github.com/logix-project/logix) library for cross-validation (comparison baseline).
   - Some customization is done to support the newest Huggingface Trainer API and other needs of this project.

### Experiments

- `MLP_MNIST/`: Small-scale MLP on MNIST experiment
- `ResNet_CIFAR/`: Medium-scale ResNet9 on CIFAR-2 experiment
- `MusicTransformer_MAESTRO/`: MusicTransformer on MAESTRO experiment
- `GPT2_wikitext/`: GPT-2 on WikiText-2 experiment
- `Llama3_8B_OWT/`: Llama3-8B on OpenWebText experiment

## API Reference

### Projection Types

The following projection types are supported (use lowercase in command line arguments):

| Type          | Description                            |
| ------------- | -------------------------------------- |
| `normal`      | Gaussian random projection             |
| `rademacher`  | Rademacher random projection           |
| `sjlt`        | Sparse Johnson-Lindenstrauss Transform |
| `fjlt`        | Fast Johnson-Lindenstrauss Transform   |
| `random_mask` | Random feature mask selection          |
| `grass`       | GraSS projection                       |
| `identity`    | No projection                          |

### Hessian Types (for Influence Function)

| Type       | Description                              |
| ---------- | ---------------------------------------- |
| `eFIM`     | Empirical Fisher Information Matrix      |
| `Identity` | No Hessian approximation (raw gradients) |

### Command Line Format

```bash
# Sparsification (Stage 1): TYPE-DIM*DIM for factorized
--sparsification random_mask-128*128

# Projection (Stage 2): TYPE-DIMENSION
--projection sjlt-4096

# Hessian type for Influence Function
--hessian eFIM   # or Identity
```

## Quick Start

Each experiment folder contains a `job/` directory with SLURM job scripts. These scripts provide complete examples for running the experiments.

### Small/Medium-Scale Experiments

For **MLP+MNIST**, **ResNet+CIFAR**, and **MusicTransformer+MAESTRO**, the LDS ground truth and models are provided by dattri. See the `job/` folder in each experiment directory:
- `score.slurm`: Main attribution experiments
- `selective_mask.slurm`: Optional SelectiveMask training

### GPT2+Wikitext

GPT2 requires training multiple models for LDS ground truth computation. See `GPT2_wikitext/job/`:
- `train.slurm`: Fine-tune 50 models with different random subsets
- `groundtruth.slurm`: Compute LDS ground truth
- `selective_mask.slurm`: Optional SelectiveMask training
- `score.slurm`: Main attribution experiments

Example compression configurations:
- **FactGraSS**: `--sparsification random_mask-128*128 --projection sjlt-4096`
- **LoGra**: `--sparsification normal-64*64 --projection identity`

### Llama3-8B+OpenWebText

For billion-scale models, no LDS ground truth is computed. The attribution pipeline uses phased execution. See `Llama3_8B_OWT/job/`:
- `cache.slurm`: Cache gradients with SLURM array for parallelization (automatically computes preconditioners and IFVP when hessian="eFIM")
- `attribute.slurm`: Final attribution computation
- `selective_mask.slurm`: Optional SelectiveMask training

> [!Note]
> The `cache` mode automatically runs `compute_preconditioners()` and `compute_ifvp()` when using `--hessian eFIM`. The `--worker` argument enables parallelization for large-scale caching.

## Citation

If you find this repository valuable, please give it a star! Got any questions or feedback? Feel free to open an issue. Using this in your work? Please reference us using the provided citation:

```bibtex
@inproceedings{hu2025grass,
  author    = {Pingbang Hu and Joseph Melkonian and Weijing Tang and Han Zhao and Jiaqi W. Ma},
  title     = {GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
