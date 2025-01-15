# Instruction for my own use

## Install

```
pip install torch numpy transformers datasets tiktoken tqdm
```
Dependencies:
- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `tqdm` for progress bars <3

## Data 

```
python data/prepare_pile.py
```

This may run about >24 hours, we can also just use a subset. 

## Running 

```
python train_with_shapley.py --method "In-Run Data Shapley" --batch_size 16 --val_batch_size 1 --seed 42
```

## Note for this version of code
- This is the most readable version of the code that does not implement the exact second-order In-Run Data Shapley (the result is almost identical). Here we simply use an identity matrix to approximate Hessian. 
- The implementation of score calculation is slightly different from the paper, the full optimization by reducing 2 backpropgations to 1 backpropagation in Appendix D.3 is not implemented, but the runtime is almost the same (and very close to regular training). 
- The last few lines of code in `train_with_shapley.py` moves the computed scores from GPU to CPU which could be slow. Can avoid this by only do the calculation every few iterations, but for readability purpose just use the current version. 
- Adding validation data size will increase the runtime (same for regular training if you increase training batch size). 

