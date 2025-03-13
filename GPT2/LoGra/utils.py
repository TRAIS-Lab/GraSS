import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from GPT2.GC.utlis import transpose_Conv1D

def replace_conv1d_modules(model):
    # GPT-2 is defined in terms of Conv1D. However, this does not work for EK-FAC.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)

def construct_model(resume=False):
    config = AutoConfig.from_pretrained(
        "openai-community/gpt2",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    replace_conv1d_modules(model)
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2", use_fast=True, trust_remote_code=True
    )
    if resume:
        original_model = GPT2LMHeadModel.from_pretrained("../checkpoints/wd=0.0_lr=5e-5/0/")
        state_dict = transpose_Conv1D(original_model.state_dict())
        for key in list(state_dict.keys()):
            if "model." in key:
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)
    return model, tokenizer