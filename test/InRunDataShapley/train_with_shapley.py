import os
import time
import math
import pickle
from contextlib import nullcontext
import gc

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from modelgc import GPTConfig, GPT
from layers.helper import *
from utils import *
import tiktoken

import json
import argparse
from tqdm import tqdm

from dataloader import load_all_data, get_batch_from_dataset, get_batch_subdomain



parser = argparse.ArgumentParser(description='In-Run Data Shapley score computation.')

parser.add_argument('--method', type=str, default='Regular')

# Data parameters
parser.add_argument('--block_size', type=int, default=1024)

# Training parameters
parser.add_argument('--batch_size', type=int, default=16, help='Batch size in each backward pass')
parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--warmup_step', type=int, default=2000)
parser.add_argument('--learning_rate', type=float, default=6e-4) # max learning rate
parser.add_argument('--optimizer', type=str, default='adam')

# Validation parameters
parser.add_argument('--train_set', type=str, default='pile')
parser.add_argument('--val_set', type=str, default='EuroParl')

# Misc parameters
parser.add_argument('--max_steps', type=int, default=600000)
parser.add_argument('--seed', type=int, default=42)

# Eval parameters
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--eval_interval', type=int, default=200)
parser.add_argument('--eval_iter', type=int, default=100)
parser.add_argument('--eval_bs', type=int, default=16)

args = parser.parse_args()

method = args.method
batch_size = args.batch_size
val_batch_size = args.val_batch_size
learning_rate = args.learning_rate
min_lr = learning_rate * 0.1 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
block_size = args.block_size # sequence length


# -----------------------------------------------------------------------------
# Setup result folder and file

result_folder_dir = '/scratch/gpfs/tw8948/ContPretrain/nanogpt/{}_{}_result/'.format(
    args.train_set, args.val_set)


if not os.path.exists(result_folder_dir):
    os.makedirs(result_folder_dir)
    print(f"Directory '{result_folder_dir}' was created.")
else:
    print(f"Directory '{result_folder_dir}' already exists.")

# Setup result file name
result_dir = result_folder_dir + '{}-BS{}-ValBS{}-LR{}-Warmup{}'.format(
    method, args.batch_size, args.val_batch_size, 
    args.learning_rate, args.warmup_step)

result_dir = result_dir + '_{}'.format(args.optimizer)
result_dir = result_dir + '_Seed{}'.format(args.seed)

# Create the result directory for storing training logs
train_result_dir = result_dir + '_results.json'


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText

# Now it means the optimizer is being re-initialized for finetuning. 
init_from = 'scratch'

# data setting
# currently we do not support gradient accumulation for code readability, but it's easy to add
args.full_batch_size = args.batch_size
gradient_accumulation_steps = int(args.full_batch_size / args.batch_size) # used to simulate larger batch sizes

if gradient_accumulation_steps > 1:
    print('In this version of code, when using gradient accumulation steps, 2nd-order In-Run Data Shapley will be only an approximation.')

# model (GPT2-Small)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = args.warmup_step
lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla

# optimizer
if args.optimizer == 'sgd':
    use_sgd = True
else:
    use_sgd = False

# learning rate decay settings
decay_lr = True # whether to decay the learning rate

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

eval_iters = args.eval_iter



# # -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# # -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Set random seed
set_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Note: torch.autocast is a PyTorch context manager that allows you to perform operations in mixed precision.
# This may cause deviation of gradient dot-product calculation. 
# Therefore it's better to use float32 for the gradient calculation.
ctx = nullcontext() # if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



# load the data
dataset = load_all_data()
print('Pile dataset loaded')

# get a batch of training data. 
def get_batch(split, batch_size, return_idx=False):
    return get_batch_from_dataset(split, batch_size, dataset, return_idx=return_idx)

# get a batch of validation data, can change to arbitrary data here. 
def get_val_batch(batch_size, domain_name, return_idx=False, return_first=False):
    return get_batch_subdomain('val', batch_size, domain_name, return_idx=return_idx, return_first=return_first) 



# init these up here
iter_num = 0
best_val_loss = 1e9


# model init
meta_vocab_size = None
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, 
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
gptconf.use_flash = True
model = GPT(gptconf)


##### replace the layer here #### 
model.to(device)
trainable_layers = find_GClayers(model)
trainable_layers = trainable_layers[1:] # remove the first layer for memory efficiency

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, sgd=use_sgd)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])




# -----------------------------------------------------------------------------
# helper functions for evaluation and training

@torch.no_grad()
def estimate_loss(eval_iters=eval_iters, eval_bs=args.eval_bs):
    model.eval()

    out = {}
    for split in ['train', 'val', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size=eval_bs)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)





# -----------------------------------------------------------------------------
# training loop

value_record = {'index': [], 
                'First-order In-Run Data Shapley': [], 
                'Second-order In-Run Data Shapley': []}

timing = True

if timing:
    time_lst = []

raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


# # Speed up moving from GPU to CPU by warming up.
# time_warm = time.time()
# torch.cuda.init()
# dummy = torch.zeros(1).cuda()
# _ = dummy.cpu()
# print('Time for warming up: {}'.format(time.time() - time_warm))


# get validation data
X_val, Y_val = get_val_batch(val_batch_size, args.val_set)


while True:

    t0 = time.time()

    X, Y, batch_idx = get_batch('train', batch_size=batch_size, return_idx=True)

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    print('Learning Rate: {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0 and not timing:

        losses = estimate_loss()
        train_loss, val_loss, test_loss = losses['train'], losses['val'], losses['test']
        print(f"step {iter_num}: train-FT loss {train_loss:.4f}, val-FT loss {val_loss:.4f}, test-FT loss {test_loss:.4f}")

        #### Save Training Results ####
        file_path = train_result_dir

        # Read the existing record, if available
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r") as file:
                record = json.load(file)
        else:
            record = []

        new_entry = {
            "train_loss": train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss,
            "eval_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
            "test_loss": test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss,
            "step": iter_num
        }

        # Append the new record entry to the list
        record.append(new_entry)

        # Write the updated record back to the file
        with open(file_path, "w") as file:
            json.dump(record, file, indent=4)

        pickle.dump( value_record, open(file_path+'.value', 'wb') )

    if args.eval_only:
        print('Eval only mode, exiting now')
        sys.exit(0)


    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:

            if args.method == 'Regular':
                logits, loss = model(X, Y)

            elif args.method == 'In-Run Data Shapley':

                # Note: second order interaction here is an approximation
                # which uses Identity matrix to approximate the Hessian matrix
                # The implementation here is different from the paper, but the results and speed are similar.
                first_order_score, second_order_interaction = compute_value_per_iter_inrun(
                        model, device=device, train_data=(X, Y), 
                        val_data=(X_val, Y_val), optimizer=optimizer, 
                        trainable_layers=trainable_layers, 
                        return_tracin_and_similarity=True, 
                        return_val_similarity=False)

                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(X, Y)

            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # accumulate gradients over microsteps, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)  # Unscale gradients first if using mixed precision

    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Gradient Update
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0

    if timing:
        time_lst.append(dt)
        print('Average Time: {}'.format( np.mean(time_lst) ))
    print('Time Iter {}: {}'.format( iter_num, dt ))

    iter_num += 1

    # Note: the following code moves the tensors to CPU for computing the In-Run Data Shapley values.
    # Moving between GPU and CPU is time-consuming.
    # If you want to speed up the computation, you can modify the code by only moving the tensors to CPU at the end of the training
    # or every few iterations.
    if method == 'In-Run Data Shapley':

        move_to_cpu_start = time.time()
        first_order_score = first_order_score.cpu().numpy()
        second_order_interaction = second_order_interaction.cpu().numpy()

        if args.optimizer == 'adam':

            # For Adam optimizer, we need to normalize the values by the norm of the gradients.
            norm = np.sqrt( np.diag(second_order_interaction) + 1e-8 )
            first_order_score = first_order_score / norm
            second_order_interaction = second_order_interaction / norm
            second_order_interaction = second_order_interaction / norm[:, np.newaxis]

            # Note: the following code omit a lr factor for both order values.
            first_order_value = first_order_score
            second_order_value = first_order_value - np.sum(second_order_interaction, axis=1) * lr / 2
        else:
            first_order_value = first_order_score
            second_order_value = first_order_value - np.sum(second_order_interaction, axis=1) * lr / 2

        print('First-order In-Run Data Shapley: {}'.format(first_order_value))
        print('Second-order In-Run Data Shapley: {}'.format(second_order_value))
        
        value_record['index'].append(batch_idx)
        value_record['First-order In-Run Data Shapley'].append(first_order_value)
        value_record['Second-order In-Run Data Shapley'].append(second_order_value)

        move_to_cpu_end = time.time()
        print('Time for moving to CPU: {}'.format(move_to_cpu_end - move_to_cpu_start))