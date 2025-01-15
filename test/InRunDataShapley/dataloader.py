import os, sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from modelgc import GPTConfig, GPT
import tiktoken
from datasets import load_from_disk

import json
import argparse
from tqdm import tqdm
import numpy as np

import cvxpy as cp

from domain_list import PILE_DOMAIN_LIST, FORGET_DOMAIN_RANK


class DataSizeError(Exception):
    pass


def load_and_sample_data(domain_proportions, domain_of_interest):
    # Calculate total size of data from domain_of_interest for train, val, and test
    train_data_of_interest = np.memmap(f'/scratch/gpfs/tw8948/LESS/pile-6m/train-{domain_of_interest}.bin', dtype=np.uint16, mode='r')
    val_data_of_interest = np.memmap(f'/scratch/gpfs/tw8948/LESS/pile-6m/validation-{domain_of_interest}.bin', dtype=np.uint16, mode='r')
    test_data_of_interest = np.memmap(f'/scratch/gpfs/tw8948/LESS/pile-6m/test-{domain_of_interest}.bin', dtype=np.uint16, mode='r')

    total_size_of_interest_train = len(train_data_of_interest)
    total_size_of_interest_val = len(val_data_of_interest)
    total_size_of_interest_test = len(test_data_of_interest)

    # Determine the size of data from each domain based on the proportions
    num_samples_train = total_size_of_interest_train / domain_proportions[domain_of_interest]
    num_samples_val = total_size_of_interest_val / domain_proportions[domain_of_interest]
    num_samples_test = total_size_of_interest_test / domain_proportions[domain_of_interest]

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    for domain, proportion in domain_proportions.items():
        # Define file paths
        finetune_traindata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/train-{domain}.bin'
        finetune_valdata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/validation-{domain}.bin'
        finetune_testdata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/test-{domain}.bin'
        
        # Load data
        train_data = np.memmap(finetune_traindata_dir, dtype=np.uint16, mode='r')
        val_data = np.memmap(finetune_valdata_dir, dtype=np.uint16, mode='r')
        test_data = np.memmap(finetune_testdata_dir, dtype=np.uint16, mode='r')

        # Calculate the number of samples to take from each domain
        samples_train = int(num_samples_train * proportion)
        samples_val = int(num_samples_val * proportion)
        samples_test = int(num_samples_test * proportion)

        sampled_train_data = train_data[:samples_train]
        sampled_val_data = val_data[:samples_val]
        sampled_test_data = test_data[:samples_test]

        # Append to mixed data lists
        mixed_train_data.append(sampled_train_data)
        mixed_val_data.append(sampled_val_data)
        mixed_test_data.append(sampled_test_data)
    
    # Concatenate all samples to form the final mixed datasets
    mixed_train_data = np.concatenate(mixed_train_data)
    mixed_val_data = np.concatenate(mixed_val_data)
    mixed_test_data = np.concatenate(mixed_test_data)

    dataset = {'train': mixed_train_data, 
               'val': mixed_val_data, 
               'test': mixed_test_data}
    
    return dataset





def load_all_data():

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    for domain in PILE_DOMAIN_LIST:

        # Define file paths
        finetune_traindata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/train-{domain}.bin'
        finetune_valdata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/validation-{domain}.bin'
        finetune_testdata_dir = f'/scratch/gpfs/tw8948/LESS/pile-6m/test-{domain}.bin'
        
        # Load data
        train_data = np.memmap(finetune_traindata_dir, dtype=np.uint16, mode='r')
        val_data = np.memmap(finetune_valdata_dir, dtype=np.uint16, mode='r')
        test_data = np.memmap(finetune_testdata_dir, dtype=np.uint16, mode='r')

        # Append to mixed data lists
        mixed_train_data.append(train_data)
        mixed_val_data.append(val_data)
        mixed_test_data.append(test_data)
    
    # Concatenate all samples to form the final mixed datasets
    mixed_train_data = np.concatenate(mixed_train_data)
    mixed_val_data = np.concatenate(mixed_val_data)
    mixed_test_data = np.concatenate(mixed_test_data)

    dataset = {'train': mixed_train_data, 
               'val': mixed_val_data, 
               'test': mixed_test_data}
    
    return dataset



def get_batch_from_dataset(split, batch_size, dataset, 
              block_size=1024, device='cuda', device_type='cuda', 
              i_iter=-1, order_lst=None, return_idx=False, return_first=False):

    data = dataset[split]
    
    if len(data) - block_size == 0:
        ix = [0]
    elif return_first:
        ix = [0]
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    if return_idx:
        return x, y, ix
    else:
        return x, y


def get_batch_subdomain(split, batch_size, domain_name, 
                        return_idx=False, return_first=False, 
                        block_size=1024, device='cuda', device_type='cuda'):

    traindata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/train-{}.bin'.format(domain_name)
    valdata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/validation-{}.bin'.format(domain_name)
    testdata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/test-{}.bin'.format(domain_name)

    if split == 'train':
        data = np.memmap(traindata_dir, dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(valdata_dir, dtype=np.uint16, mode='r')
    elif split == 'test':
        data = np.memmap(testdata_dir, dtype=np.uint16, mode='r')
    else:
        raise ValueError(f"Invalid split: {split}")
    
    if len(data) - block_size == 0:
        ix = [0]
    elif return_first:
        ix = [0]
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    if return_idx:
        return x, y, ix
    else:
        return x, y


# Get Pile-CC batch
def get_REST_batch(split, batch_size, return_idx=False, return_first=False, block_size=1024, device='cuda', device_type='cuda'):
    return get_batch_subdomain(split, batch_size, 'Pile-CC', return_idx, return_first, block_size, device, device_type)


def get_domain_proportions(args):

    assert len( args.domain_name.split('_') ) == 4

    # Extracting domain information
    domain_of_interest = args.domain_name.split('_')[0]
    mode_of_other = args.domain_name.split('_')[1]
    number_of_other = int(args.domain_name.split('_')[2])
    ratio_of_other = float(args.domain_name.split('_')[3])

    domain_proportions = {}
    domain_proportions[domain_of_interest] = 1 - ratio_of_other

    other_domain_lists = []
    if mode_of_other in PILE_DOMAIN_LIST:
        other_domain_lists = [mode_of_other]
    else:
        other_domain_lists = FORGET_DOMAIN_RANK[2:2+number_of_other]

    if mode_of_other in ['Uniform']:
        for domain_name in other_domain_lists:
            domain_proportions[domain_name] = ratio_of_other / number_of_other
    elif mode_of_other == 'Solve':
        alpha = solve_optimal_alpha(domain_of_interest, 
                                    other_domain_lists, 
                                    ratio_of_other, 
                                    log_barrier_coef=args.log_barrier_coef)
        for i, domain_name in enumerate(other_domain_lists):
            domain_proportions[domain_name] = alpha[i+1]
    elif mode_of_other in PILE_DOMAIN_LIST:
        domain_name = mode_of_other
        domain_proportions[domain_name] = ratio_of_other
    else:
        print('Do not support the current mode')
        sys.exit(1)

    return domain_of_interest, other_domain_lists, domain_proportions


def extract_key_value(file_path, key, s=1000):

  # Open the text file containing the data
  with open(file_path, 'r') as file:
      # Read all lines from the file
      lines = file.readlines()

  # Initialize an empty list to store test_loss values
  test_loss_values = []

  key = '"'+key+'":'

  count = 0

  # Iterate through each line
  for line in lines:
      # Check if 'test_loss' is in the line
      if key in line:
          count += 1
          # Find the start index of the numeric value following 'test_loss':
          start_index = line.find(key) + len(key)
          # Extract the substring containing the number
          numeric_part = line[start_index:].split(',')[0].strip()
          # Convert to float and append to the list
          test_loss_values.append(float(numeric_part))
          if count > s:
            break

  # Now you have a list of all test_loss values
  return test_loss_values


def solve_optimal_alpha(domain_of_interest, other_domain_lists, ratio_of_other, log_barrier_coef=0.1):

    all_domains = [domain_of_interest] + other_domain_lists

    dir_first_order = '/scratch/gpfs/tw8948/ContPretrain/nanogpt/pile_Pile-CC_Every_result/'
    dir_first_order = dir_first_order + 'Regular-BS1600-ValBS16-LR0.0006-Warmup2000_finetune_Npre500000_adam_Seed42_results.json.TaylerDist'
    first_order_coef = []
    for domain in all_domains:
        dot_prod_lst = extract_key_value(dir_first_order, "Pile-CC-{}-dot".format(domain))
        first_order_coef.append(np.mean(dot_prod_lst))

    second_order_coef = []
    for domain_1 in all_domains:
        dir_second_order = '/scratch/gpfs/tw8948/ContPretrain/nanogpt/pile_{}_Every_result/'.format(domain_1)
        dir_second_order = dir_second_order + 'Regular-BS1600-ValBS16-LR0.0006-Warmup2000_finetune_Npre500000_adam_Seed42_results.json.TaylerDist'
        for domain_2 in all_domains:
            vHv_lst = extract_key_value(dir_second_order, "{}-{}-vHv".format(domain_1, domain_2))
            second_order_coef.append(np.mean(vHv_lst))

    first_order_coef = np.array(first_order_coef)
    second_order_coef = np.array(second_order_coef).reshape(len(all_domains), len(all_domains))
    second_order_coef = 0.5 * (second_order_coef + second_order_coef.T)

    print('First order coef:')
    print(first_order_coef)

    print('Second order coef:')
    print(second_order_coef)

    # Define the optimization variables
    alpha = cp.Variable(len(all_domains))

    # Define the constraints
    constraints = [
        alpha[0] == 1 - ratio_of_other,
        cp.sum(alpha) == 1,
        alpha >= 1e-5
    ]

    # Define the objective function with log-barrier term
    log_barrier = -cp.sum(cp.log(alpha[1:]))  # Penalize values close to zero, exclude the first variable

    for coef in [1e-3]:
        objective = cp.Minimize( - first_order_coef.T @ alpha + coef * cp.quad_form(alpha, second_order_coef) + log_barrier_coef * log_barrier )

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        result = problem.solve()

        print("Optimal value:", result)
        print("Optimal solution:", alpha.value)

    return alpha.value



def solve_optimal_weights(first_order_coef, second_order_coef, log_barrier_coef=0.1):

    n_variables = len(first_order_coef)

    first_order_coef = np.array(first_order_coef)
    second_order_coef = np.array(second_order_coef)

    # Make sure the second order matrix is symmetric
    second_order_coef = 0.5 * (second_order_coef + second_order_coef.T)

    # print('First order coef:')
    # print(first_order_coef)

    # print('Second order coef:')
    # print(second_order_coef)

    # Define the optimization variables
    alpha = cp.Variable(n_variables)

    # Define the constraints
    # add lower and upper bound for alpha to make sure it is not too small or too large
    # lack of diversity can significantly affect the performance
    constraints = [
        cp.sum(alpha) == 1,
        # alpha >= 0.01,
        # alpha <= 0.1
    ]

    # Define the objective function with log-barrier term
    log_barrier = -cp.sum(cp.log(alpha))  # Penalize values close to zero, exclude the first variable

    for coef in [1e-4]:
        objective = cp.Minimize( - first_order_coef.T @ alpha + coef * cp.quad_form(alpha, second_order_coef) + log_barrier_coef * log_barrier )

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        result = problem.solve()

        print("Optimal value:", result)
        print("Optimal solution:", alpha.value)

    return alpha.value




def solve_zero_weights(first_order_coef, second_order_coef, log_barrier_coef=0.1):

    n_variables = len(first_order_coef)

    first_order_coef = np.array(first_order_coef)
    second_order_coef = np.array(second_order_coef)

    # Make sure the second order matrix is symmetric
    second_order_coef = 0.5 * (second_order_coef + second_order_coef.T)

    # print('First order coef:')
    # print(first_order_coef)

    # print('Second order coef:')
    # print(second_order_coef)

    # Define the optimization variables
    alpha = cp.Variable(n_variables)

    coef = 1e-4

    # Define the constraints
    # add lower and upper bound for alpha to make sure it is not too small or too large
    # lack of diversity can significantly affect the performance
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0,
    ]

    print(first_order_coef)

    # make objective function the norm of first_order_coef.T @ alpha
    objective = cp.Minimize( cp.norm(first_order_coef.T @ alpha) )

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    print("Optimal value:", result)
    print("Optimal solution:", alpha.value)

    return alpha.value





# Do not include the first order coef in the optimization
def solve_optimal_weights_nofirst(first_order_coef, second_order_coef):

    n_variables = len(first_order_coef)

    first_order_coef = np.array(first_order_coef)
    second_order_coef = np.array(second_order_coef)

    # Make sure the second order matrix is symmetric
    second_order_coef = 0.5 * (second_order_coef + second_order_coef.T)

    # print('First order coef:')
    # print(first_order_coef)

    # print('Second order coef:')
    # print(second_order_coef)

    # Define the optimization variables
    alpha = cp.Variable(n_variables)

    # Define the constraints
    # add lower and upper bound for alpha to make sure it is not too small or too large
    # lack of diversity can significantly affect the performance
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0.01,
    ]

    # Define the objective function with log-barrier term
    log_barrier = -cp.sum(cp.log(alpha))  # Penalize values close to zero, exclude the first variable

    for coef in [1e-2]:
        objective = cp.Minimize( coef * cp.quad_form(alpha, second_order_coef) + 0 * log_barrier )

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        result = problem.solve()

        print("Optimal value:", result)
        print("Optimal solution:", alpha.value)

    return alpha.value



def get_MMLU_validation(split, subject, batch_size=1, return_idx=False, 
                        block_size=1024, device='cuda', device_type='cuda'):

    file_name = f"/scratch/gpfs/tw8948/LESS/mmlu_{subject}_{split}.json"

    with open(file_name, 'r') as file:
        data = json.load(file)

    indices = data['ids']

    if split == 'val':
        ix = [0]
        x = torch.stack([torch.from_numpy(np.array(indices[:-1]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.array(indices[1:]).astype(np.int64)) for i in ix])
    else:
        ix = np.random.randint(0, len(data['ids']) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(np.array(data['ids'][i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.array(data['ids'][i+1:i+1+block_size]).astype(np.int64)) for i in ix]) 

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y



