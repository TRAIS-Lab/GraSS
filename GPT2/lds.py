import torch
import torch.nn as nn
import os

from tqdm import tqdm
import numpy as np

from scipy.stats import spearmanr
import csv

def read_nodes(file_path):
    int_list = []
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            for item in row:
                try:
                    int_list.append(int(item))
                except ValueError:
                    print(f"Warning: '{item}' could not be converted to an integer and was skipped.")
    return int_list

def calculate_one(score, training_setting):
    try:
        score = score.detach().cpu()
        # Prepare node list
        nodes_str = [f"./checkpoints/{training_setting}/{i}/train_index.csv" for i in range(50)]
        full_nodes = list(range(4656))
        node_list = []
        for node_str in nodes_str:
            numbers = read_nodes(node_str)
            index = [full_nodes.index(number) for number in numbers]
            node_list.append(index)

        # Load ground truth
        loss_list = torch.load(f"./results/{training_setting}/gt.pt", map_location=torch.device('cpu')).detach()

        # Calculate approximations
        approx_output = []
        for i in range(len(nodes_str)):
            score_approx_0 = score[node_list[i], :]
            sum_0 = torch.sum(score_approx_0, axis=0)
            approx_output.append(sum_0)

        # Calculate correlations
        res = 0
        counter = 0
        for i in range(score.shape[1]):
            tmp = spearmanr(
                np.array([approx_output[k][i] for k in range(len(approx_output))]),
                np.array([loss_list[k][i].numpy() for k in range(len(loss_list))])
            ).statistic
            if not np.isnan(tmp):
                res += tmp
                counter += 1

        return res/counter if counter > 0 else float('nan'), loss_list, approx_output
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return None, None, None