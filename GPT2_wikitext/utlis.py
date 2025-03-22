import torch
import numpy as np
from scipy.stats import spearmanr
import csv

def lds(score, training_setting):
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

def setup_projection_kwargs(args, device):
    if args.projection is None:
        proj_method = "Identity"
        proj_factorize = False
        proj_dim = -1

    proj_method, proj_dim = args.projection.split("-")
    # proj_dim might be of the form 'proj_dim*proj_dim' for factorized projection, for simply 'proj_dim' for non-factorized projection
    if "*" in proj_dim:
        proj_factorize = True
        proj_dim = proj_dim.split("*")
        assert proj_dim[0] == proj_dim[1], "Projection dimension must be the same for factorized projection."

        proj_dim = int(proj_dim[0]) # Convert to integer
        if proj_method == "SJLT":
            assert int(proj_dim[0]) > 512, "Projection dimension must be greater than 512 for to project the entire gradient to avoid the slow down due to local collisions."
    else:
        proj_factorize = False
        proj_dim = int(proj_dim) # Convert to integer

    projector_kwargs = {
        "proj_dim": proj_dim,
        "proj_max_batch_size": 32,
        "proj_seed": args.seed,
        "proj_factorize": proj_factorize,
        "device": device,
        "method": proj_method,
        "use_half_precision": False,
        "threshold": args.threshold,
        "random_drop": args.random_drop,
    }

    return projector_kwargs

def batch_size(baseline, tda):
    if baseline == "GC":
        if tda == "GD":
            train_batch_size = 8
            test_batch_size = 8
        elif tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 6
            test_batch_size = 6
    elif baseline == "LoGra":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 8
            test_batch_size = 8
    elif baseline == "LogIX":
        if tda in ["IF-NONE", "IF-RAW", "IF-KFAC", "IF-EKFAC"]:
            train_batch_size = 32
            test_batch_size = 32
    elif baseline == "dattri":
        if tda in ["TRAK", "GD"]:
            train_batch_size = 4
            test_batch_size = 4
    else:
        raise ValueError("Invalid baseline and tda combination.")

    return train_batch_size, test_batch_size

def result_filename(args):
    filename_parts = []

    if args.projection is not None:
        filename_parts.append(args.projection)

    filename_parts.append(f"thrd-{args.threshold}")
    filename_parts.append(f"rdp-{args.random_drop}")

    training_setting = args.output_dir.split("/")[-1]
    # Join parts and save the file
    result_filename = f"./results/{training_setting}/{args.baseline}/{args.tda}/{args.layer}/{'_'.join(filename_parts)}.pt"

    return result_filename