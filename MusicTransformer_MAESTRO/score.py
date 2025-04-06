import argparse

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
from torch import nn

from _dattri.algorithm.trak import TRAKAttributor
from _dattri.benchmark.load import load_benchmark
from _dattri.metric import lds
from _dattri.task import AttributionTask
from _dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proj_method", type=str, default="Gaussian")
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--damping", type=float, default=0.1)
    args = parser.parse_args()

    # create cifar 10 dataset
    model_details, groundtruth = load_benchmark(
        model="musictransformer", dataset="maestro", metric="lds"
    )

    model = model_details["model"].to(args.device)
    model = model.eval()

    def loss_trak(params, data_target_pair):
        x, y = data_target_pair
        x_t = x.unsqueeze(0)
        y_t = y.unsqueeze(0)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

        output = torch.func.functional_call(model_details["model"], params, x_t)
        output_last = output[:, -1, :]
        y_last = y_t[:, -1]

        logp = -loss_fn(output_last, y_last)
        logit_func = logp - torch.log(1 - torch.exp(logp))
        return logit_func.squeeze(0)


    def correctness_p(params, data_target_pair):
        x, y = data_target_pair
        x_t = x.unsqueeze(0)
        y_t = y.unsqueeze(0)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

        output = torch.func.functional_call(model_details["model"], params, x_t)
        output_last = output[:, -1, :]
        y_last = y_t[:, -1]
        logp = -loss_fn(output_last, y_last)

        return torch.exp(logp)

    task = AttributionTask(
        model=model,
        loss_func=loss_trak,
        checkpoints=model_details["models_half"][:10]
    )

    projector_kwargs = {
        "device": args.device,
        "use_half_precision": False,
        "method": args.proj_method,
        "proj_dim": args.proj_dim,
    }

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=correctness_p,
        device=args.device,
        projector_kwargs=projector_kwargs,
        regularization=args.damping,
    )

    train_dataloader = torch.utils.data.DataLoader(
        model_details["train_dataset"],
        batch_size=16,
        shuffle=False,
        sampler=model_details["train_sampler"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        model_details["test_dataset"],
        batch_size=16,
        shuffle=False,
    )

    attributor.cache(train_dataloader)

    with torch.no_grad():
        score = attributor.attribute(test_dataloader=test_dataloader)

    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))

if __name__ == "__main__":
    main()