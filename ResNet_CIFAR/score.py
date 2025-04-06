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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proj_method", type=str, default="Gaussian")
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--damping", type=float, default=0.1)
    args = parser.parse_args()

    # create cifar 10 dataset
    model_details, groundtruth = load_benchmark(
        model="resnet9", dataset="cifar2", metric="lds"
    )

    model = model_details["model"].to(args.device)
    model = model.eval()

    def f(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        logp = -loss(yhat, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    task = AttributionTask(model=model, loss_func=f, checkpoints=model_details["models_half"][:10])

    projector_kwargs = {
        "device": args.device,
        "use_half_precision": False,
        "method": args.proj_method,
        "proj_dim": args.proj_dim,
    }

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
        projector_kwargs=projector_kwargs,
        regularization=args.damping,
    )

    train_dataloader = torch.utils.data.DataLoader(
        model_details["train_dataset"],
        batch_size=64,
        sampler=model_details["train_sampler"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        model_details["test_dataset"],
        batch_size=64,
        sampler=model_details["test_sampler"],
    )

    attributor.cache(train_dataloader)

    with torch.no_grad():
        score = attributor.attribute(test_dataloader=test_dataloader)

    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))

if __name__ == "__main__":
    main()