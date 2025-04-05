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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # create cifar 10 dataset
    model_details, groundtruth = load_benchmark(
        model="resnet9", dataset="cifar2", metric="lds"
    )

    print(groundtruth[0].shape)

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

    task = AttributionTask(model=model, loss_func=f, checkpoints=model_details["models_full"][:5])

    projector_kwargs = {
        "device": args.device,
        "use_half_precision": False,
        "method": "SJLT",
        "proj_dim": 4096,
    }

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
        projector_kwargs=projector_kwargs,
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

    proj_time = attributor.cache(train_dataloader)
    print("projection time:", proj_time)
    with torch.no_grad():
        score = attributor.attribute(test_dataloader=test_dataloader)

    print("score shape:", score.shape)
    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))