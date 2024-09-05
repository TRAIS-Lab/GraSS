"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import torch
from torch import nn
import argparse

from MLP.mnist_mlp import create_mlp_model
from retrain import key_value_pair
from dattri.benchmark.datasets.mnist import create_mnist_dataset
from dattri.benchmark.utils import SubsetSampler
from dattri.metrics.ground_truth import calculate_lds_ground_truth


SUPPORTED_MODELS = ["lr", "mlp", "resnet18", "resnet9", "musictransformer"]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Calculate the groundtruth via the retrained models.")

    argparser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        help=f"The dataset to use for retraining.\
               It should be one of {SUPPORTED_MODELS}.",
    )
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument(
        "--extra_param",
        type=key_value_pair,
        action="append",
        help="extra parameters to be passed to the create model function.\
              Must be in key=value format.",
    )
    args = argparser.parse_args()

    kwargs = {}
    kwargs.update(args.extra_param or {})

    activation_fn = None
    if args.model == "mlp":
        if args.extra_param:
            for key, value in args.extra_param:
                if key == "activation_fn":
                    activation_fn = value
                    break

    dataset_train, dataset_test = create_mnist_dataset(".")

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=SubsetSampler(range(500)),
    )

    model = create_mlp_model("mnist", **kwargs)
    model.to(args.device)
    model.eval()

    def target_func(ckpt_path, dataloader):
        params = torch.load(ckpt_path)
        model.load_state_dict(params)  # assuming model is defined somewhere
        model.eval()
        target_values = []
        loss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs.to(args.device))
                target_values.append(loss(outputs, labels.to(args.device)))
        return torch.Tensor(target_values)

    target_values, indices = calculate_lds_ground_truth(target_func, f"./result/retrain/{args.model}_{activation_fn}", test_loader)
    print(target_values[50:].shape)
    print(indices[50:].shape)

    torch.save(target_values[50:], f"./result/groundtruth/lds/{args.model}_{activation_fn}/target_values_lds.pt")
    torch.save(indices[50:], f"./result/groundtruth/lds/{args.model}_{activation_fn}/indices_lds.pt")