import argparse

import torch

from gpt_utils import construct_model, get_datasets, set_seed
from transformers import Trainer, TrainingArguments, default_data_collator

from _logix.huggingface import LogIXArguments, patch_trainer

def main():
    parser = argparse.ArgumentParser("GLUE Influence Analysis")
    parser.add_argument("--project", type=str, default="wiki")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    set_seed(0)

    # prepare model & data loader
    model, tokenizer = construct_model(resume=True)
    model.eval()
    train_dataset, test_dataset = get_datasets()
    LogIXTrainer = patch_trainer(Trainer)

    # 1. Computing EK-FAC factors for training data
    logix_args_train = LogIXArguments(
        project=args.project,
        config=args.config_path,
        lora=True,
        hessian="raw",
        save="grad",
        train_data=True,
        label_key="input_ids",
        log_num_workers=4,
    )
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = LogIXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        args=training_args,
        logix_args=logix_args_train,
    )
    trainer.extract_log()

    # 2. Computing influence scores for test data
    logix_args_test = LogIXArguments(
        project=args.project,
        config=args.config_path,
        lora=True,
        hessian="raw",
        save="grad",
        train_data=False,
        label_key="input_ids",
        initialize_from_log=True,
        log_batch_size=args.batch_size,
    )
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        report_to="none",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer = LogIXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        args=training_args,
        logix_args=logix_args_test,
    )
    result = trainer.influence()
    if_scores = result["influence"].T
    print(if_scores)

    torch.save(if_scores, "gpt_influence.pt")

if __name__ == "__main__":
    main()