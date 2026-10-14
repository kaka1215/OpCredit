from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from opcredit.agent.modeling import configured_arguments, load_model, load_tokenizer
from opcredit.data.prepare import read_records


def encode_conversation(messages: list[dict], tokenizer, max_length: int):
    ids = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=False
        )
    )
    if len(ids) > max_length:
        return None
    labels = [-100] * len(ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = list(
            tokenizer.apply_chat_template(
                messages[:index], tokenize=True, return_dict=False, add_generation_prompt=True
            )
        )
        after = list(
            tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, return_dict=False, add_generation_prompt=False
            )
        )
        if ids[: len(before)] != before or ids[: len(after)] != after:
            raise ValueError("The chat template changes earlier tokens across turns.")
        end = len(after)
        while end > len(before) and not tokenizer.decode([ids[end - 1]]).strip():
            end -= 1
        labels[len(before) : end] = ids[len(before) : end]
    if not any(label != -100 for label in labels):
        raise ValueError("A trajectory must contain assistant supervision.")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class TrajectoryDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.samples = []
        skipped = 0
        for row in read_records(path):
            sample = encode_conversation(row["messages"], tokenizer, max_length)
            if sample is None:
                skipped += 1
            else:
                self.samples.append(sample)
        logging.info(
            "Encoded %d trajectories; excluded %d over-length traces.",
            len(self.samples),
            skipped,
        )
        if not self.samples:
            raise ValueError("No complete SFT trajectories fit the configured maximum length.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class AssistantCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        width = max(len(sample["input_ids"]) for sample in samples)
        pads = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        return {
            key: torch.tensor(
                [sample[key] + [pad] * (width - len(sample[key])) for sample in samples]
            )
            for key, pad in pads.items()
        }


def main():
    parser = argparse.ArgumentParser(
        description="Supervised training on template-based KG trajectories."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--warmup-ratio", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), required=True)
    parser.add_argument(
        "--attention", required=True, choices=("eager", "sdpa", "flash_attention_2")
    )
    parser.add_argument("--lr-scheduler", required=True)
    parser.add_argument("--max-grad-norm", type=float, required=True)
    parser.add_argument("--logging-steps", type=int, required=True)
    parser.add_argument("--save-total-limit", type=int, required=True)
    parser.add_argument("--dataloader-workers", type=int, required=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = configured_arguments(parser)
    if (
        min(
            args.epochs,
            args.micro_batch_size,
            args.max_length,
            args.save_steps,
            args.logging_steps,
            args.save_total_limit,
        )
        < 1
    ):
        parser.error("Epoch, batch, token, logging and checkpoint settings must be positive.")
    if not 0 <= args.warmup_ratio <= 1 or min(args.weight_decay, args.max_grad_norm) < 0:
        parser.error("Invalid warmup or regularization setting.")
    if args.learning_rate <= 0 or args.dataloader_workers < 0 or args.seed < 0:
        parser.error("Invalid learning rate, worker count or seed.")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("max_steps must be positive or -1 for epoch-based training.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    unit = args.micro_batch_size * world_size
    if unit < 1 or args.effective_batch_size < unit or args.effective_batch_size % unit:
        parser.error("Effective batch size must be divisible by micro batch size times world size.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Use a new output directory or specify --resume.")
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.precision, args.attention)
    model.config.use_cache = False
    dataset = TrajectoryDataset(args.data, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.effective_batch_size // unit,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        use_cpu=args.cpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=not args.cpu,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "final")
        (output / "run_config.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
