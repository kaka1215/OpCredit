from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from opcredit.agent.prompts import SYSTEM_PROMPT


def configured_arguments(parser: argparse.ArgumentParser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path)
    known, _ = selector.parse_known_args(argv)
    parser.add_argument("--config", type=Path)
    if known.config:
        try:
            values = json.loads(known.config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read configuration: {error}")
        if not isinstance(values, dict):
            parser.error("Configuration must be a JSON object.")
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
        unknown = set(values) - actions.keys()
        if unknown:
            parser.error(f"Unknown configuration fields: {sorted(unknown)}")
        defaults = {}
        for name, value in values.items():
            if value is None:
                continue
            action = actions[name]
            if any(token.split("=", 1)[0] in action.option_strings for token in argv):
                continue
            items = value if isinstance(action, argparse._AppendAction) else [value]
            if not isinstance(items, list) or not items:
                parser.error(f"{name} must be a nonempty list.")
            converted = []
            for item in items:
                if item is None or isinstance(item, (dict, list)):
                    parser.error(f"Invalid value for {name}.")
                if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
                    if type(item) is not bool:
                        parser.error(f"{name} must be a JSON boolean.")
                else:
                    if isinstance(item, bool) or (action.type is int and type(item) is not int):
                        parser.error(f"Invalid numeric value for {name}.")
                    if action.type not in (int, float) and not isinstance(item, str):
                        parser.error(f"{name} must be a string.")
                    if isinstance(item, str) and not item.strip():
                        parser.error(f"{name} must not be empty.")
                    try:
                        item = action.type(item) if action.type else item
                    except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
                        parser.error(f"Invalid value for {name}: {error}")
                if action.choices is not None and item not in action.choices:
                    parser.error(f"Invalid choice for {name}: {item}")
                converted.append(item)
            defaults[name] = (
                converted if isinstance(action, argparse._AppendAction) else converted[0]
            )
            action.required = False
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"{name} must be finite.")
        if isinstance(value, str) and not value.strip():
            parser.error(f"{name} must not be empty.")
    return args


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Use an instruction tokenizer with a chat template and EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_dtype(precision: str):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def load_model(path: str, precision: str, attention: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=model_dtype(precision),
        attn_implementation=attention,
        trust_remote_code=False,
    )


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name in (
        "attention_dropout",
        "hidden_dropout",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
    ):
        if hasattr(model.config, name):
            setattr(model.config, name, 0.0)


class ChatBuffer:
    def __init__(self, tokenizer, observation: str):
        self.tokenizer = tokenizer
        self.input_ids = list(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation},
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        )
        self.response_mask = [False] * len(self.input_ids)

    def append_generation(self, tokens: list[int]):
        if not tokens:
            raise ValueError("A sampled decision must contain tokens.")
        self.input_ids.extend(tokens)
        self.response_mask.extend([True] * len(tokens))

    def append_observation(self, observation: str):
        marker = "OPCREDIT_ASSISTANT_BOUNDARY"
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Previous observation."},
                {"role": "assistant", "content": marker},
                {"role": "user", "content": observation},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.count(marker) != 1:
            raise ValueError("The chat template does not preserve assistant content.")
        suffix = self.tokenizer.encode(rendered.split(marker, 1)[1], add_special_tokens=False)
        if (
            suffix
            and self.input_ids[-1] in self.tokenizer.all_special_ids
            and self.input_ids[-1] == suffix[0]
        ):
            suffix = suffix[1:]
        self.input_ids.extend(suffix)
        self.response_mask.extend([False] * len(suffix))
