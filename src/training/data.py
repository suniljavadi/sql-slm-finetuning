from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset


def format_prompt(record: dict[str, Any]) -> str:
    return (
        "### Instruction:\n"
        f"{record['instruction']}\n\n"
        "### Request:\n"
        f"{record['input']}\n\n"
        "### Schema:\n"
        f"{record['schema']}\n\n"
        "### SQL:\n"
    )


class SQLInstructionDataset(Dataset):
    """Tokenize SQL examples while computing loss only on the expected answer."""

    def __init__(self, records: Sequence[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        if max_seq_length < 3:
            raise ValueError("max_seq_length must be at least 3")
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id")

        self.examples: list[dict[str, list[int]]] = []
        for record in records:
            self.examples.append(self._encode(record, tokenizer, max_seq_length))

    @staticmethod
    def _encode(record: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
        prompt_ids = tokenizer(
            format_prompt(record),
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length - 2,
        )["input_ids"]
        response_budget = max_seq_length - len(prompt_ids) - 1
        if response_budget < 1:
            raise ValueError("Prompt leaves no room for an SQL answer")
        response_ids = tokenizer(
            record["output"],
            add_special_tokens=False,
            truncation=True,
            max_length=response_budget,
        )["input_ids"]
        if not response_ids:
            raise ValueError("Tokenized SQL answer is empty")
        response_ids.append(tokenizer.eos_token_id)

        input_ids = list(prompt_ids) + list(response_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + list(response_ids),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalLMCollator:
    """Right-pad token batches and exclude padding from the loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        batch_size = len(features)
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}