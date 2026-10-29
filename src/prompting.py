from typing import Any


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
