from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_jsonl_if_present(path: Path) -> list[Any]:
    return read_jsonl(path) if path.exists() else []


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(record, allow_nan=False) + "\n")
        file.flush()
