import logging
from collections.abc import Callable, Hashable, Sequence
from itertools import batched
from pathlib import Path
from typing import cast

import jsonlines

logger = logging.getLogger(__name__)


def recorded_keys[RecordT, KeyT: Hashable](output_path: Path, record_key: Callable[[RecordT], KeyT]) -> set[KeyT]:
    if not output_path.exists():
        return set()
    with jsonlines.open(output_path, "r") as reader:
        return {record_key(cast(RecordT, record)) for record in reader}


def process_unrecorded_items[ItemT, RecordT, KeyT: Hashable](
    items: Sequence[ItemT],
    output_path: Path,
    item_key: Callable[[ItemT], KeyT],
    record_key: Callable[[RecordT], KeyT],
    process_chunk: Callable[[Sequence[ItemT]], list[RecordT]],
    chunk_size: int,
) -> None:
    """Appends process_chunk's records to output_path one chunk at a time, skipping items
    whose key output_path already records, so an interrupted run resumes where it stopped."""
    already_recorded = recorded_keys(output_path, record_key)
    unrecorded_items = [item for item in items if item_key(item) not in already_recorded]
    logger.info(
        "%s: %d items to process, %d already recorded",
        output_path.name,
        len(unrecorded_items),
        len(items) - len(unrecorded_items),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk in batched(unrecorded_items, chunk_size):
        records = process_chunk(chunk)
        with jsonlines.open(output_path, "a") as writer:
            writer.write_all(records)
        logger.info("%s: recorded %d more", output_path.name, len(records))
