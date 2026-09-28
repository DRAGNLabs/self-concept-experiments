"""Sampling seeds shared across experiments."""

import zlib


def turn_sampling_seed(sample_seed: int, example_id: str, turn: int) -> int:
    """A seed fixed by (sample_seed, example_id, turn), so sampled text doesn't depend on batching,
    shard layout or resumes."""
    key = f"{sample_seed}:{example_id}" + (f":{turn}" if turn else "")
    return zlib.crc32(key.encode())
