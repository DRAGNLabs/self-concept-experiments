import zlib

from selfconcept.self_recognition.schemas import Split


def prompt_split(prompt_id: int, holdout_fraction: float, seed: int) -> Split:
    """Depends only on the id, so appending prompts never reshuffles existing assignments."""
    return "holdout" if zlib.crc32(f"{seed}:{prompt_id}".encode()) / 2**32 < holdout_fraction else "train"
