from pathlib import Path
from typing import NamedTuple

from selfconcept.measurement.interface import MeasurementName
from selfconcept.measurement.templates import ModelFamily


class RunConfig(NamedTuple):
    """Field names match the CLI's argparse destinations, which manifests record for resume checks."""
    measurement: MeasurementName
    model: str
    family: ModelFamily | None
    revision: str | None
    assistant_axis: Path | None
    axis_layers: list[int] | None
    out: Path
    corpus: Path
    n: int
    code_n: int
    offset: int
    code_offset: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    probe_source: Path | None
    max_new_tokens: int
    code_max_new_tokens: int
    max_attempts: int
    suffix: str
    train_only: bool
    scenarios: list[str]
