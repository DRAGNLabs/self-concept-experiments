from typing import TypedDict

import arviz as az
import numpy as np
import xarray as xr
from jaxtyping import Float, Int
from scipy.special import expit

from selfconcept.arm_search.interface import EffectEstimate
from selfconcept.arm_search.ladder_model import LadderCells, TermName


class TermDraws(TypedDict):
    origin_intercept: Float[np.ndarray, "draw origin"]
    gate_effect: Float[np.ndarray, "draw gate"]
    setting_effects: list[Float[np.ndarray, "draw level"]]
    group_effect: Float[np.ndarray, "draw group"]
    gate_group_deviation: Float[np.ndarray, "draw gate group"]


class LadderDraws(TypedDict):
    term_draws_by_name: dict[TermName, TermDraws]
    group_origin_index: Int[np.ndarray, " group"]


class ArmFeatures(TypedDict):
    group_index: Int[np.ndarray, " arm"]
    off_target_gate: Float[np.ndarray, "arm gate"]
    setting_codes: Int[np.ndarray, "arm setting"]


class FitDiagnostics(TypedDict):
    diagnostic_value_by_name: dict[str, float]
    warnings: list[str]


def _pooled_draws(posterior: xr.Dataset, name: str) -> Float[np.ndarray, "draw *shape"]:
    values = posterior[name].to_numpy()
    return values.reshape(-1, *values.shape[2:])


def ladder_draws(trace: xr.DataTree, cells: LadderCells) -> LadderDraws:
    posterior = trace.children["posterior"].to_dataset()
    return LadderDraws(
        term_draws_by_name={
            term_name: TermDraws(
                origin_intercept=_pooled_draws(posterior, f"{term_name}_origin_intercept"),
                gate_effect=_pooled_draws(posterior, f"{term_name}_gate_effect"),
                setting_effects=[
                    _pooled_draws(posterior, f"{term_name}_{setting_name}_effect")
                    for setting_name in cells["setting_levels_by_name"]
                ],
                group_effect=_pooled_draws(posterior, f"{term_name}_group_effect"),
                gate_group_deviation=_pooled_draws(posterior, f"{term_name}_gate_group_deviation"),
            )
            for term_name in cells["term_names"]
        },
        group_origin_index=cells["group_origin_index"],
    )


def term_logit(
    term_draws: TermDraws, group_origin_index: Int[np.ndarray, " group"], features: ArmFeatures
) -> Float[np.ndarray, "draw arm"]:
    group_index, off_target_gate = features["group_index"], features["off_target_gate"]
    setting_logit = sum(
        (
            effect[:, features["setting_codes"][:, position]]
            for position, effect in enumerate(term_draws["setting_effects"])
        ),
        np.zeros((term_draws["group_effect"].shape[0], len(group_index))),
    )
    return (
        term_draws["origin_intercept"][:, group_origin_index[group_index]]
        + term_draws["gate_effect"] @ off_target_gate.T
        + term_draws["group_effect"][:, group_index]
        + np.einsum("dga,ag->da", term_draws["gate_group_deviation"][:, :, group_index], off_target_gate)
        + setting_logit
    )


def hit_probability(draws: LadderDraws, term_count: int, features: ArmFeatures) -> Float[np.ndarray, "draw arm"]:
    """Probability per seed of meeting the first `term_count` criteria."""
    term_draws = list(draws["term_draws_by_name"].values())[:term_count]
    return np.prod(
        [expit(term_logit(draws_of_term, draws["group_origin_index"], features)) for draws_of_term in term_draws],
        axis=0,
    )


def fit_diagnostics(trace: xr.DataTree, max_r_hat: float, max_divergences: int) -> FitDiagnostics:
    r_hat = az.rhat(trace)
    max_r_hat_value = max(float(r_hat[name].max()) for name in r_hat.data_vars)
    divergence_count = int(trace.children["sample_stats"].to_dataset()["diverging"].to_numpy().sum())
    return FitDiagnostics(
        diagnostic_value_by_name={"max_r_hat": max_r_hat_value, "divergence_count": float(divergence_count)},
        warnings=[
            *([f"max R-hat {max_r_hat_value:.3f} > {max_r_hat}"] if max_r_hat_value > max_r_hat else []),
            *(
                [f"{divergence_count} divergences > {max_divergences}"]
                if divergence_count > max_divergences
                else []
            ),
        ],
    )


def _effect_estimate(effect_name: str, level_name: str, draws: Float[np.ndarray, " draw"]) -> EffectEstimate:
    lower, upper = np.quantile(draws, [0.05, 0.95])
    return EffectEstimate(
        effect_name=effect_name, level_name=level_name, mean=float(draws.mean()), lower=float(lower), upper=float(upper)
    )


def ladder_effect_estimates(trace: xr.DataTree, cells: LadderCells) -> list[EffectEstimate]:
    """Overall gate and free-setting effects (logit, vs the target gate value or reference level), and the group
    spreads per term, which set how well group rankings transfer up the ladder."""
    posterior = trace.children["posterior"].to_dataset()
    gate_overall, term_spread = _pooled_draws(posterior, "gate_overall"), _pooled_draws(posterior, "effect_term_spread")
    return [
        *(
            _effect_estimate(f"{gate_name} off target", "overall", gate_overall[:, position])
            for position, gate_name in enumerate(cells["gate_names"])
        ),
        *(
            _effect_estimate(setting_name, level, _pooled_draws(posterior, f"{setting_name}_overall")[:, position])
            for setting_name, levels in cells["setting_levels_by_name"].items()
            for position, level in enumerate(levels[1:])
        ),
        _effect_estimate("effect term spread", "", term_spread),
        *(
            _effect_estimate(
                "group spread", term_name, _pooled_draws(posterior, f"{term_name}_group_effect_spread")
            )
            for term_name in cells["term_names"]
        ),
        *(
            _effect_estimate(
                f"{gate_name} group spread",
                term_name,
                _pooled_draws(posterior, f"{term_name}_gate_group_deviation_spread")[:, position],
            )
            for term_name in cells["term_names"]
            for position, gate_name in enumerate(cells["gate_names"])
        ),
    ]
