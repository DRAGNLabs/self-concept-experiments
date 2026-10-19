from typing import NewType, NotRequired, TypedDict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from jaxtyping import Float, Int
from pytensor.tensor.variable import TensorVariable

from selfconcept.arm_search.interface import SettingName

TermName = NewType("TermName", str)


class TermCells(TypedDict):
    """Episodes grouped by arm for one criterion model term; only episodes that met the earlier terms' criteria
    enter. `off_target_gate` is 1 where a gate setting is away from its target value."""

    group_index: Int[np.ndarray, " cell"]
    off_target_gate: Float[np.ndarray, "cell gate"]
    setting_codes: Int[np.ndarray, "cell setting"]
    trial_count: Int[np.ndarray, " cell"]
    success_count: Int[np.ndarray, " cell"]


class LadderCells(TypedDict):
    """`term_names` in ladder order; a free setting's code indexes its levels, and level 0 is the reference. Each
    group has one origin (e.g. the benchmark a problem comes from); groups of an origin share per-term intercepts."""

    term_names: list[TermName]
    cells_by_term: dict[TermName, TermCells]
    group_origin_index: Int[np.ndarray, " group"]
    origin_count: int
    gate_names: list[SettingName]
    setting_levels_by_name: dict[SettingName, list[str]]


class LadderPriors(TypedDict):
    """The optional fields replace the shared means and spread scales for single parameters (informed priors);
    `setting_effect_mean_by_level_by_setting` gives overall effects of non-reference levels, 0 where missing."""

    intercept_mean: float
    intercept_sd: float
    effect_sd: float
    effect_term_spread_sd: float
    group_spread_sd: float
    intercept_mean_by_origin_by_term: NotRequired[dict[TermName, list[float]]]
    gate_effect_mean: NotRequired[list[float]]
    setting_effect_mean_by_level_by_setting: NotRequired[dict[SettingName, dict[str, float]]]
    group_spread_sd_by_term: NotRequired[dict[TermName, float]]
    gate_group_spread_sd_by_term: NotRequired[dict[TermName, float]]


class LadderSamplerSettings(TypedDict):
    chains: int
    draws_per_chain: int
    tune: int
    target_accept: float
    random_seed: int


class TermParameters(TypedDict):
    origin_intercept: TensorVariable
    gate_effect: TensorVariable
    setting_effects: list[TensorVariable]
    group_effect: TensorVariable
    gate_group_deviation: TensorVariable


def _effect_by_term(
    effect_name: str,
    overall_mean: Float[np.ndarray, " level"],
    term_names: list[TermName],
    term_spread: TensorVariable,
    priors: LadderPriors,
) -> dict[TermName, TensorVariable]:
    """An overall effect shared by all terms plus a term deviation with a learned spread."""
    level_count = len(overall_mean)
    overall = pm.Normal(f"{effect_name}_overall", overall_mean, priors["effect_sd"], shape=level_count)
    term_raw = pm.Normal(f"{effect_name}_term_raw", 0.0, 1.0, shape=(len(term_names), level_count))
    return {
        term_name: pt.as_tensor_variable(overall + term_spread * term_raw[term_position])
        for term_position, term_name in enumerate(term_names)
    }


def _group_deviation(name: str, shape: tuple[int, ...], spread_sd: float) -> TensorVariable:
    spread = pm.HalfNormal(f"{name}_spread", spread_sd, shape=shape[:-1])
    raw = pm.Normal(f"{name}_raw", 0.0, 1.0, shape=shape)
    return pt.as_tensor_variable(pm.Deterministic(name, spread[..., None] * raw))


def _term_parameters(cells: LadderCells, priors: LadderPriors) -> dict[TermName, TermParameters]:
    term_names, gate_count = cells["term_names"], len(cells["gate_names"])
    group_count = len(cells["group_origin_index"])
    term_spread = pm.HalfNormal("effect_term_spread", priors["effect_term_spread_sd"])
    gate_effect_mean = np.array(priors.get("gate_effect_mean", [0.0] * gate_count))
    gate_effect_by_term = _effect_by_term("gate", gate_effect_mean, term_names, term_spread, priors)
    setting_mean_by_level_by_setting = priors.get("setting_effect_mean_by_level_by_setting", {})
    setting_effect_by_term_by_name = {
        setting_name: _effect_by_term(
            setting_name,
            np.array([setting_mean_by_level_by_setting.get(setting_name, {}).get(level, 0.0) for level in levels[1:]]),
            term_names,
            term_spread,
            priors,
        )
        for setting_name, levels in cells["setting_levels_by_name"].items()
        if len(levels) > 1
    }
    intercept_mean_by_origin_by_term = priors.get("intercept_mean_by_origin_by_term", {})
    group_spread_sd_by_term = priors.get("group_spread_sd_by_term", {})
    gate_group_spread_sd_by_term = priors.get("gate_group_spread_sd_by_term", {})
    return {
        term_name: TermParameters(
            origin_intercept=pt.as_tensor_variable(
                pm.Normal(
                    f"{term_name}_origin_intercept",
                    np.array(
                        intercept_mean_by_origin_by_term.get(
                            term_name, [priors["intercept_mean"]] * cells["origin_count"]
                        )
                    ),
                    priors["intercept_sd"],
                    shape=cells["origin_count"],
                )
            ),
            gate_effect=pt.as_tensor_variable(
                pm.Deterministic(f"{term_name}_gate_effect", gate_effect_by_term[term_name])
            ),
            # a one-level setting (e.g. a model with one effort) has nothing to estimate: its effect is a constant 0
            setting_effects=[
                pt.as_tensor_variable(
                    pm.Deterministic(
                        f"{term_name}_{setting_name}_effect",
                        pt.concatenate([pt.zeros(1), setting_effect_by_term_by_name[setting_name][term_name]]),
                    )
                )
                if setting_name in setting_effect_by_term_by_name
                else pt.zeros(1)
                for setting_name in cells["setting_levels_by_name"]
            ],
            group_effect=_group_deviation(
                f"{term_name}_group_effect",
                (group_count,),
                group_spread_sd_by_term.get(term_name, priors["group_spread_sd"]),
            ),
            gate_group_deviation=_group_deviation(
                f"{term_name}_gate_group_deviation",
                (gate_count, group_count),
                gate_group_spread_sd_by_term.get(term_name, priors["group_spread_sd"]),
            ),
        )
        for term_name in term_names
    }


def _add_term_likelihood(
    term_name: TermName, cells: TermCells, parameters: TermParameters, group_origin_index: TensorVariable
) -> None:
    group_index = pm.Data(f"{term_name}_group_index", cells["group_index"])
    off_target_gate = pm.Data(f"{term_name}_off_target_gate", cells["off_target_gate"])
    setting_codes = pt.as_tensor_variable(pm.Data(f"{term_name}_setting_codes", cells["setting_codes"]))
    setting_terms = [
        pt.take(effect, setting_codes[:, position]) for position, effect in enumerate(parameters["setting_effects"])
    ]
    logit = (
        pt.take(parameters["origin_intercept"], pt.take(group_origin_index, group_index))
        + pt.dot(off_target_gate, parameters["gate_effect"])
        + pt.take(parameters["group_effect"], group_index)
        + pt.sum(off_target_gate * parameters["gate_group_deviation"][:, group_index].T, axis=1)
        + (pt.sum(pt.stack(setting_terms), axis=0) if setting_terms else 0.0)
    )
    pm.Binomial(
        f"{term_name}_success_count",
        n=pm.Data(f"{term_name}_trial_count", cells["trial_count"]),
        logit_p=logit,
        observed=pm.Data(f"{term_name}_observed_success_count", cells["success_count"]),
    )


def sample_ladder_model(
    cells: LadderCells, priors: LadderPriors, sampler_settings: LadderSamplerSettings
) -> xr.DataTree:
    """Terms without cells keep their prior, so tiers no episode has reached yet still get predictions."""
    with pm.Model():
        parameters_by_term = _term_parameters(cells, priors)
        group_origin_index = pt.as_tensor_variable(
            pm.Data("group_origin_index", cells["group_origin_index"])
        )
        for term_name, term_cells in cells["cells_by_term"].items():
            if len(term_cells["trial_count"]) > 0:
                _add_term_likelihood(term_name, term_cells, parameters_by_term[term_name], group_origin_index)
        return pm.sample(
            draws=sampler_settings["draws_per_chain"],
            tune=sampler_settings["tune"],
            chains=sampler_settings["chains"],
            cores=sampler_settings["chains"],
            target_accept=sampler_settings["target_accept"],
            random_seed=sampler_settings["random_seed"],
            progressbar=False,
        )
