from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

import numpy as np
from jaxtyping import Float
from scipy.stats import wilcoxon

from selfconcept.assistant_axis_cot.records import REGIONS, AxisName, Condition, TokenProjectionRecord

type Measure = Literal["projection", "cosine"]
type ContentRegion = Literal["cot", "final"]

MEASURES: tuple[Measure, ...] = ("projection", "cosine")
CONTENT_REGIONS: tuple[ContentRegion, ...] = ("cot", "final")
MAD_TO_STANDARD_DEVIATION = 1.4826  # scales the MAD of normal data to its standard deviation


class RegionMoments(TypedDict):
    token_count: int
    mean: float
    variance: float  # population (ddof=0), so the token-weighted decomposition below is exact
    robust_variance: float  # (scaled MAD)^2


class ConversationSummary(TypedDict):
    role: str
    question_index: int
    condition: Condition
    moments_by_region: dict[ContentRegion, RegionMoments]


class PairedTest(TypedDict):
    count: int
    estimate: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float


class VarianceDecomposition(TypedDict):
    pooled: float
    within: float
    between: float


class PersonaDrop(TypedDict):
    question_count: int
    drop_by_region: dict[ContentRegion, PairedTest]
    final_minus_cot: PairedTest
    unprompted_within_sd_by_region: dict[ContentRegion, float]
    standardized_final_minus_cot: PairedTest
    drop_by_region_by_role: dict[str, dict[ContentRegion, float]]


def token_values(record: TokenProjectionRecord, axis: AxisName, measure: Measure) -> Float[np.ndarray, " n_tokens"]:
    projections = record["projections_by_axis"][axis]
    return projections if measure == "projection" else projections / record["residual_norms"]


def region_values(
    record: TokenProjectionRecord, axis: AxisName, measure: Measure, region: ContentRegion
) -> Float[np.ndarray, " n_tokens"]:
    return token_values(record, axis, measure)[record["region_codes"] == REGIONS.index(region)]


def has_both_regions(record: TokenProjectionRecord, min_region_tokens: int) -> bool:
    return all(
        np.count_nonzero(record["region_codes"] == REGIONS.index(region)) >= min_region_tokens
        for region in CONTENT_REGIONS
    )


def region_moments(values: Float[np.ndarray, " n_tokens"]) -> RegionMoments:
    median_absolute_deviation = float(np.median(np.abs(values - np.median(values))))
    return {
        "token_count": len(values),
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "robust_variance": (MAD_TO_STANDARD_DEVIATION * median_absolute_deviation) ** 2,
    }


def summarize_conversations(
    records: Sequence[TokenProjectionRecord], axis: AxisName, measure: Measure, min_region_tokens: int
) -> list[ConversationSummary]:
    """Conversations with at least min_region_tokens in both CoT and final; the rest are dropped."""
    return [
        {
            "role": record["role"],
            "question_index": record["question_index"],
            "condition": record["condition"],
            "moments_by_region": {
                region: region_moments(region_values(record, axis, measure, region)) for region in CONTENT_REGIONS
            },
        }
        for record in records
        if has_both_regions(record, min_region_tokens)
    ]


def question_bootstrap_ci(
    values: Float[np.ndarray, " n"],
    question_indices: Sequence[int],
    rng: np.random.Generator,
    resample_count: int,
) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling whole questions, since values sharing a question are not independent."""
    values_by_question: defaultdict[int, list[float]] = defaultdict(list)
    for value, question_index in zip(values, question_indices, strict=True):
        values_by_question[question_index].append(value)
    question_values = [np.asarray(question_values) for question_values in values_by_question.values()]
    question_sums = np.array([question_value.sum() for question_value in question_values])
    question_counts = np.array([len(question_value) for question_value in question_values])
    resampled = rng.integers(0, len(question_values), size=(resample_count, len(question_values)))
    resampled_means = question_sums[resampled].sum(axis=1) / question_counts[resampled].sum(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_p_value(values: Float[np.ndarray, " n"]) -> float:
    result: Any = wilcoxon(values)  # scipy's decorator erases the return type
    return float(result.pvalue)


def paired_test(
    values: Float[np.ndarray, " n"],
    question_indices: Sequence[int],
    rng: np.random.Generator,
    resample_count: int,
    estimate_transform: Callable[[float], float] = float,
) -> PairedTest:
    """Mean of per-unit paired values with a question-bootstrap CI and a two-sided Wilcoxon signed-rank p;
    estimate_transform maps the mean and CI ends (e.g. exp for log ratios)."""
    ci_low, ci_high = question_bootstrap_ci(values, question_indices, rng, resample_count)
    return {
        "count": len(values),
        "estimate": estimate_transform(float(values.mean())),
        "ci_low": estimate_transform(ci_low),
        "ci_high": estimate_transform(ci_high),
        "wilcoxon_p": wilcoxon_p_value(values),
    }


def cot_minus_final_mean(
    summaries: Sequence[ConversationSummary], rng: np.random.Generator, resample_count: int
) -> PairedTest:
    differences = np.array([
        summary["moments_by_region"]["cot"]["mean"] - summary["moments_by_region"]["final"]["mean"]
        for summary in summaries
    ])
    return paired_test(differences, [summary["question_index"] for summary in summaries], rng, resample_count)


def cot_over_final_variance_ratio(
    summaries: Sequence[ConversationSummary],
    rng: np.random.Generator,
    resample_count: int,
    variance_key: Literal["variance", "robust_variance"],
) -> PairedTest:
    """Geometric-mean CoT/final variance ratio across conversations; below 1 means CoT varies less."""
    log_ratios = np.log([
        summary["moments_by_region"]["cot"][variance_key] / summary["moments_by_region"]["final"][variance_key]
        for summary in summaries
    ])
    return paired_test(
        log_ratios, [summary["question_index"] for summary in summaries], rng, resample_count, estimate_transform=np.exp
    )


def variance_decomposition(summaries: Sequence[ConversationSummary], region: ContentRegion) -> VarianceDecomposition:
    """Token-pooled variance = token-weighted mean within-conversation variance + token-weighted variance of
    conversation means."""
    moments = [summary["moments_by_region"][region] for summary in summaries]
    token_counts = np.array([moment["token_count"] for moment in moments])
    means = np.array([moment["mean"] for moment in moments])
    variances = np.array([moment["variance"] for moment in moments])
    pooled_mean = np.average(means, weights=token_counts)
    within = float(np.average(variances, weights=token_counts))
    between = float(np.average((means - pooled_mean) ** 2, weights=token_counts))
    return {"pooled": within + between, "within": within, "between": between}


def mean_by_question(summaries: Sequence[ConversationSummary], region: ContentRegion) -> dict[int, float]:
    means_by_question: defaultdict[int, list[float]] = defaultdict(list)
    for summary in summaries:
        means_by_question[summary["question_index"]].append(summary["moments_by_region"][region]["mean"])
    return {question_index: float(np.mean(means)) for question_index, means in means_by_question.items()}


def drop_by_region_by_role(summaries: Sequence[ConversationSummary]) -> dict[str, dict[ContentRegion, float]]:
    """Per persona role, the unprompted minus role mean of conversation means, averaged over questions."""
    unprompted = [summary for summary in summaries if summary["condition"] == "unprompted"]
    persona = [summary for summary in summaries if summary["condition"] == "persona"]
    unprompted_mean_by_question_by_region = {region: mean_by_question(unprompted, region) for region in CONTENT_REGIONS}
    return {
        role: {
            region: float(np.mean([
                unprompted_mean_by_question_by_region[region][question] - role_mean
                for question, role_mean in mean_by_question(
                    [summary for summary in persona if summary["role"] == role], region
                ).items()
                if question in unprompted_mean_by_question_by_region[region]
            ]))
            for region in CONTENT_REGIONS
        }
        for role in sorted({summary["role"] for summary in persona})
    }


def persona_drop(summaries: Sequence[ConversationSummary], rng: np.random.Generator, resample_count: int) -> PersonaDrop:
    """Per question, the unprompted minus persona mean of conversation means, in each region; positive means the
    persona lowers the projection."""
    unprompted = [summary for summary in summaries if summary["condition"] == "unprompted"]
    persona = [summary for summary in summaries if summary["condition"] == "persona"]
    unprompted_mean_by_question_by_region = {region: mean_by_question(unprompted, region) for region in CONTENT_REGIONS}
    persona_mean_by_question_by_region = {region: mean_by_question(persona, region) for region in CONTENT_REGIONS}
    shared_questions = sorted(
        unprompted_mean_by_question_by_region["final"].keys() & persona_mean_by_question_by_region["final"].keys()
    )
    drops_by_region = {
        region: np.array([
            unprompted_mean_by_question_by_region[region][question] - persona_mean_by_question_by_region[region][question]
            for question in shared_questions
        ])
        for region in CONTENT_REGIONS
    }
    unprompted_within_sd_by_region: dict[ContentRegion, float] = {
        region: float(np.sqrt(variance_decomposition(unprompted, region)["within"])) for region in CONTENT_REGIONS
    }
    standardized_drops_by_region = {
        region: drops_by_region[region] / unprompted_within_sd_by_region[region] for region in CONTENT_REGIONS
    }

    return {
        "question_count": len(shared_questions),
        "drop_by_region": {
            region: paired_test(drops_by_region[region], shared_questions, rng, resample_count)
            for region in CONTENT_REGIONS
        },
        "final_minus_cot": paired_test(
            drops_by_region["final"] - drops_by_region["cot"], shared_questions, rng, resample_count
        ),
        "unprompted_within_sd_by_region": unprompted_within_sd_by_region,
        "standardized_final_minus_cot": paired_test(
            standardized_drops_by_region["final"] - standardized_drops_by_region["cot"],
            shared_questions,
            rng,
            resample_count,
        ),
        "drop_by_region_by_role": drop_by_region_by_role(summaries),
    }
