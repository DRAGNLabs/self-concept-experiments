import logging
from collections.abc import Callable
from datetime import UTC, datetime

from selfconcept.arm_search.episode_tally import covered_group_ids, realized_cost, realized_reward_by_tier
from selfconcept.arm_search.interface import ArmId, EpisodeResult, SearchApplication
from selfconcept.arm_search.records import (
    PlanningOutcome,
    RoundPlan,
    RoundResult,
    SearchConfig,
    SearchSummary,
    StopReason,
)
from selfconcept.arm_search.state import (
    append_episodes,
    append_round_plan,
    append_round_result,
    initialize_state,
    load_episodes,
    load_rounds,
)

logger = logging.getLogger(__name__)


def _episode_key(episode: EpisodeResult) -> tuple[ArmId, int]:
    return episode["arm"]["arm_id"], episode["seed"]


def _complete_round(application: SearchApplication, config: SearchConfig, plan: RoundPlan) -> None:
    """A manager that died while appending the round's episodes left some of them stored; seeds never repeat for an
    arm, so stored episodes with a requested key are this round's and are neither counted as before nor stored twice."""
    state_dir = config["state_dir"]
    requested_keys = {(unit["arm"]["arm_id"], seed) for unit in plan["units"] for seed in unit["seeds"]}
    stored_episodes = load_episodes(state_dir)
    stored_keys = {_episode_key(episode) for episode in stored_episodes}
    episodes_before = [episode for episode in stored_episodes if _episode_key(episode) not in requested_keys]
    reward_before_by_tier = realized_reward_by_tier(episodes_before, config["ladder"])
    target_covered_before = covered_group_ids(episodes_before, config["ladder"][-1])
    new_episodes = application.run_batch(plan["units"], plan["round_index"])
    returned_keys = [_episode_key(episode) for episode in new_episodes]
    if len(set(returned_keys)) != len(returned_keys) or not set(returned_keys) <= requested_keys:
        raise ValueError(f"run_batch for round {plan['round_index']} returned unrequested or duplicate episodes")
    if len(returned_keys) < len(requested_keys):
        logger.warning(
            "round %d: %d of %d requested episodes missing (failed runs)",
            plan["round_index"],
            len(requested_keys) - len(returned_keys),
            len(requested_keys),
        )
    append_episodes(state_dir, [episode for episode in new_episodes if _episode_key(episode) not in stored_keys])
    reward_after_by_tier = realized_reward_by_tier(load_episodes(state_dir), config["ladder"])
    realized_reward_by_tier_name = {
        tier_name: reward_after_by_tier[tier_name] - reward_before
        for tier_name, reward_before in reward_before_by_tier.items()
    }
    realized_hits_by_tier = {
        tier["tier_name"]: float(len(covered_group_ids(new_episodes, tier) - target_covered_before))
        for tier in config["ladder"]
    }
    append_round_result(
        state_dir,
        plan["round_index"],
        RoundResult(
            completed_at=datetime.now(UTC).isoformat(),
            realized_reward_by_tier=realized_reward_by_tier_name,
            realized_hits_by_tier=realized_hits_by_tier,
            realized_cost=realized_cost(new_episodes),
        ),
    )
    logger.info(
        "round %d at tier %s: realized hits %.3g (expected %.3g)",
        plan["round_index"],
        plan["tier_name"],
        realized_hits_by_tier[plan["tier_name"]],
        plan["expected_hits_by_tier"][plan["tier_name"]],
    )


def _summary(config: SearchConfig, stop_reason: StopReason, detail: str) -> SearchSummary:
    episodes = load_episodes(config["state_dir"])
    logger.info("search stopped: %s (%s)", stop_reason, detail)
    return SearchSummary(
        rounds=load_rounds(config["state_dir"]),
        total_reward_by_tier=realized_reward_by_tier(episodes, config["ladder"]),
        total_cost=realized_cost(episodes),
        stop_reason=stop_reason,
        detail=detail,
    )


def run_search(
    application: SearchApplication,
    config: SearchConfig,
    plan_next_round: Callable[[SearchConfig], PlanningOutcome],
) -> SearchSummary:
    """Alternate planning and batches until a stop condition. Resumes from the state dir, rerunning an unfinished
    batch."""
    initialize_state(config)
    reward_target = config["stop_conditions"]["reward_target"]
    while True:
        rounds = load_rounds(config["state_dir"])
        if rounds and rounds[-1]["result"] is None:
            _complete_round(application, config, rounds[-1]["plan"])
            continue
        target_reward = len(covered_group_ids(load_episodes(config["state_dir"]), config["ladder"][-1]))
        if reward_target is not None and target_reward >= reward_target:
            return _summary(config, "reached_reward_target", f"target-tier reward {target_reward}")
        outcome = plan_next_round(config)
        if outcome["kind"] == "search_stopped":
            return _summary(config, outcome["stop_reason"], outcome["detail"])
        append_round_plan(config["state_dir"], outcome["plan"])
