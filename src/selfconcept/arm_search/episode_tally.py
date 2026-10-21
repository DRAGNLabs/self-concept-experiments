from selfconcept.arm_search.interface import (
    Arm,
    ArmId,
    BatchUnit,
    EpisodeResult,
    GroupId,
    SuccessDefinition,
    Tier,
    TierName,
)


def assign_unit_seeds(chosen_arms: list[Arm], episodes: list[EpisodeResult], seeds_per_unit: int) -> list[BatchUnit]:
    """Merge repeated picks of an arm into one unit whose seeds continue past the arm's highest used seed."""
    arm_by_arm_id = {arm["arm_id"]: arm for arm in chosen_arms}
    next_seed_by_arm_id: dict[ArmId, int] = {
        arm_id: max((episode["seed"] for episode in episodes if episode["arm"]["arm_id"] == arm_id), default=-1) + 1
        for arm_id in arm_by_arm_id
    }
    pick_count_by_arm_id = {
        arm_id: sum(arm["arm_id"] == arm_id for arm in chosen_arms) for arm_id in next_seed_by_arm_id
    }
    return [
        BatchUnit(
            arm=arm_by_arm_id[arm_id],
            seeds=list(range(next_seed, next_seed + pick_count_by_arm_id[arm_id] * seeds_per_unit)),
        )
        for arm_id, next_seed in sorted(next_seed_by_arm_id.items())
    ]


def is_success(episode: EpisodeResult, success: SuccessDefinition) -> bool:
    return episode["category_by_factor"].get(success["factor_name"]) in success["success_category_names"]


def is_eligible(arm: Arm, tier: Tier) -> bool:
    return all(
        arm["gate_value_by_setting"][setting_name] in allowed_values
        for setting_name, allowed_values in tier["gate_values_by_setting"].items()
    )


def is_hit(episode: EpisodeResult, tier: Tier) -> bool:
    return is_eligible(episode["arm"], tier) and is_success(episode, tier["success"])


def covered_group_ids(episodes: list[EpisodeResult], tier: Tier) -> set[GroupId]:
    return {episode["arm"]["group_id"] for episode in episodes if is_hit(episode, tier)}


def realized_reward_by_tier(episodes: list[EpisodeResult], ladder: list[Tier]) -> dict[TierName, float]:
    return {tier["tier_name"]: float(len(covered_group_ids(episodes, tier))) for tier in ladder}


def realized_cost(episodes: list[EpisodeResult]) -> float:
    return float(sum(episode["cost"] for episode in episodes))
