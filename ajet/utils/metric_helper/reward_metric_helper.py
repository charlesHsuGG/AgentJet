"""
deep_finance Reward Metrics Helper

Provides standalone utility functions for reward_stats extraction and SwanLab metrics formatting.

Data sources:
1. Finance Evaluator (finance_raw, finance_contribution)
2. OpenJudge Graders (openjudge_xxx_raw, openjudge_xxx_contribution)

SwanLab metrics directory structure:
- rewards/                    Top-level aggregated scores
- rewards/dimensions/         Raw scores (unweighted): finance_raw, openjudge_*_raw
- rewards/contribution/       Weighted contributions: finance_contribution, openjudge_*_contribution
- rewards/openjudge/          OpenJudge grader specific metrics
- judge_time/                 Judge time consumption statistics
"""

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from ajet.schema.trajectory import Reward


def extract_reward_stats_from_trajectories(trajectories: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract reward_stats from trajectories list.

    Args:
        trajectories: List of trajectory objects containing log_metrics

    Returns:
        List of reward_stats dictionaries
    """
    reward_stats_list = []
    for traj in trajectories:
        if hasattr(traj, 'log_metrics') and traj.log_metrics:
            if 'reward_stats' in traj.log_metrics:
                reward_stats_list.append(traj.log_metrics['reward_stats'])
    return reward_stats_list


def compute_reward_metrics(reward_stats_list: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """
    Compute SwanLab metrics from reward_stats list.

    Data sources:
    1. Finance Evaluator (finance_raw, finance_contribution)
    2. OpenJudge Graders (openjudge_xxx_raw, openjudge_xxx_contribution)

    Args:
        reward_stats_list: List of reward_stats dictionaries
        prefix: Metric name prefix (e.g., "val/" for validation phase)

    Returns:
        Formatted metrics dictionary ready for SwanLab reporting
    """
    if not reward_stats_list:
        return {}

    n = len(reward_stats_list)
    metrics = {}

    # ========== Top-level Scores (General) ==========
    final_reward_list = [rs.pop('final_reward', 0.0) for rs in reward_stats_list if 'final_reward' in rs]
    fused_reward_list = [rs.pop('fused_reward', 0.0) for rs in reward_stats_list if 'fused_reward' in rs]
    penalty_list = [rs.pop('penalty', 0.0) for rs in reward_stats_list if 'penalty' in rs]
    step_reward_list = [rs.pop('step_reward', 0.0) for rs in reward_stats_list if 'step_reward' in rs]

    # Penalty statistics
    non_zero_penalties = [p for p in penalty_list if p != 0.0]

    # Top-level metrics
    if final_reward_list:
        metrics[f"{prefix}rewards/final_reward_mean"] = float(np.mean(final_reward_list))
    if fused_reward_list:
        metrics[f"{prefix}rewards/fused_reward_mean"] = float(np.mean(fused_reward_list))
    if penalty_list:
        metrics[f"{prefix}rewards/penalty_mean"] = float(np.mean(penalty_list))
    if step_reward_list:
        metrics[f"{prefix}rewards/step_reward_mean"] = float(np.mean(step_reward_list))
    if non_zero_penalties:
        metrics[f"{prefix}rewards/penalty_count"] = float(len(non_zero_penalties))
        metrics[f"{prefix}rewards/penalty_rate"] = float(len(non_zero_penalties) / n * 100) if n > 0 else 0.0

    # ========== OpenJudge Metrics ==========
    # OpenJudge graders: presentation_quality, grounding, audit
    openjudge_graders = [
        "presentation_quality",
        "grounding",
        "planning",
        "audit",
    ]

    for grader_name in openjudge_graders:
        raw_key = f"openjudge_{grader_name}_raw"
        contrib_key = f"openjudge_{grader_name}_contribution"

        raw_list = [rs.pop(raw_key, 0.0) for rs in reward_stats_list]
        contrib_list = [rs.pop(contrib_key, 0.0) for rs in reward_stats_list]

        # Only report when non-zero values exist
        if any(v != 0.0 for v in raw_list):
            metrics[f"{prefix}rewards/openjudge/{grader_name}_raw_mean"] = float(np.mean(raw_list))
        if any(v != 0.0 for v in contrib_list):
            metrics[f"{prefix}rewards/openjudge/{grader_name}_contribution_mean"] = float(np.mean(contrib_list))

    # OpenJudge time consumption statistics
    grading_time_list = [rs.pop('grading_time', 0.0) for rs in reward_stats_list]
    if any(v != 0.0 for v in grading_time_list):
        metrics[f"{prefix}judge_time/openjudge_grading_time_mean"] = float(np.mean(grading_time_list))
        metrics[f"{prefix}judge_time/openjudge_grading_time_max"] = float(np.max(grading_time_list))

    # ========== Finance Evaluator Metrics ==========
    finance_raw_list = [rs.pop('finance_raw', 0.0) for rs in reward_stats_list]
    finance_contribution_list = [rs.pop('finance_contribution', 0.0) for rs in reward_stats_list]

    if any(v != 0.0 for v in finance_raw_list):
        metrics[f"{prefix}rewards/dimensions/finance_raw_mean"] = float(np.mean(finance_raw_list))

    if any(v != 0.0 for v in finance_contribution_list):
        metrics[f"{prefix}rewards/contribution/finance_contribution_mean"] = float(np.mean(finance_contribution_list))

    # ========== General Time Consumption Statistics ==========
    judge_total_time_list = [rs.pop('judge_total_time', 0.0) for rs in reward_stats_list]
    if any(v != 0.0 for v in judge_total_time_list):
        metrics[f"{prefix}judge_time/judge_total_time_mean"] = float(np.mean(judge_total_time_list))
        metrics[f"{prefix}judge_time/judge_total_time_max"] = float(np.max(judge_total_time_list))

    reward_details = {key: [d[key] for d in reward_stats_list] for key in reward_stats_list[0] if isinstance(reward_stats_list[0][key], (int, float))}
    for key, rewards in reward_details.items():
        rewards = [reward for reward in rewards if reward != -100.0]
        if rewards:
            metrics[f"{prefix}rewards/{key}_min"] = float(np.nanmin(rewards))
            metrics[f"{prefix}rewards/{key}_mean"] = float(np.nanmean(rewards))
            metrics[f"{prefix}rewards/{key}_max"] = float(np.nanmax(rewards))
    return metrics


def compute_reward_metrics_from_trajectories(trajectories: List[Any], prefix: str = "") -> Dict[str, float]:
    """
    Training phase: Extract reward_stats from trajectories and compute metrics.

    Args:
        trajectories: List of trajectory objects

    Returns:
        Formatted metrics dictionary
    """
    reward_stats_list = extract_reward_stats_from_trajectories(trajectories)
    return compute_reward_metrics(reward_stats_list, prefix=prefix)


def populate_reward_metadata_from_stats(reward: "Reward", reward_stats: Dict[str, Any]) -> None:
    """
    Populate Reward.metadata with all reward statistics.

    Args:
        reward: The Reward object to populate
        reward_stats: The reward_stats dictionary from judge
    """
    if not reward_stats:
        return

    # Directly copy all reward_stats into metadata
    reward.metadata.update(reward_stats)
