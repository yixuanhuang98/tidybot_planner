#!/usr/bin/env python3
"""Evaluate residual-learning dataset quality per episode.

Metrics:
1) fallback ratio
   - exact ratio when `fallback_planning_action_count` exists in data.pkl
   - heuristic estimate otherwise
2) residual magnitude mean
3) per-component correction magnitude:
   - base pose error (L2 over 3D)
   - arm position error (L2 over 3D)
   - arm rotation error (angle in radians)
   - gripper error (absolute)
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation


def _safe_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _action_to_delta_components(
    gt_action: Dict[str, Any], planning_action: Dict[str, Any]
) -> Dict[str, float]:
    base_delta = _safe_array(gt_action["base_pose"]) - _safe_array(planning_action["base_pose"])
    arm_pos_delta = _safe_array(gt_action["arm_pos"]) - _safe_array(planning_action["arm_pos"])
    gripper_delta = float(
        np.squeeze(_safe_array(gt_action["gripper_pos"]) - _safe_array(planning_action["gripper_pos"]))
    )

    r_gt = Rotation.from_quat(_safe_array(gt_action["arm_quat"]))
    r_plan = Rotation.from_quat(_safe_array(planning_action["arm_quat"]))
    arm_rotvec_delta = (r_gt * r_plan.inv()).as_rotvec()

    delta10 = np.concatenate(
        [base_delta, arm_pos_delta, arm_rotvec_delta, np.array([gripper_delta], dtype=np.float64)]
    )
    return {
        "residual_norm": float(np.linalg.norm(delta10)),
        "base_l2": float(np.linalg.norm(base_delta)),
        "arm_pos_l2": float(np.linalg.norm(arm_pos_delta)),
        "arm_rot_angle": float(np.linalg.norm(arm_rotvec_delta)),
        "gripper_abs": float(abs(gripper_delta)),
    }


def _load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _estimate_fallback_ratio(actions: List[Dict[str, Any]], planning_actions: List[Dict[str, Any]]) -> float:
    """Heuristic fallback estimation when explicit count is unavailable.

    We estimate fallback when planner action is exactly unchanged from previous step
    while human action changed noticeably.
    """
    if len(actions) < 2 or len(planning_actions) < 2:
        return 0.0

    fallback_hits = 0
    for i in range(1, min(len(actions), len(planning_actions))):
        pa_prev = planning_actions[i - 1]
        pa_curr = planning_actions[i]
        gt_prev = actions[i - 1]
        gt_curr = actions[i]

        pa_same = (
            np.allclose(pa_curr["base_pose"], pa_prev["base_pose"])
            and np.allclose(pa_curr["arm_pos"], pa_prev["arm_pos"])
            and np.allclose(pa_curr["arm_quat"], pa_prev["arm_quat"])
            and np.allclose(pa_curr["gripper_pos"], pa_prev["gripper_pos"])
        )
        gt_changed = not (
            np.allclose(gt_curr["base_pose"], gt_prev["base_pose"])
            and np.allclose(gt_curr["arm_pos"], gt_prev["arm_pos"])
            and np.allclose(gt_curr["arm_quat"], gt_prev["arm_quat"])
            and np.allclose(gt_curr["gripper_pos"], gt_prev["gripper_pos"])
        )
        if pa_same and gt_changed:
            fallback_hits += 1
    return float(fallback_hits / len(actions))


@dataclass
class EpisodeMetrics:
    episode_name: str
    steps: int
    fallback_ratio: float
    fallback_source: str
    residual_norm_mean: float
    base_l2_mean: float
    arm_pos_l2_mean: float
    arm_rot_angle_mean: float
    gripper_abs_mean: float
    quality_label: str = "unknown"


@dataclass
class LabelThresholds:
    fallback_good_max: float
    fallback_bad_min: float
    residual_good_max: float
    residual_bad_min: float
    base_good_max: float
    base_bad_min: float


def _assign_quality_label(m: EpisodeMetrics, th: LabelThresholds) -> str:
    # Conservative quality gate:
    # - good: low fallback, low residual norm, low base override
    # - bad: clear replacement-like signals
    # - suspect: in between
    if (
        m.fallback_ratio >= th.fallback_bad_min
        or m.residual_norm_mean >= th.residual_bad_min
        or m.base_l2_mean >= th.base_bad_min
    ):
        return "bad"
    if (
        m.fallback_ratio <= th.fallback_good_max
        and m.residual_norm_mean <= th.residual_good_max
        and m.base_l2_mean <= th.base_good_max
    ):
        return "good"
    return "suspect"


def compute_episode_metrics(episode_dir: Path) -> Optional[EpisodeMetrics]:
    data_path = episode_dir / "data.pkl"
    if not data_path.exists():
        return None

    data = _load_pickle(data_path)
    actions = data.get("actions", [])
    planning_actions = data.get("planning_actions", None)
    if planning_actions is None:
        return None
    if len(actions) == 0 or len(planning_actions) == 0:
        return None

    n = min(len(actions), len(planning_actions))
    component_rows = [_action_to_delta_components(actions[i], planning_actions[i]) for i in range(n)]

    explicit_fallback = data.get("fallback_planning_action_count", None)
    if explicit_fallback is not None:
        fallback_ratio = float(explicit_fallback / max(len(actions), 1))
        fallback_source = "explicit"
    else:
        fallback_ratio = _estimate_fallback_ratio(actions, planning_actions)
        fallback_source = "heuristic"

    return EpisodeMetrics(
        episode_name=episode_dir.name,
        steps=len(actions),
        fallback_ratio=fallback_ratio,
        fallback_source=fallback_source,
        residual_norm_mean=float(np.mean([r["residual_norm"] for r in component_rows])),
        base_l2_mean=float(np.mean([r["base_l2"] for r in component_rows])),
        arm_pos_l2_mean=float(np.mean([r["arm_pos_l2"] for r in component_rows])),
        arm_rot_angle_mean=float(np.mean([r["arm_rot_angle"] for r in component_rows])),
        gripper_abs_mean=float(np.mean([r["gripper_abs"] for r in component_rows])),
    )


def _fmt_ratio(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _print_table(metrics: Iterable[EpisodeMetrics]) -> None:
    rows = list(metrics)
    if not rows:
        print("No residual episodes found.")
        return

    header = (
        "episode".ljust(22)
        + "steps".rjust(8)
        + "fallback".rjust(12)
        + "src".rjust(10)
        + "res_norm".rjust(12)
        + "base_l2".rjust(11)
        + "arm_l2".rjust(10)
        + "rot_rad".rjust(10)
        + "grip_abs".rjust(11)
        + "label".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for m in rows:
        print(
            m.episode_name.ljust(22)
            + str(m.steps).rjust(8)
            + _fmt_ratio(m.fallback_ratio).rjust(12)
            + m.fallback_source.rjust(10)
            + f"{m.residual_norm_mean:.4f}".rjust(12)
            + f"{m.base_l2_mean:.4f}".rjust(11)
            + f"{m.arm_pos_l2_mean:.4f}".rjust(10)
            + f"{m.arm_rot_angle_mean:.4f}".rjust(10)
            + f"{m.gripper_abs_mean:.4f}".rjust(11)
            + m.quality_label.rjust(10)
        )

    print("\nSummary")
    print("-------")
    print(f"Episodes: {len(rows)}")
    print(f"Mean fallback ratio: {_fmt_ratio(float(np.mean([m.fallback_ratio for m in rows])))}")
    print(f"Mean residual norm: {float(np.mean([m.residual_norm_mean for m in rows])):.4f}")
    print(f"Mean base_l2: {float(np.mean([m.base_l2_mean for m in rows])):.4f}")
    print(f"Mean arm_pos_l2: {float(np.mean([m.arm_pos_l2_mean for m in rows])):.4f}")
    print(f"Mean arm_rot_angle(rad): {float(np.mean([m.arm_rot_angle_mean for m in rows])):.4f}")
    print(f"Mean gripper_abs: {float(np.mean([m.gripper_abs_mean for m in rows])):.4f}")
    print(f"Label counts: good={sum(m.quality_label == 'good' for m in rows)}, "
          f"suspect={sum(m.quality_label == 'suspect' for m in rows)}, "
          f"bad={sum(m.quality_label == 'bad' for m in rows)}")


def _write_csv(metrics: Iterable[EpisodeMetrics], csv_path: Path) -> None:
    rows = list(metrics)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "steps",
                "fallback_ratio",
                "fallback_source",
                "residual_norm_mean",
                "base_l2_mean",
                "arm_pos_l2_mean",
                "arm_rot_angle_mean",
                "gripper_abs_mean",
                "quality_label",
            ]
        )
        for m in rows:
            writer.writerow(
                [
                    m.episode_name,
                    m.steps,
                    m.fallback_ratio,
                    m.fallback_source,
                    m.residual_norm_mean,
                    m.base_l2_mean,
                    m.arm_pos_l2_mean,
                    m.arm_rot_angle_mean,
                    m.gripper_abs_mean,
                    m.quality_label,
                ]
            )
    print(f"CSV saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/residual_demos_v2")
    parser.add_argument(
        "--contains",
        default="",
        help="Only include episode names containing this substring.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional path to export episode metrics as CSV.",
    )
    parser.add_argument("--fallback-good-max", type=float, default=0.005)
    parser.add_argument("--fallback-bad-min", type=float, default=0.02)
    parser.add_argument("--residual-good-max", type=float, default=0.20)
    parser.add_argument("--residual-bad-min", type=float, default=0.35)
    parser.add_argument("--base-good-max", type=float, default=0.10)
    parser.add_argument("--base-bad-min", type=float, default=0.15)
    args = parser.parse_args()

    thresholds = LabelThresholds(
        fallback_good_max=args.fallback_good_max,
        fallback_bad_min=args.fallback_bad_min,
        residual_good_max=args.residual_good_max,
        residual_bad_min=args.residual_bad_min,
        base_good_max=args.base_good_max,
        base_bad_min=args.base_bad_min,
    )

    root = Path(args.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input dir does not exist: {root}")

    episode_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.contains:
        episode_dirs = [p for p in episode_dirs if args.contains in p.name]

    metrics: List[EpisodeMetrics] = []
    skipped = 0
    for episode_dir in episode_dirs:
        try:
            m = compute_episode_metrics(episode_dir)
            if m is None:
                skipped += 1
                continue
            m.quality_label = _assign_quality_label(m, thresholds)
            metrics.append(m)
        except Exception as exc:
            skipped += 1
            print(f"Warning: failed to evaluate {episode_dir.name}: {exc}")

    _print_table(metrics)
    if args.csv_out:
        _write_csv(metrics, Path(args.csv_out))
    if skipped > 0:
        print(f"\nSkipped episodes: {skipped}")


if __name__ == "__main__":
    main()
