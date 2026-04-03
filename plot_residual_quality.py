#!/usr/bin/env python3
"""Plot residual dataset quality diagnostics from CSV metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_metrics(csv_path: Path) -> Dict[str, List]:
    rows = {
        "episode": [],
        "fallback_ratio": [],
        "residual_norm_mean": [],
        "base_l2_mean": [],
        "gripper_abs_mean": [],
        "quality_label": [],
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows["episode"].append(r["episode"])
            rows["fallback_ratio"].append(float(r["fallback_ratio"]))
            rows["residual_norm_mean"].append(float(r["residual_norm_mean"]))
            rows["base_l2_mean"].append(float(r["base_l2_mean"]))
            rows["gripper_abs_mean"].append(float(r["gripper_abs_mean"]))
            rows["quality_label"].append(r["quality_label"])
    return rows


def _plot_hist(data: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 24) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=bins, alpha=0.85, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_scatter(base_l2: np.ndarray, gripper_abs: np.ndarray, labels: List[str], out_path: Path) -> None:
    color_map = {"good": "#2ca02c", "suspect": "#ff7f0e", "bad": "#d62728"}
    plt.figure(figsize=(7, 5))
    for label in ["good", "suspect", "bad"]:
        idx = [i for i, x in enumerate(labels) if x == label]
        if not idx:
            continue
        plt.scatter(
            base_l2[idx],
            gripper_abs[idx],
            s=28,
            alpha=0.8,
            label=f"{label} ({len(idx)})",
            color=color_map[label],
        )
    plt.title("base_l2 vs gripper_abs")
    plt.xlabel("base_l2_mean")
    plt.ylabel("gripper_abs_mean")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_label_pie(labels: List[str], out_path: Path) -> None:
    names = ["good", "suspect", "bad"]
    counts = [sum(1 for x in labels if x == n) for n in names]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts,
        labels=[f"{n} ({c})" for n, c in zip(names, counts)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Quality label distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/residual_metrics_all.csv")
    parser.add_argument("--output-dir", default="data/residual_quality_plots")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_metrics(input_csv)
    if len(rows["episode"]) == 0:
        raise ValueError(f"No rows found in CSV: {input_csv}")

    residual_norm = np.array(rows["residual_norm_mean"], dtype=np.float64)
    fallback_ratio = np.array(rows["fallback_ratio"], dtype=np.float64)
    base_l2 = np.array(rows["base_l2_mean"], dtype=np.float64)
    gripper_abs = np.array(rows["gripper_abs_mean"], dtype=np.float64)

    _plot_hist(
        residual_norm,
        title="Residual Norm Distribution",
        xlabel="residual_norm_mean",
        out_path=out_dir / "residual_norm_distribution.png",
    )
    _plot_hist(
        fallback_ratio,
        title="Fallback Ratio Distribution",
        xlabel="fallback_ratio",
        out_path=out_dir / "fallback_ratio_distribution.png",
    )
    _plot_scatter(
        base_l2=base_l2,
        gripper_abs=gripper_abs,
        labels=rows["quality_label"],
        out_path=out_dir / "base_l2_vs_gripper_abs.png",
    )
    _plot_label_pie(
        labels=rows["quality_label"],
        out_path=out_dir / "quality_label_distribution.png",
    )

    print(f"Saved plots to: {out_dir}")
    print("- residual_norm_distribution.png")
    print("- fallback_ratio_distribution.png")
    print("- base_l2_vs_gripper_abs.png")
    print("- quality_label_distribution.png")


if __name__ == "__main__":
    main()
