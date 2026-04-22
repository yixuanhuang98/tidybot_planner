#!/usr/bin/env python3
"""Check residual HDF5 masks against ``convert_to_robomimic_hdf5_residual`` rules.

Uses ``compute_residual_mask_diagnostics`` (same z_diff / acc / mask as the converter)
with ``--z-diff-max`` and ``--acc-move-max``, compares to ``obs/residual_mask`` in the file,
and plots diagnostics (optional).

HDF5 layout: data/demo_<i>/obs/arm_pos, obs/cube*_pos, obs/residual_mask, ...
Per-demo PNG (--plot-demos-dir) includes a gripper subplot: planning (``obs/planning_action`` last
dim), GT = planning + ``actions`` gripper delta (dim 9), and optional ``obs/gripper_pos``.

``recomputed`` mask: same formula as the converter, using ``--z-diff-max`` and
``--acc-move-max``. If you omit them, values are read from ``data/`` attributes
``residual_mask_z_diff_max`` and ``residual_mask_acc_move_max`` (written at
conversion time) so they match the stored ``residual_mask``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from convert_to_robomimic_hdf5_residual import compute_residual_mask_diagnostics

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _cube_pos_keys(obs_group: h5py.Group) -> list[str]:
    return sorted(
        k for k in obs_group.keys() if k.startswith("cube") and k.endswith("_pos")
    )


def min_xy_ee_to_cubes(arm_pos: np.ndarray, obs_group: h5py.Group, t: int) -> float:
    """Min ||EE_xy - cube_xy|| over cubes present at timestep t."""
    arm_xy = np.asarray(arm_pos[t, :2], dtype=np.float64)
    dmin = np.inf
    for ck in _cube_pos_keys(obs_group):
        cube = np.asarray(obs_group[ck][t], dtype=np.float64)
        d = float(np.linalg.norm(arm_xy - cube[:2]))
        dmin = min(dmin, d)
    return dmin if np.isfinite(dmin) else float("nan")


def load_demo_arrays(f: h5py.File, demo_key: str):
    g = f["data"][demo_key]
    obs = g["obs"]
    arm = np.asarray(obs["arm_pos"][:])
    mask = np.asarray(obs["residual_mask"][:]).reshape(-1)
    if arm.shape[0] != mask.shape[0]:
        raise ValueError(f"{demo_key}: arm_pos T={arm.shape[0]} vs residual_mask T={mask.shape[0]}")
    return obs, arm, mask


def observations_list_from_demo(obs: h5py.Group, T: int) -> list[dict]:
    """Per-step dicts with arm_pos + cube*_pos (same shape as EpisodeReader observations)."""
    cube_keys = _cube_pos_keys(obs)
    out: list[dict] = []
    for t in range(T):
        d: dict = {}
        if "arm_pos" in obs:
            d["arm_pos"] = np.asarray(obs["arm_pos"][t], dtype=np.float64)
        for ck in cube_keys:
            d[ck] = np.asarray(obs[ck][t], dtype=np.float64)
        out.append(d)
    return out


def _decode_h5_attr(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def demo_plot_title(h5: h5py.File, demo_key: str, hdf5_stem: str, T: int) -> str:
    """Title for figures: HDF name, demo key, optional source_episode_name from converter attrs."""
    grp = h5["data"][demo_key]
    src = _decode_h5_attr(grp.attrs.get("source_episode_name", None))
    parts = [hdf5_stem, demo_key]
    if src:
        parts.append(src)
    return " / ".join(parts) + f"  (T={T})"


def ee_cube_distances_per_step(arm: np.ndarray, obs_group: h5py.Group) -> np.ndarray:
    """Min XY EE–cube distance for every timestep (same as min_xy_ee_to_cubes loop)."""
    T = arm.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        out[t] = min_xy_ee_to_cubes(arm, obs_group, t)
    return out


def cube_xy_motion_series(obs_group: h5py.Group, T: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-step cube XY motion: max over cubes of ||p_xy[t]-p_xy[t-1]|| (t=0 -> 0).

    Returns (max_step, per_cube_steps) where per_cube_steps has shape (T, n_cubes) or None.
    """
    keys = _cube_pos_keys(obs_group)
    if not keys or T < 1:
        return np.zeros(T, dtype=np.float64), None
    stacks = [np.asarray(obs_group[k][:T], dtype=np.float64) for k in keys]
    P = np.stack(stacks, axis=1)  # (T, n_cubes, 3)
    dxy = np.linalg.norm(np.diff(P[:, :, :2], axis=0), axis=-1)  # (T-1, n_cubes)
    max_step = np.zeros(T, dtype=np.float64)
    max_step[1:] = np.max(dxy, axis=1)
    per_cube = np.zeros((T, dxy.shape[1]), dtype=np.float64)
    per_cube[1:, :] = dxy
    return max_step, per_cube


def gripper_planning_gt_obs_series(
    demo_group: h5py.Group, obs_group: h5py.Group, T: int
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Planning gripper (13D last), GT = planning + delta (actions[..., -1]), optional obs/gripper_pos.

    Returns (planning_g, gt_g, obs_g) each length T or None if not available.
    """
    planning_g: np.ndarray | None = None
    gt_g: np.ndarray | None = None
    obs_g: np.ndarray | None = None

    if "planning_action" in obs_group:
        pa = np.asarray(obs_group["planning_action"][:T], dtype=np.float64)
        if pa.ndim == 2 and pa.shape[1] >= 1:
            planning_g = np.squeeze(pa[:, -1])

    if "actions" in demo_group:
        act = np.asarray(demo_group["actions"][:T], dtype=np.float64)
        if act.ndim == 2 and act.shape[1] >= 10:
            delta_g = np.squeeze(act[:, 9])
            if planning_g is not None and planning_g.shape[0] == delta_g.shape[0]:
                gt_g = planning_g + delta_g

    if "gripper_pos" in obs_group:
        gp = np.asarray(obs_group["gripper_pos"][:T], dtype=np.float64)
        obs_g = np.squeeze(gp)
        if obs_g.ndim != 1:
            obs_g = obs_g.reshape(-1)

    return planning_g, gt_g, obs_g


def z_diff_per_cube_series(arm: np.ndarray, obs_group: h5py.Group, T: int) -> np.ndarray | None:
    """Per-cube z_diff = EE_z_world - cube_top (converter uses min over cubes)."""
    keys = _cube_pos_keys(obs_group)
    if not keys:
        return None
    from constants import EE_Z_OFFSET_ARM_TO_WORLD_Z

    ee_z_world = np.asarray(arm[:T, 2], dtype=np.float64) + float(EE_Z_OFFSET_ARM_TO_WORLD_Z)
    cube_z = np.stack([np.asarray(obs_group[k][:T, 2], dtype=np.float64) for k in keys], axis=1)
    cube_top = 2.0 * cube_z
    return ee_z_world[:, np.newaxis] - cube_top


def plot_demo_timeseries(
    steps: np.ndarray,
    dist: np.ndarray,
    z_min: np.ndarray,
    z_per: np.ndarray | None,
    mask_stored: np.ndarray,
    mask_calc: np.ndarray,
    acc_trace: np.ndarray,
    cube_motion_max: np.ndarray,
    cube_motion_per: np.ndarray | None,
    cube_keys: list[str],
    title: str,
    xy_ref: float,
    z_diff_max: float,
    acc_move_max: float,
    cube_step_ref: float,
    gripper_planning: np.ndarray | None,
    gripper_gt: np.ndarray | None,
    gripper_obs: np.ndarray | None,
    out_path: Path | None,
    show: bool = False,
) -> None:
    """x=step: XY dist, min z-diff (converter), masks, acc, cube XY motion, gripper."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib required for plotting")
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        6, 1, sharex=True, figsize=(10, 14), constrained_layout=True
    )
    ax1.plot(steps, dist, color="C0", lw=0.8, label="min XY EE→cube (m)")
    ax1.axhline(xy_ref, color="gray", ls="--", lw=0.9, label=f"xy_ref={xy_ref}")
    ax1.set_ylabel("XY dist (m)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(title)
    ax2.plot(steps, z_min, color="C3", lw=1.0, label="min_c(EE_z − cube_top)")
    ax2.axhline(z_diff_max, color="gray", ls="--", lw=0.9, label=f"z_diff_max={z_diff_max}")
    if z_per is not None and z_per.shape[1] > 0:
        for j, ck in enumerate(cube_keys):
            ax2.plot(
                steps,
                z_per[:, j],
                lw=0.55,
                alpha=0.55,
                label=f"{ck.replace('_pos', '')} z-diff",
            )
    ax2.set_ylabel("z-diff (m)")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax3.fill_between(steps, 0.0, mask_stored, step="post", color="C1", alpha=0.25, label="mask (HDF5)")
    ax3.plot(steps, mask_stored, color="C1", drawstyle="steps-post", lw=0.9)
    ax3.fill_between(steps, 0.0, mask_calc, step="post", color="C4", alpha=0.25, label="mask (recomputed)")
    ax3.plot(steps, mask_calc, color="C4", drawstyle="steps-post", lw=0.9, ls="--")
    ax3.set_ylabel("mask")
    ax3.set_ylim(-0.05, 1.15)
    ax3.legend(loc="upper right", fontsize=8)
    ax4.plot(steps, acc_trace, color="C5", lw=1.0, label="acc (cube XY after z_start)")
    ax4.axhline(acc_move_max, color="gray", ls="--", lw=0.9, label=f"acc_move_max={acc_move_max}")
    ax4.set_ylabel("acc (m)")
    ax4.legend(loc="upper right", fontsize=8)
    ax5.plot(steps, cube_motion_max, color="C2", lw=1.0, label="max cube ||Δxy|| (m)")
    ax5.axhline(cube_step_ref, color="gray", ls="--", lw=0.9, label=f"cube_step_ref={cube_step_ref}")
    if cube_motion_per is not None and cube_motion_per.shape[1] > 0:
        for j, ck in enumerate(cube_keys):
            ax5.plot(
                steps,
                cube_motion_per[:, j],
                lw=0.55,
                alpha=0.55,
                label=ck.replace("_pos", ""),
            )
    ax5.set_ylabel("cube motion (m)")
    ax5.legend(loc="upper right", fontsize=7, ncol=2)
    if gripper_planning is not None:
        ax6.plot(steps, gripper_planning, color="C0", lw=0.75, label="planning gripper (obs/planning_action[-1])")
    if gripper_gt is not None:
        ax6.plot(steps, gripper_gt, color="C1", lw=1,ls="dotted", label="GT gripper (planning + actions Δ)")
    if gripper_obs is not None:
        ax6.plot(steps, gripper_obs, color="C2", lw=1.75, ls="--", alpha=0.9, label="obs/gripper_pos (state)")
    if gripper_planning is None and gripper_gt is None and gripper_obs is None:
        ax6.text(
            0.5,
            0.5,
            "No gripper series (need obs/planning_action and demo/actions for planning+GT)",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=9,
        )
    ax6.set_ylabel("gripper")
    ax6.set_xlabel("step")
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend(loc="upper right", fontsize=7)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hdf5", type=Path, help="Path to .hdf5 file")
    p.add_argument(
        "--z-diff-max",
        type=float,
        default=None,
        help="Override mask param; if omitted, use data/residual_mask_z_diff_max from HDF5, else 0.05",
    )
    p.add_argument(
        "--acc-move-max",
        type=float,
        default=None,
        help="Override mask param; if omitted, use data/residual_mask_acc_move_max from HDF5, else 0.12",
    )
    p.add_argument(
        "--xy-ref",
        type=float,
        default=0.11,
        help="Reference line on XY-distance subplot only (not used in mask)",
    )
    p.add_argument(
        "--cube-step-ref",
        type=float,
        default=0.003,
        help="Optional horizontal line on per-step cube motion subplot",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="If set, save z_diff histogram (mask==0 vs 1) + z_diff_max line",
    )
    p.add_argument(
        "--plot-demos-dir",
        type=Path,
        default=None,
        help="Directory to save one PNG per demo (mask, z/acc, cube motion, gripper planning vs GT)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="With --plot-demos-dir, plt.show() each figure (blocks until closed)",
    )
    args = p.parse_args()

    total_steps = 0
    mismatch_steps = 0
    all_z: list[np.ndarray] = []
    all_acc: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    demo_mask_active_frac: list[float] = []
    demo_longest_mask_run: list[int] = []

    with h5py.File(args.hdf5, "r") as f:
        if "data" not in f:
            raise SystemExit("Expected top-level group 'data' (robomimic-style).")
        dg = f["data"]
        z_attr = dg.attrs.get("residual_mask_z_diff_max", None)
        a_attr = dg.attrs.get("residual_mask_acc_move_max", None)
        z_diff_max = (
            float(args.z_diff_max)
            if args.z_diff_max is not None
            else (float(z_attr) if z_attr is not None else 0.05)
        )
        acc_move_max = (
            float(args.acc_move_max)
            if args.acc_move_max is not None
            else (float(a_attr) if a_attr is not None else 0.12)
        )
        src_z = "CLI" if args.z_diff_max is not None else ("HDF5" if z_attr is not None else "default")
        src_a = "CLI" if args.acc_move_max is not None else ("HDF5" if a_attr is not None else "default")
        print(
            f"Mask params: z_diff_max={z_diff_max} ({src_z}), acc_move_max={acc_move_max} ({src_a})"
        )
        if z_attr is None and args.z_diff_max is None:
            print("Note: HDF5 has no residual_mask_z_diff_max; using default 0.05 (re-run converter to embed).")
        if a_attr is None and args.acc_move_max is None:
            print("Note: HDF5 has no residual_mask_acc_move_max; using default 0.12 (re-run converter to embed).")

        demo_keys = sorted(k for k in dg.keys() if k.startswith("demo_"))
        if not demo_keys:
            raise SystemExit("No demo_* groups under data/")

        for dk in demo_keys:
            demo_grp = f["data"][dk]
            obs, arm, mask_stored = load_demo_arrays(f, dk)
            T = arm.shape[0]
            dist_arr = ee_cube_distances_per_step(arm, obs)
            obs_list = observations_list_from_demo(obs, T)
            mask_calc, z_trace, acc_trace = compute_residual_mask_diagnostics(
                obs_list, z_diff_max, acc_move_max
            )
            mask_calc = np.asarray(mask_calc, dtype=np.float64).reshape(-1)
            mask_st = np.asarray(mask_stored, dtype=np.float64).reshape(-1)
            agree = (mask_st > 0.5) == (mask_calc > 0.5)
            total_steps += T
            mismatch_steps += int((~agree).sum())

            z_arr = np.asarray(z_trace, dtype=np.float64).reshape(-1)
            acc_arr = np.asarray(acc_trace, dtype=np.float64).reshape(-1)
            all_z.append(z_arr)
            all_acc.append(acc_arr)
            all_mask.append(mask_st.astype(np.float64).reshape(-1))
            demo_mask_active_frac.append(float(np.mean(mask_st > 0.5)))
            mbin = (mask_st > 0.5).astype(np.int32)
            run = best = 0
            for v in mbin:
                if v:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            demo_longest_mask_run.append(int(best))

            if args.plot_demos_dir is not None:
                if not HAS_MPL:
                    print("matplotlib not installed; skip --plot-demos-dir")
                else:
                    args.plot_demos_dir.mkdir(parents=True, exist_ok=True)
                    steps = np.arange(T, dtype=np.int32)
                    out_file = args.plot_demos_dir / f"{dk}.png"
                    ckeys = _cube_pos_keys(obs)
                    z_per = z_diff_per_cube_series(arm, obs, T)
                    z_min = np.asarray(z_trace, dtype=np.float64)
                    mot_max, mot_per = cube_xy_motion_series(obs, T)
                    g_plan, g_gt, g_obs = gripper_planning_gt_obs_series(demo_grp, obs, T)
                    plot_demo_timeseries(
                        steps,
                        dist_arr,
                        z_min,
                        z_per,
                        mask_st.astype(np.float64),
                        mask_calc,
                        acc_trace.astype(np.float64),
                        mot_max,
                        mot_per,
                        ckeys,
                        title=demo_plot_title(f, dk, args.hdf5.name, T),
                        xy_ref=args.xy_ref,
                        z_diff_max=z_diff_max,
                        acc_move_max=acc_move_max,
                        cube_step_ref=args.cube_step_ref,
                        gripper_planning=g_plan,
                        gripper_gt=g_gt,
                        gripper_obs=g_obs,
                        out_path=out_file,
                        show=args.show,
                    )
                    if np.any(~agree):
                        n_bad = int((~agree).sum())
                        print(f"{dk}: mask mismatch vs recomputed on {n_bad}/{T} steps")
                    print(f"Saved {out_file}")

    z_all = np.concatenate(all_z) if all_z else np.array([])
    acc_all = np.concatenate(all_acc) if all_acc else np.array([])
    mask_all = np.concatenate(all_mask) if all_mask else np.array([])
    m1 = mask_all > 0.5
    m0 = ~m1

    print(f"File: {args.hdf5}")
    print(
        f"Mask check (z_diff_max={z_diff_max}, acc_move_max={acc_move_max}): "
        f"agree {total_steps - mismatch_steps}/{total_steps} steps "
        f"({100.0 * (1.0 - mismatch_steps / max(total_steps, 1)):.2f}% match)"
    )
    n = mask_all.size
    print(f"Demos: {len(demo_mask_active_frac)}  steps: {n}  mask==1: {int(m1.sum())} ({100.0 * m1.mean():.1f}%)")
    if demo_mask_active_frac:
        daf = np.asarray(demo_mask_active_frac, dtype=np.float64)
        dlr = np.asarray(demo_longest_mask_run, dtype=np.float64)
        print(
            f"Per-demo mask==1 fraction: mean={daf.mean():.3f}  min={daf.min():.3f}  max={daf.max():.3f}"
        )
        print(
            f"Per-demo longest consecutive mask==1 (steps): mean={dlr.mean():.1f}  "
            f"min={int(dlr.min())}  max={int(dlr.max())}"
        )
    print()

    def summarize(name: str, x: np.ndarray):
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"{name}: (empty)")
            return
        print(
            f"{name}: n={x.size}  mean={x.mean():.4f}  std={x.std():.4f}  "
            f"min={x.min():.4f}  p50={np.percentile(x, 50):.4f}  "
            f"p90={np.percentile(x, 90):.4f}  max={x.max():.4f}"
        )

    print("z_diff = min_c(EE_z - cube_top) [m]")
    summarize("  [all steps]", z_all)
    summarize("  [mask==0]", z_all[m0])
    summarize("  [mask==1]", z_all[m1])
    print(f"  (threshold line z_diff_max={z_diff_max})")
    print()
    print("acc = cumulative cube XY travel after first z_ok [m]")
    summarize("  [all steps]", acc_all)
    summarize("  [mask==0]", acc_all[m0])
    summarize("  [mask==1]", acc_all[m1])
    print(f"  (cap acc_move_max={acc_move_max})")

    if args.out_png and HAS_MPL:
        plt.figure(figsize=(8, 4))
        z0 = z_all[m0]
        z1 = z_all[m1]
        parts = []
        if z0.size:
            parts.append(float(np.nanmin(z0)))
            parts.append(float(np.nanmax(z0)))
        if z1.size:
            parts.append(float(np.nanmin(z1)))
            parts.append(float(np.nanmax(z1)))
        parts.append(float(z_diff_max))
        lo = min(parts) if parts else -0.2
        hi = max(parts) if parts else 0.1
        if not np.isfinite(lo):
            lo = -0.2
        if not np.isfinite(hi):
            hi = 0.1
        if hi <= lo:
            hi = lo + 0.05
        bins = np.linspace(lo, hi, 56)
        if z0.size:
            plt.hist(z0, bins=bins, alpha=0.55, label="mask==0", density=True)
        if z1.size:
            plt.hist(z1, bins=bins, alpha=0.55, label="mask==1", density=True)
        plt.axvline(z_diff_max, color="k", ls="--", label=f"z_diff_max={z_diff_max}")
        plt.xlabel("z_diff (m)")
        plt.ylabel("density")
        plt.title("z_diff distribution by residual_mask")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        print(f"\nSaved figure: {args.out_png}")
    elif args.out_png and not HAS_MPL:
        print("matplotlib not installed; skip --out-png")


if __name__ == "__main__":
    main()
