#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import multiprocessing
import os
import re
import sys
from pathlib import Path

import numpy as np

from tqdm import tqdm

from price_simulator.src.algorithm.demand import LogitDemand
from price_simulator.src.algorithm.environment import ContSynchronEnvironment
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from price_simulator.src.algorithm.policies import EpsilonGreedy

TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
STEP_RE = re.compile(r"_step(\d+)", re.IGNORECASE)
IR_SETTLE_PERIODS = 50
DEVIATION_HORIZON_T = 100
MULTI_PERIOD_DEVIATION_LENGTH = 5
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_GRID_DEVIATION_POINTS = 15
_TF = None


def configure_tensorflow_runtime(gpu_id: str | None) -> None:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def tensorflow_module():
    global _TF
    if _TF is None:
        import tensorflow as tf

        _TF = tf
    return _TF


def sac_dependencies():
    from price_simulator.src.algorithm.agents.sac import SACContinuous
    from price_simulator.src.algorithm.agents.sac import build_sac_kwargs

    return SACContinuous, build_sac_kwargs


def rollout_deviation(
    env: ContSynchronEnvironment,
    current_state_tf: tf.Tensor,
    qualities: tuple[float, ...],
    marginal_costs: np.ndarray,
    defector_idx: int,
    forced_action_norm: float,
    forced_length: int,
) -> np.ndarray:
    tf = tensorflow_module()
    dev_profits = []
    state_dev_tf = current_state_tf
    for t in range(DEVIATION_HORIZON_T):
        dev_actions = []
        dev_actions_tf = []
        for i, a in enumerate(env.agents):
            if t < forced_length and i == defector_idx:
                dev_actions.append(forced_action_norm)
                dev_actions_tf.append(
                    tf.constant([[forced_action_norm]], dtype=tf.float32)
                )
            else:
                action_tf, _ = a._sample_action(
                    state_dev_tf, deterministic=True, seed_step=None
                )
                dev_actions.append(float(action_tf.numpy().reshape(-1)[0]))
                dev_actions_tf.append(action_tf)
        dev_real_prices = tuple(env._denormalize_action(a) for a in dev_actions)
        dev_qs = env.demand.get_quantities(dev_real_prices, qualities)
        dev_rews = tuple(
            np.multiply(np.subtract(dev_real_prices, marginal_costs), dev_qs)
        )
        dev_profits.append(dev_rews)
        state_dev_tf = tf.concat(dev_actions_tf, axis=1)
    return np.asarray(dev_profits, dtype=np.float32)


def rollout_baseline(
    env: ContSynchronEnvironment,
    current_state_tf: tf.Tensor,
    qualities: tuple[float, ...],
    marginal_costs: np.ndarray,
) -> np.ndarray:
    tf = tensorflow_module()
    base_profits = []
    state_base_tf = current_state_tf
    for _ in range(DEVIATION_HORIZON_T):
        base_actions = []
        base_actions_tf = []
        for a in env.agents:
            action_tf, _ = a._sample_action(
                state_base_tf, deterministic=True, seed_step=None
            )
            base_actions.append(float(action_tf.numpy().reshape(-1)[0]))
            base_actions_tf.append(action_tf)
        base_real_prices = tuple(env._denormalize_action(a) for a in base_actions)
        base_qs = env.demand.get_quantities(base_real_prices, qualities)
        base_rews = tuple(
            np.multiply(np.subtract(base_real_prices, marginal_costs), base_qs)
        )
        base_profits.append(base_rews)
        state_base_tf = tf.concat(base_actions_tf, axis=1)
    return np.asarray(base_profits, dtype=np.float32)


def differential_gain(
    dev_arr: np.ndarray, base_arr: np.ndarray, defector_idx: int
) -> float:
    diff_col = dev_arr[:, defector_idx] - base_arr[:, defector_idx]
    return float(np.mean(diff_col / base_arr[:, defector_idx]))


def relative_discounted_gain(
    dev_arr: np.ndarray,
    base_arr: np.ndarray,
    defector_idx: int,
    discount_factor: float,
) -> float:
    diff_col = dev_arr[:, defector_idx] - base_arr[:, defector_idx]
    weights = np.power(discount_factor, np.arange(diff_col.shape[0], dtype=np.float32))
    discounted_gain = float(np.sum(diff_col * weights))
    discounted_base = float(np.sum(base_arr[:, defector_idx] * weights))
    return float(discounted_gain / discounted_base)


def format_deviation_table(step_diff_gains: dict[int, list[float]]) -> str:
    rng = np.random.default_rng(0)
    lines = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append(
        "Step & 25th percentile (\\%) & Median (\\%) "
        "& 75th percentile (\\%) & Mean (\\%) "
        "& Unprofitable \\% [95\\% CI] \\\\"
    )
    lines.append("\\hline")

    for step in sorted(step_diff_gains):
        diff_vals = np.asarray(step_diff_gains[step], dtype=np.float32)
        if diff_vals.size == 0:
            continue
        p25_diff, p50_diff, p75_diff = np.percentile(diff_vals, [25, 50, 75])
        mean_diff = float(np.mean(diff_vals))
        unprofit_mask = diff_vals <= 0.0
        pct_diff = 100.0 * float(np.mean(unprofit_mask))
        boot_indices = rng.integers(
            0, unprofit_mask.size, size=(5000, unprofit_mask.size)
        )
        boot_props = unprofit_mask[boot_indices].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_props, [2.5, 97.5])
        step_label = f"\\num{{{step}}}"

        p25_diff_pct = 100.0 * p25_diff
        p50_diff_pct = 100.0 * p50_diff
        p75_diff_pct = 100.0 * p75_diff
        mean_diff_pct = 100.0 * mean_diff

        p25_cell = f"{p25_diff_pct:.2g}\\%"
        p50_cell = f"{p50_diff_pct:.2g}\\%"
        p75_cell = f"{p75_diff_pct:.2g}\\%"
        pct_cell = f"{pct_diff:.2g}\\%"
        ci_cell = f"{100.0 * ci_low:.2g}\\%--{100.0 * ci_high:.2g}\\%"
        pct_ci_cell = f"{pct_cell} [{ci_cell}]"

        lines.append(
            f"{step_label} & {p25_cell} & {p50_cell} & {p75_cell} "
            f"& {mean_diff_pct:.2g}\\% & {pct_ci_cell} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def nearest_grid_index(price_grid: np.ndarray, price: float) -> int:
    return int(np.argmin(np.abs(price_grid - price)))


def format_price_cell(value: float) -> str:
    return f"{value:.2f}"


def format_grid_deviation_table(
    grid_prices: np.ndarray,
    pre_price_counts: dict[int, int],
    grid_discounted_gains: dict[int, dict[int, list[float]]],
    grid_br_comparison_counts: dict[int, list[int]],
) -> str:
    total_pre_prices = sum(pre_price_counts.values())
    n_price_cols = int(grid_prices.size)
    if total_pre_prices == 0:
        return "% No grid-deviation observations were available.\n"

    col_spec = "lc" + ("c" * n_price_cols)
    price_header = " & ".join(format_price_cell(p) for p in grid_prices)
    row_indices = [idx for idx in range(n_price_cols) if pre_price_counts.get(idx, 0)]

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(
        f"\\multicolumn{{{n_price_cols + 2}}}{{c}}"
        "{Panel a: Average percentage gain from the deviation "
        "in terms of discounted profits} \\\\"
    )
    lines.append("\\hline")
    lines.append(f" & & \\multicolumn{{{n_price_cols}}}{{c}}{{Deviation price}} \\\\")
    lines.append(f"Pre-shock price & Freq. & {price_header} \\\\")
    lines.append("\\hline")
    for row_idx in row_indices:
        row_gains = grid_discounted_gains.get(row_idx, {})
        freq = pre_price_counts[row_idx] / total_pre_prices
        cells = []
        for col_idx in range(n_price_cols):
            gains = row_gains.get(col_idx, [])
            if gains:
                cells.append(f"{100.0 * float(np.mean(gains)):.2f}")
            else:
                cells.append("")
        lines.append(
            f"{format_price_cell(grid_prices[row_idx])} & {freq:.2f} & "
            + " & ".join(cells)
            + " \\\\"
        )

    lines.append("\\hline")
    lines.append(
        f"\\multicolumn{{{n_price_cols + 2}}}{{c}}"
        "{Panel b: Frequency of unprofitable deviations.} \\\\"
    )
    lines.append("\\hline")
    lines.append(f" & & \\multicolumn{{{n_price_cols}}}{{c}}{{Deviation price}} \\\\")
    lines.append(f"Pre-shock price & Freq. & {price_header} \\\\")
    lines.append("\\hline")
    for row_idx in row_indices:
        row_gains = grid_discounted_gains.get(row_idx, {})
        freq = pre_price_counts[row_idx] / total_pre_prices
        cells = []
        for col_idx in range(n_price_cols):
            gains = row_gains.get(col_idx, [])
            if gains:
                gain_arr = np.asarray(gains, dtype=np.float32)
                cells.append(f"{float(np.mean(gain_arr <= 0.0)):.2f}")
            else:
                cells.append("")
        lines.append(
            f"{format_price_cell(grid_prices[row_idx])} & {freq:.2f} & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    consistent_prices = []
    best_price_idx = None
    best_share = -1.0
    for price_idx in range(n_price_cols):
        wins, total = grid_br_comparison_counts.get(price_idx, [0, 0])
        if total == 0:
            continue
        share = wins / total
        if share > best_share:
            best_share = share
            best_price_idx = price_idx
        if wins == total:
            consistent_prices.append(grid_prices[price_idx])

    lines.append("")
    lines.append("\\medskip")
    lines.append("\\noindent\\textit{Static best-response comparison.} ")
    if consistent_prices:
        prices_text = ", ".join(format_price_cell(p) for p in consistent_prices)
        lines.append(
            "The following fixed-grid deviation prices are more profitable than "
            f"the static best response in every matched observation: {prices_text}."
        )
    elif best_price_idx is not None:
        lines.append(
            "No fixed-grid deviation price is more profitable than the static "
            "best response in every matched observation. The highest share is "
            f"{100.0 * best_share:.2f}\\%, at deviation price "
            f"{format_price_cell(grid_prices[best_price_idx])}."
        )
    else:
        lines.append(
            "No fixed-grid deviation price has matched static-best-response "
            "comparisons."
        )

    return "\n".join(lines) + "\n"


def empty_timestamp_result(warnings: list[str] | None = None) -> dict[str, object]:
    return {
        "step_diff_gains": {},
        "multi_period_step_diff_gains": {},
        "grid_price_values": None,
        "grid_pre_price_counts": {},
        "grid_discounted_gains": {},
        "grid_br_comparison_counts": {},
        "raw_rows": [],
        "warnings": warnings or [],
    }


def analyze_timestamp(
    ts: str,
    artifacts_dir: Path,
    checkpoints_dir: Path,
    settle_periods: int,
    pre_window_periods: int,
    grid_deviation_points: int,
) -> dict[str, object]:
    tf = tensorflow_module()
    SACContinuous, build_sac_kwargs = sac_dependencies()
    warnings = []
    step_diff_gains: dict[int, list[float]] = {}
    multi_period_step_diff_gains: dict[int, list[float]] = {}
    grid_price_values: np.ndarray | None = None
    grid_pre_price_counts: dict[int, int] = {}
    grid_discounted_gains: dict[int, dict[int, list[float]]] = {}
    grid_br_comparison_counts: dict[int, list[int]] = {}
    raw_rows = []
    agent_kwargs = build_sac_kwargs()

    try:
        price_files = sorted(artifacts_dir.glob(f"*_prices_{ts}.npy"))
        arrays = []
        for f in price_files:
            arr = np.asarray(np.load(f))
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    arr = arr.mean(axis=1)
            arrays.append(arr)
        if not arrays:
            warnings.append(f"Warning: no price artifacts for run {ts}.")
            return empty_timestamp_result(warnings)
        lengths = [arr.shape[0] for arr in arrays]
        min_len = min(lengths)
        if any(length != min_len for length in lengths):
            arrays = [arr[:min_len] for arr in arrays]
        prices = np.stack(arrays).T
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        step_map: dict[int, dict[str, object]] = {}
        for cp_file in checkpoints_dir.glob(f"*_{ts}_step*.weights.h5"):
            step_match = STEP_RE.search(cp_file.name)
            if not step_match:
                continue
            step = int(step_match.group(1))
            if step not in step_map:
                step_map[step] = {"step": step}
            parts = cp_file.name.split("_")
            agent_id = parts[0]
            component_type = cp_file.name.split("_")[-1].replace(".weights.h5", "")
            key_name = f"{agent_id}_{component_type}"
            step_map[step][key_name] = cp_file

        for checkpoint in (step_map[k] for k in sorted(step_map)):
            step_value = checkpoint["step"]
            if not isinstance(step_value, int):
                continue
            step = step_value
            actor_items = []
            for key, value in checkpoint.items():
                if not isinstance(value, Path):
                    continue
                if not key.endswith("_actor"):
                    continue
                match = re.search(r"agent(\d+)_", key)
                agent_idx = int(match.group(1)) if match else 999
                actor_items.append((agent_idx, value))
            actor_paths = [path for _, path in sorted(actor_items)]
            if not actor_paths:
                continue

            idx = min(max(step - 1, 0), prices.shape[0] - 1)
            start_idx = max(0, idx - pre_window_periods)
            steady = np.mean(prices[start_idx : idx + 1], axis=0)

            agents = [
                SACContinuous(**agent_kwargs, seed=0),
                SACContinuous(**agent_kwargs, seed=1),
            ]
            env = ContSynchronEnvironment(
                markup=0.1,
                n_periods=150_000,
                demand=LogitDemand(outside_quality=0.0, price_sensitivity=0.25),
                agents=agents,
            )
            if len(actor_paths) != len(env.agents):
                warnings.append(
                    f"Warning: run {ts} step {step} has {len(actor_paths)} actor paths."
                )
                continue
            for agent_idx, agent in enumerate(env.agents):
                if hasattr(agent, "decision"):
                    agent.decision = EpsilonGreedy(eps=0.0)
                agent.play_price([0, 0], [], 1, 0)
                agent.actor.load_weights(actor_paths[agent_idx])

            qualities = tuple(a.quality for a in env.agents)
            marginal_costs = np.array([a.marginal_cost for a in env.agents])
            eq = EquilibriumCalculator(demand=env.demand)
            current_grid_prices = np.linspace(
                env.min_price, env.max_price, grid_deviation_points
            )
            if grid_price_values is None:
                grid_price_values = current_grid_prices

            for defector_idx, _ in enumerate(env.agents):
                current_state_tf = tf.convert_to_tensor(
                    np.expand_dims(steady, axis=0), dtype=tf.float32
                )
                for _ in range(settle_periods):
                    actions_tf = []
                    for a in env.agents:
                        action_tf, _ = a._sample_action(
                            current_state_tf, deterministic=True, seed_step=None
                        )
                        actions_tf.append(action_tf)
                    current_state_tf = tf.concat(actions_tf, axis=1)

                pre_action_norm = float(
                    current_state_tf.numpy().reshape(-1)[defector_idx]
                )
                pre_deviation_price = env._denormalize_action(pre_action_norm)
                pre_price_idx = nearest_grid_index(
                    current_grid_prices, pre_deviation_price
                )
                grid_pre_price_counts[pre_price_idx] = (
                    grid_pre_price_counts.get(pre_price_idx, 0) + 1
                )

                base_actions_t0_list = []
                for a in env.agents:
                    action_tf, _ = a._sample_action(
                        current_state_tf, deterministic=True, seed_step=None
                    )
                    base_actions_t0_list.append(float(action_tf.numpy().reshape(-1)[0]))
                base_actions_t0 = tuple(base_actions_t0_list)
                base_prices_t0 = tuple(
                    env._denormalize_action(a) for a in base_actions_t0
                )
                br_price = eq.reaction_function(
                    prices=np.array(base_prices_t0),
                    qualities=np.array(qualities),
                    marginal_costs=marginal_costs,
                    i=defector_idx,
                )
                price_range = env.max_price - env.min_price
                br_action_norm = 2 * (br_price - env.min_price) / price_range - 1

                base_arr = rollout_baseline(
                    env, current_state_tf, qualities, marginal_costs
                )
                dev_arr = rollout_deviation(
                    env,
                    current_state_tf,
                    qualities,
                    marginal_costs,
                    defector_idx,
                    br_action_norm,
                    forced_length=1,
                )
                multi_period_dev_arr = rollout_deviation(
                    env,
                    current_state_tf,
                    qualities,
                    marginal_costs,
                    defector_idx,
                    br_action_norm,
                    forced_length=MULTI_PERIOD_DEVIATION_LENGTH,
                )
                one_period_gain = differential_gain(dev_arr, base_arr, defector_idx)
                multi_period_gain = differential_gain(
                    multi_period_dev_arr, base_arr, defector_idx
                )
                one_period_discounted_gain = relative_discounted_gain(
                    dev_arr,
                    base_arr,
                    defector_idx,
                    DEFAULT_DISCOUNT_FACTOR,
                )
                raw_rows.append(
                    {
                        "run_id": ts,
                        "step": step,
                        "defector_idx": defector_idx,
                        "settle_periods": settle_periods,
                        "one_period_gain": one_period_gain,
                        "multi_period_gain": multi_period_gain,
                    }
                )

                for grid_price_idx, grid_price in enumerate(current_grid_prices):
                    grid_action_norm = (
                        2 * (grid_price - env.min_price) / price_range - 1
                    )
                    grid_dev_arr = rollout_deviation(
                        env,
                        current_state_tf,
                        qualities,
                        marginal_costs,
                        defector_idx,
                        grid_action_norm,
                        forced_length=1,
                    )
                    grid_gain = relative_discounted_gain(
                        grid_dev_arr,
                        base_arr,
                        defector_idx,
                        DEFAULT_DISCOUNT_FACTOR,
                    )
                    if pre_price_idx not in grid_discounted_gains:
                        grid_discounted_gains[pre_price_idx] = {}
                    if grid_price_idx not in grid_discounted_gains[pre_price_idx]:
                        grid_discounted_gains[pre_price_idx][grid_price_idx] = []
                    grid_discounted_gains[pre_price_idx][grid_price_idx].append(
                        grid_gain
                    )

                    if grid_price_idx not in grid_br_comparison_counts:
                        grid_br_comparison_counts[grid_price_idx] = [0, 0]
                    if grid_gain > one_period_discounted_gain + 1e-12:
                        grid_br_comparison_counts[grid_price_idx][0] += 1
                    grid_br_comparison_counts[grid_price_idx][1] += 1

                if step not in step_diff_gains:
                    step_diff_gains[step] = []
                step_diff_gains[step].append(one_period_gain)
                if one_period_gain <= 0.0:
                    if step not in multi_period_step_diff_gains:
                        multi_period_step_diff_gains[step] = []
                    multi_period_step_diff_gains[step].append(multi_period_gain)

        return {
            "step_diff_gains": step_diff_gains,
            "multi_period_step_diff_gains": multi_period_step_diff_gains,
            "grid_price_values": grid_price_values,
            "grid_pre_price_counts": grid_pre_price_counts,
            "grid_discounted_gains": grid_discounted_gains,
            "grid_br_comparison_counts": grid_br_comparison_counts,
            "raw_rows": raw_rows,
            "warnings": warnings,
        }
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def parse_gpu_ids(gpus: str | None) -> list[str]:
    if gpus is None:
        return []
    return [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]


def init_parallel_worker(gpu_id: str | None) -> None:
    configure_tensorflow_runtime(gpu_id)


def iter_timestamp_results(
    timestamps: list[str],
    artifacts_dir: Path,
    checkpoints_dir: Path,
    workers: int,
    gpu_ids: list[str],
    settle_periods: int,
    pre_window_periods: int,
    grid_deviation_points: int,
):
    if not timestamps:
        return

    if workers == 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        configure_tensorflow_runtime(gpu_id)
        for ts in tqdm(timestamps, total=len(timestamps)):
            yield analyze_timestamp(
                ts,
                artifacts_dir,
                checkpoints_dir,
                settle_periods,
                pre_window_periods,
                grid_deviation_points,
            )
        return

    ctx = multiprocessing.get_context("spawn")
    executors = []
    futures = []
    try:
        for worker_idx in range(workers):
            gpu_id = gpu_ids[worker_idx % len(gpu_ids)] if gpu_ids else None
            executors.append(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=init_parallel_worker,
                    initargs=(gpu_id,),
                )
            )

        for idx, ts in enumerate(timestamps):
            executor = executors[idx % len(executors)]
            futures.append(
                executor.submit(
                    analyze_timestamp,
                    ts,
                    artifacts_dir,
                    checkpoints_dir,
                    settle_periods,
                    pre_window_periods,
                    grid_deviation_points,
                )
            )

        completed = concurrent.futures.as_completed(futures)
        for future in tqdm(completed, total=len(futures)):
            yield future.result()
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)


def merge_timestamp_result(
    result: dict[str, object],
    step_diff_gains: dict[int, list[float]],
    multi_period_step_diff_gains: dict[int, list[float]],
    grid_pre_price_counts: dict[int, int],
    grid_discounted_gains: dict[int, dict[int, list[float]]],
    grid_br_comparison_counts: dict[int, list[int]],
    raw_rows: list[dict[str, object]],
) -> np.ndarray | None:
    for step, values in result["step_diff_gains"].items():
        step_diff_gains.setdefault(step, []).extend(values)

    for step, values in result["multi_period_step_diff_gains"].items():
        multi_period_step_diff_gains.setdefault(step, []).extend(values)

    for price_idx, count in result["grid_pre_price_counts"].items():
        grid_pre_price_counts[price_idx] = grid_pre_price_counts.get(price_idx, 0) + count

    for row_idx, row_gains in result["grid_discounted_gains"].items():
        target_row = grid_discounted_gains.setdefault(row_idx, {})
        for col_idx, gains in row_gains.items():
            target_row.setdefault(col_idx, []).extend(gains)

    for price_idx, counts in result["grid_br_comparison_counts"].items():
        target_counts = grid_br_comparison_counts.setdefault(price_idx, [0, 0])
        target_counts[0] += counts[0]
        target_counts[1] += counts[1]

    raw_rows.extend(result["raw_rows"])
    return result["grid_price_values"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path.cwd() / "artifacts")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--multi-period-output", type=Path, default=None)
    parser.add_argument("--grid-deviation-output", type=Path, default=None)
    parser.add_argument(
        "--grid-deviation-points", type=int, default=DEFAULT_GRID_DEVIATION_POINTS
    )
    parser.add_argument("--settle-periods", type=int, default=IR_SETTLE_PERIODS)
    parser.add_argument("--pre-window-periods", type=int, default=IR_SETTLE_PERIODS)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of timestamp workers. Use --gpus to pin workers to GPUs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids assigned to workers round-robin, e.g. 0,1.",
    )
    args = parser.parse_args(argv)
    if args.grid_deviation_points <= 0:
        parser.error("--grid-deviation-points must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    artifacts_dir = args.artifacts_dir
    checkpoints_dir = artifacts_dir / "checkpoints"
    plots_dir = artifacts_dir / "plots"
    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (summary_dir / "deviation_tables.tex")
    multi_period_output_path = args.multi_period_output or (
        summary_dir / "multi_period_deviation_tables.tex"
    )
    grid_deviation_output_path = args.grid_deviation_output or (
        summary_dir / "grid_deviation_tables.tex"
    )

    unique_timestamps = set()
    for path in artifacts_dir.glob("*.npy"):
        timestamp_match = TIMESTAMP_RE.search(path.name)
        if timestamp_match:
            unique_timestamps.add(timestamp_match.group(1))
    timestamps = sorted(unique_timestamps)
    gpu_ids = parse_gpu_ids(args.gpus)

    step_diff_gains: dict[int, list[float]] = {}
    multi_period_step_diff_gains: dict[int, list[float]] = {}
    grid_price_values: np.ndarray | None = None
    grid_pre_price_counts: dict[int, int] = {}
    grid_discounted_gains: dict[int, dict[int, list[float]]] = {}
    grid_br_comparison_counts: dict[int, list[int]] = {}
    raw_rows = []

    for result in iter_timestamp_results(
        timestamps,
        artifacts_dir,
        checkpoints_dir,
        args.workers,
        gpu_ids,
        args.settle_periods,
        args.pre_window_periods,
        args.grid_deviation_points,
    ):
        for warning in result["warnings"]:
            print(warning)
        result_grid_prices = merge_timestamp_result(
            result,
            step_diff_gains,
            multi_period_step_diff_gains,
            grid_pre_price_counts,
            grid_discounted_gains,
            grid_br_comparison_counts,
            raw_rows,
        )
        if grid_price_values is None and result_grid_prices is not None:
            grid_price_values = result_grid_prices

    output_text = format_deviation_table(step_diff_gains)
    multi_period_output_text = format_deviation_table(multi_period_step_diff_gains)
    if grid_price_values is None:
        grid_output_text = "% No grid-deviation observations were available.\n"
    else:
        grid_output_text = format_grid_deviation_table(
            grid_price_values,
            grid_pre_price_counts,
            grid_discounted_gains,
            grid_br_comparison_counts,
        )
    output_path.write_text(output_text)
    multi_period_output_path.write_text(multi_period_output_text)
    grid_deviation_output_path.write_text(grid_output_text)
    if args.raw_output is not None:
        with args.raw_output.open("w", newline="") as raw_file:
            writer = csv.DictWriter(
                raw_file,
                fieldnames=[
                    "run_id",
                    "step",
                    "defector_idx",
                    "settle_periods",
                    "one_period_gain",
                    "multi_period_gain",
                ],
            )
            writer.writeheader()
            writer.writerows(raw_rows)
    print(output_text)
    print(multi_period_output_text)
    print(grid_output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
