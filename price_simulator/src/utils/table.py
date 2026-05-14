#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from price_simulator.src.algorithm.agents.sac import SACContinuous
from price_simulator.src.algorithm.agents.sac import build_sac_kwargs
from price_simulator.src.algorithm.demand import LogitDemand
from price_simulator.src.algorithm.environment import ContSynchronEnvironment
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from price_simulator.src.algorithm.policies import EpsilonGreedy

TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
STEP_RE = re.compile(r"_step(\d+)", re.IGNORECASE)
IR_SETTLE_PERIODS = 50
DEVIATION_HORIZON_T = 100
MULTI_PERIOD_DEVIATION_LENGTH = 5


def rollout_deviation(
    env: ContSynchronEnvironment,
    current_state_tf: tf.Tensor,
    qualities: tuple[float, ...],
    marginal_costs: np.ndarray,
    defector_idx: int,
    forced_action_norm: float,
    forced_length: int,
) -> np.ndarray:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path.cwd() / "artifacts")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--multi-period-output", type=Path, default=None)
    parser.add_argument("--settle-periods", type=int, default=IR_SETTLE_PERIODS)
    parser.add_argument("--pre-window-periods", type=int, default=IR_SETTLE_PERIODS)
    parser.add_argument("--raw-output", type=Path, default=None)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    checkpoints_dir = artifacts_dir / "checkpoints"
    plots_dir = artifacts_dir / "plots"
    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (summary_dir / "deviation_tables.tex")
    multi_period_output_path = args.multi_period_output or (
        summary_dir / "multi_period_deviation_tables.tex"
    )

    unique_timestamps = set()
    for path in artifacts_dir.glob("*.npy"):
        timestamp_match = TIMESTAMP_RE.search(path.name)
        if timestamp_match:
            unique_timestamps.add(timestamp_match.group(1))

    step_diff_gains: dict[int, list[float]] = {}
    multi_period_step_diff_gains: dict[int, list[float]] = {}
    raw_rows = []

    agent_kwargs = build_sac_kwargs()

    for ts in tqdm(unique_timestamps, total=len(unique_timestamps)):
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
            print(f"Warning: no price artifacts for run {ts}.")
            continue
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
            start_idx = max(0, idx - args.pre_window_periods)
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
                print(
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

            for defector_idx, _ in enumerate(env.agents):
                current_state_tf = tf.convert_to_tensor(
                    np.expand_dims(steady, axis=0), dtype=tf.float32
                )
                for _ in range(args.settle_periods):
                    actions_tf = []
                    for a in env.agents:
                        action_tf, _ = a._sample_action(
                            current_state_tf, deterministic=True, seed_step=None
                        )
                        actions_tf.append(action_tf)
                    current_state_tf = tf.concat(actions_tf, axis=1)

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
                raw_rows.append(
                    {
                        "run_id": ts,
                        "step": step,
                        "defector_idx": defector_idx,
                        "settle_periods": args.settle_periods,
                        "one_period_gain": one_period_gain,
                        "multi_period_gain": multi_period_gain,
                    }
                )

                if step not in step_diff_gains:
                    step_diff_gains[step] = []
                step_diff_gains[step].append(one_period_gain)
                if one_period_gain <= 0.0:
                    if step not in multi_period_step_diff_gains:
                        multi_period_step_diff_gains[step] = []
                    multi_period_step_diff_gains[step].append(multi_period_gain)

    output_text = format_deviation_table(step_diff_gains)
    multi_period_output_text = format_deviation_table(multi_period_step_diff_gains)
    output_path.write_text(output_text)
    multi_period_output_path.write_text(multi_period_output_text)
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
