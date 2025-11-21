#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
DEFAULT_DISCOUNT_FACTOR = 0.95
IR_SETTLE_PERIODS = 50
END_PLOT_T = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path.cwd() / "artifacts")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    checkpoints_dir = artifacts_dir / "checkpoints"
    plots_dir = artifacts_dir / "plots"
    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (summary_dir / "deviation_tables.tex")

    unique_timestamps = list(
        {
            TIMESTAMP_RE.search(path.name).group(1)
            for path in artifacts_dir.glob("*.npy")
            if TIMESTAMP_RE.search(path.name)
        }
    )

    step_disc_gains: dict[int, list[float]] = {}

    rng = np.random.default_rng(0)

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
            step = int(checkpoint["step"])
            actor_paths = []
            if "actor_paths" in checkpoint:
                actor_paths = list(checkpoint["actor_paths"])
            else:
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
            start_idx = max(0, idx - IR_SETTLE_PERIODS)
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
                agent.play_price([0, 0], 1, 0, use_target=False)
                agent.actor.load_weights(actor_paths[agent_idx])

            qualities = tuple(a.quality for a in env.agents)
            marginal_costs = np.array([a.marginal_cost for a in env.agents])
            eq = EquilibriumCalculator(demand=env.demand)

            for defector_idx, _ in enumerate(env.agents):
                current_state_tf = tf.convert_to_tensor(
                    np.expand_dims(steady, axis=0), dtype=tf.float32
                )
                for _ in range(IR_SETTLE_PERIODS):
                    actions_tf = []
                    for a in env.agents:
                        action_tf, _ = a._sample_action(
                            current_state_tf, seed_step=0
                        )
                        actions_tf.append(action_tf)
                    current_state_tf = tf.concat(actions_tf, axis=1)

                base_actions_t0_list = []
                for a in env.agents:
                    action_tf, _ = a._sample_action(
                        current_state_tf, seed_step=0
                    )
                    base_actions_t0_list.append(
                        float(action_tf.numpy().reshape(-1)[0])
                    )
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

                dev_profits = []
                base_profits = []
                state_dev_tf = current_state_tf
                state_base_tf = current_state_tf
                for t in range(END_PLOT_T):
                    dev_actions = []
                    dev_actions_tf = []
                    for i, a in enumerate(env.agents):
                        if t == 0 and i == defector_idx:
                            dev_actions.append(br_action_norm)
                            dev_actions_tf.append(
                                tf.constant([[br_action_norm]], dtype=tf.float32)
                            )
                        else:
                            action_tf, _ = a._sample_action(
                                state_dev_tf, seed_step=0
                            )
                            dev_actions.append(
                                float(action_tf.numpy().reshape(-1)[0])
                            )
                            dev_actions_tf.append(action_tf)
                    dev_real_prices = tuple(
                        env._denormalize_action(a) for a in dev_actions
                    )
                    dev_qs = env.demand.get_quantities(dev_real_prices, qualities)
                    dev_rews = tuple(
                        np.multiply(
                            np.subtract(dev_real_prices, marginal_costs), dev_qs
                        )
                    )
                    dev_profits.append(dev_rews)
                    state_dev_tf = tf.concat(dev_actions_tf, axis=1)

                    base_actions = []
                    base_actions_tf = []
                    for a in env.agents:
                        action_tf, _ = a._sample_action(
                            state_base_tf, seed_step=0
                        )
                        base_actions.append(
                            float(action_tf.numpy().reshape(-1)[0])
                        )
                        base_actions_tf.append(action_tf)
                    base_real_prices = tuple(
                        env._denormalize_action(a) for a in base_actions
                    )
                    base_qs = env.demand.get_quantities(base_real_prices, qualities)
                    base_rews = tuple(
                        np.multiply(
                            np.subtract(base_real_prices, marginal_costs), base_qs
                        )
                    )
                    base_profits.append(base_rews)
                    state_base_tf = tf.concat(base_actions_tf, axis=1)

                dev_arr = np.asarray(dev_profits, dtype=np.float32)
                base_arr = np.asarray(base_profits, dtype=np.float32)
                diff_col = dev_arr[:, defector_idx] - base_arr[:, defector_idx]
                weights = np.power(
                    DEFAULT_DISCOUNT_FACTOR,
                    np.arange(diff_col.shape[0], dtype=np.float32),
                )
                discounted_gain = float(np.sum(diff_col * weights))

                if step not in step_disc_gains:
                    step_disc_gains[step] = []
                step_disc_gains[step].append(discounted_gain)

    lines = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append(
        "Step & 25th percentile (\\%) & Median (\\%) "
        "& 75th percentile (\\%) & Mean (\\%) "
        "& Unprofitable \\%  [95\\% CI] \\\\"
    )
    lines.append("\\hline")

    for step in sorted(step_disc_gains):
        disc_vals = np.asarray(step_disc_gains[step], dtype=np.float32)
        if disc_vals.size == 0:
            continue
        p25_disc, p50_disc, p75_disc = np.percentile(disc_vals, [25, 50, 75])
        mean_disc = float(np.mean(disc_vals))
        unprofit_mask = disc_vals <= 0.0
        pct_disc = 100.0 * float(np.mean(unprofit_mask))
        boot_indices = rng.integers(
            0, unprofit_mask.size, size=(5000, unprofit_mask.size)
        )
        boot_props = unprofit_mask[boot_indices].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_props, [2.5, 97.5])
        step_label = f"\\num{{{step}}}"

        p25_disc_pct = 100.0 * p25_disc
        p50_disc_pct = 100.0 * p50_disc
        p75_disc_pct = 100.0 * p75_disc
        mean_disc_pct = 100.0 * mean_disc

        p25_cell = f"{p25_disc_pct:.2g}\\%"
        p50_cell = f"{p50_disc_pct:.2g}\\%"
        p75_cell = f"{p75_disc_pct:.2g}\\%"
        pct_cell = f"{pct_disc:.2g}\\%"
        ci_cell = f"{100.0 * ci_low:.2g}\\%--{100.0 * ci_high:.2g}\\%"
        pct_ci_cell = f"{pct_cell} [{ci_cell}]"

        lines.append(
            f"{step_label} & {p25_cell} & {p50_cell} & {p75_cell} "
            f"& {mean_disc_pct:.2g}\\% & {pct_ci_cell} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    output_text = "\n".join(lines) + "\n"
    output_path.write_text(output_text)
    print(output_text)
