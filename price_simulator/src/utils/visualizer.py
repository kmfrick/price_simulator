#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, StrMethodFormatter

from price_simulator.src.algorithm.agents.sac import SACContinuous
from price_simulator.src.algorithm.demand import LogitDemand
from price_simulator.src.algorithm.environment import ContSynchronEnvironment
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from price_simulator.src.algorithm.policies import EpsilonGreedy
from price_simulator.src.algorithm.agents.sac import build_sac_kwargs
from price_simulator.src.utils.serializer import SimulationRun

TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")
STEP_RE = re.compile(r"_step(\d+)", re.IGNORECASE)
DEFAULT_DISCOUNT_FACTOR = 0.95
START_PLOT_T = -1
END_PLOT_T = 10
MA_WINDOW = 5000
GRID_POINTS = 50
IR_SETTLE_PERIODS = 50
EVAL_CHUNK_SIZE = 2048


class PlotSuite:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.artifacts_dir = args.artifacts_dir
        self.n_agents = args.n_agents
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.plots_dir = self.artifacts_dir / "plots"
        self.summary_dir = self.plots_dir / "summary"
        self.sessions_dir = self.plots_dir / "sessions"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _select_preferred(self, paths: list[Path]) -> Path:
        finals = [p for p in paths if "_final" in p.stem]
        if finals:
            return sorted(finals)[-1]
        steps = []
        for path in paths:
            m = STEP_RE.search(path.stem)
            if m:
                steps.append((int(m.group(1)), path))
        if steps:
            return sorted(steps, key=lambda item: item[0])[-1][1]
        return sorted(paths)[-1]

    def _build_env(self) -> ContSynchronEnvironment:
        agent_kwargs = build_sac_kwargs()
        agent_kwargs["state_dim"] = self.n_agents
        agents = [
            SACContinuous(**agent_kwargs, seed=idx)
            for idx in range(self.n_agents)
        ]
        env = ContSynchronEnvironment(
            markup=0.1,
            n_periods=150_000,
            demand=LogitDemand(outside_quality=0.0, price_sensitivity=0.25),
            agents=agents,
        )
        return env

    def _checkpoint_paths(self, checkpoint: dict) -> tuple[list[Path], list[Path], list[Path]]:
        if "actor_paths" in checkpoint:
            return (
                checkpoint["actor_paths"],
                checkpoint["critic1_paths"],
                checkpoint["critic2_paths"],
            )

        actor_items = []
        critic1_items = []
        critic2_items = []
        for key, value in checkpoint.items():
            if not isinstance(value, Path):
                continue
            match = re.search(r"agent(\d+)_", key)
            agent_idx = int(match.group(1)) if match else 999
            if key.endswith("_actor"):
                actor_items.append((agent_idx, key, value))
            elif key.endswith("_critic1"):
                critic1_items.append((agent_idx, key, value))
            elif key.endswith("_critic2"):
                critic2_items.append((agent_idx, key, value))

        actor_paths = [path for _, _, path in sorted(actor_items)]
        critic1_paths = [path for _, _, path in sorted(critic1_items)]
        critic2_paths = [path for _, _, path in sorted(critic2_items)]

        if not actor_paths:
            raise KeyError("actor_paths")

        return (actor_paths, critic1_paths, critic2_paths)

    def _init_env_with_weights(
        self, actor_paths: list[Path]
    ) -> ContSynchronEnvironment:
        env = self._build_env()
        zero_state = [0.0] * self.n_agents
        for agent_idx, agent in enumerate(env.agents):
            if hasattr(agent, "decision"):
                agent.decision = EpsilonGreedy(eps=0.0)
            agent.play_price(zero_state, 1, 0, use_target=False)
            agent.actor.load_weights(actor_paths[agent_idx])
        return env

    def _mean_agent_metric(self, paths: list[Path]) -> np.ndarray:
        arrays = [np.asarray(np.load(path)) for path in paths]
        min_len = min(arr.shape[0] for arr in arrays)
        arrays = [arr[:min_len] for arr in arrays]
        return np.mean(np.stack(arrays, axis=0), axis=0)

    def _plot_phase_diagram(
        self,
        mapping: np.ndarray,
        price_grid: np.ndarray,
        output_path: Path,
        phase_context: dict[str, object],
        fixed_point: tuple[float, float] | None = None,
    ) -> None:
        p1_grid, p2_grid = np.meshgrid(price_grid, price_grid, indexing="xy")
        u = mapping[:, :, 0] - p1_grid
        v = mapping[:, :, 1] - p2_grid
        speed = np.sqrt(u * u + v * v)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        stream = ax.streamplot(
            price_grid,
            price_grid,
            u,
            v,
            color=speed,
            cmap="Reds",
            density=0.5,
            broken_streamlines=False,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = fig.colorbar(stream.lines, cax=cax)
        cbar.ax.set_ylabel("Step Magnitude", rotation=270, labelpad=10)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect("equal", adjustable="box")
        tick_positions = phase_context["tick_positions"]
        tick_labels = phase_context["tick_labels"]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        nash_norm = phase_context["nash_norm"]
        monopoly_norm = phase_context["monopoly_norm"]
        for val in np.atleast_1d(nash_norm):
            ax.axhline(val, color="k", linestyle=":", alpha=0.4)
            ax.axvline(val, color="k", linestyle=":", alpha=0.4)
        for val in np.atleast_1d(monopoly_norm):
            ax.axhline(val, color="k", linestyle="-.", alpha=0.4)
            ax.axvline(val, color="k", linestyle="-.", alpha=0.4)

        ax.set_xlabel("Agent 1 Price")
        ax.set_ylabel("Agent 2 Price")
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    def run(self) -> int:
        # Compute environment parameters
        demand = LogitDemand(outside_quality=0.0, price_sensitivity=0.25)
        calc = EquilibriumCalculator(demand=demand)
        qualities = [2.0] * self.n_agents
        costs = [1.0] * self.n_agents
        monopoly_prices = calc.get_monopoly_outcome(qualities, costs)
        nash_prices = calc.get_nash_equilibrium(qualities, costs)
        self.phase_context = None
        if self.n_agents == 2:
            monopoly_price = float(np.min(monopoly_prices))
            nash_price = float(np.min(nash_prices))
            increase = (monopoly_price - nash_price) * 0.1
            price_min = nash_price - increase
            price_max = monopoly_price + increase
            price_range = price_max - price_min

            ref_env = self._build_env()
            # We need this for the phase diagram bc everything is in [-1, 1]
            # but we want to denormalize it
            phase_nash_norm = (
                2 * (np.array(nash_prices) - price_min) / price_range - 1
            )
            phase_monopoly_norm = (
                2 * (np.array(monopoly_prices) - price_min) / price_range - 1
            )
            tick_positions = [-1, -0.5, 0, 0.5, 1]
            tick_labels = [
                f"{ref_env._denormalize_action(t):.2f}" for t in tick_positions
            ]
            self.phase_context = {
                "tick_positions": tick_positions,
                "tick_labels": tick_labels,
                "nash_norm": phase_nash_norm,
                "monopoly_norm": phase_monopoly_norm,
                "min_price": price_min,
                "max_price": price_max,
            }
        else:
            print("Skipping phase diagram: requires exactly 2 agents.")
        # Read each run
        unique_timestamps = list({
            TIMESTAMP_RE.search(path.name).group(1)
            for path in Path(self.artifacts_dir).glob("*.npy")
            if TIMESTAMP_RE.search(path.name)
        })
        runs: list[SimulationRun] = []
        for ts in sorted(unique_timestamps):
            print(f"Loading run: {ts}...")

            # List of fields in SimulationRun that need to be loaded from .npy files
            metric_fields = [
                "prices",
                "profits",
                "grad_norm",
                "average_reward",
                "q_baseline",
                "policy_loss",
                "policy_entropy",
                "q_loss",
                "temperature",
                "policy_kl",
            ]

            loaded_metrics = {}

            for field in metric_fields:
                # Glob for all agents for this specific metric and timestamp
                # e.g., artifacts/*_profits_20251226-122739.npy
                # sorted() ensures agent1 always comes before agent2
                files = sorted(self.artifacts_dir.glob(f"*_{field}_{ts}.npy"))

                # Load all files found (agent1, agent2, etc)
                arrays = []
                for f in files:
                    arr = np.asarray(np.load(f))
                    if arr.ndim > 1:
                        arr = arr.reshape(arr.shape[0], -1)
                        if arr.shape[1] == 1:
                            arr = arr[:, 0]
                        else:
                            arr = arr.mean(axis=1)
                    arrays.append(arr)
                if not arrays:
                    raise ValueError(
                        f"No artifacts found for metric '{field}' in run {ts}."
                    )
                lengths = [arr.shape[0] for arr in arrays]
                min_len = min(lengths)
                if any(length != min_len for length in lengths):
                    print(
                        f"Warning: metric '{field}' in run {ts} has "
                        "uneven lengths; truncating to shortest series."
                    )
                    arrays = [arr[:min_len] for arr in arrays]
                # Stack them into a single array (e.g. shape [Time, Num_Agents])
                loaded_metrics[field] = np.stack(arrays).T

            # Finds: agent1_..._final_actor.weights.h5, agent2_..._final_actor.weights.h5
            final_actor = sorted(self.checkpoints_dir.glob(f"*_{ts}_final_actor.weights.h5"))
            final_critic1 = sorted(self.checkpoints_dir.glob(f"*_{ts}_final_critic1.weights.h5"))
            final_critic2 = sorted(self.checkpoints_dir.glob(f"*_{ts}_final_critic2.weights.h5"))

            step_map = {}

            # Scan all files containing this timestamp and "_step"
            for cp_file in self.checkpoints_dir.glob(f"*_{ts}_step*.weights.h5"):
                step_match = STEP_RE.search(cp_file.name)
                step = int(step_match.group(1))

                if step not in step_map:
                    step_map[step] = {"step": step}

                # Construct a key for the dict, e.g. "agent1_actor"
                # Input: agent1_2025..._step90000_actor.weights.h5
                parts = cp_file.name.split('_')
                # agent_id is usually parts[0] (agent1)
                # component is usually parts[-2] (actor/critic1) before .weights.h5
                # A safer generic way: get everything before timestamp + component type
                agent_id = parts[0]
                component_type = cp_file.name.split('_')[-1].replace('.weights.h5', '')

                key_name = f"{agent_id}_{component_type}"
                step_map[step][key_name] = cp_file

            # Sort by step number to ensure list order is chronological
            sorted_checkpoints = [step_map[k] for k in sorted(step_map.keys())]

            run = SimulationRun(
                run_id=ts,
                final_actor_paths=final_actor,
                final_critic1_paths=final_critic1,
                final_critic2_paths=final_critic2,
                checkpoints=sorted_checkpoints,
                **loaded_metrics # Unpack the dictionary of loaded numpy arrays
            )
            runs.append(run)
        # Plot profit gain time series
        key = "profits"
        yname = "Profit Gain"
        all_frames = []
        for run in tqdm(runs, desc=yname):
            data_raw = getattr(run, key, None)
            data = np.asarray(data_raw, dtype=np.float32)
            data = ((data + 1.0) / 2) # Denormalize [-1, 1] -> [0, 1]
            data = data.mean(axis=1) # Mean across agents
            s = pd.Series(data)
            ma = s.rolling(MA_WINDOW, min_periods=1).mean()

            run_df = pd.DataFrame({"Time": ma.index, yname: ma.values})
            run_dir = self.sessions_dir / run.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run_df = run_df.dropna(subset=[yname])
            plt.figure()
            ax = sns.lineplot(
                data=run_df, x="Time", y=yname, estimator=None
            )
            plt.savefig(run_dir / f"{key}.png")
            plt.close()

            all_frames.append(run_df)

        summary_df = pd.concat(all_frames, ignore_index=True)
        summary_df = summary_df.dropna(subset=[yname])
        plt.figure()
        ax = sns.lineplot(
            data=summary_df,
            x="Time",
            y=yname,
            estimator="mean",
            errorbar=("pi", 90),
            sort=False,
        )
        plt.savefig(self.summary_dir / f"{key}.png")
        plt.close()
        # Plot profit gain distribution
        gains: list[float] = []
        for run in runs:
            profits_arr = np.asarray(run.profits, dtype=np.float32)
            profits_arr = (profits_arr + 1.0) / 2 # Denormalize [-1, 1] -> [0, 1]
            idx_prof = max(0, len(profits_arr) - 1)
            start_idx_prof = max(0, idx_prof - MA_WINDOW)
            relevant = profits_arr[start_idx_prof : idx_prof + 1]
            gains.append(float(np.mean(relevant)))

        plt.figure()
        gains_pct = np.asarray(gains, dtype=np.float32) * 100.0
        gains_pct = np.round(gains_pct, 1)
        ax = sns.histplot(gains_pct, bins=15, stat="percent")
        ax.xaxis.set_major_formatter(StrMethodFormatter(r"{x:.1f}\%"))
        ax.yaxis.set_major_formatter(StrMethodFormatter(r"{x:.0f}\%"))
        min_gain = float(np.min(gains_pct))
        max_gain = float(np.max(gains_pct))
        plt.xlabel(r"Profit Gain (\%)")
        plt.ylabel("Percent")
        plt.savefig(self.summary_dir / "profit_gains_hist.png")
        plt.close()
        # Plot phase diagram for each run
        runs_with_final = [run for run in runs if run.final_actor_paths]
        if runs_with_final and self.n_agents == 2:
            price_grid = np.linspace(-1, 1, GRID_POINTS)
            p1_grid, p2_grid = np.meshgrid(price_grid, price_grid, indexing="xy")
            total = price_grid.size * price_grid.size
            states = np.stack([p1_grid, p2_grid], axis=-1).reshape(-1, 2, order="F")
            for run in tqdm(runs_with_final, desc="Policy maps"):
                env = self._init_env_with_weights(run.final_actor_paths)
                mapping_flat = np.zeros((total, 2), dtype=np.float32)
                chunk_size = max(1, int(EVAL_CHUNK_SIZE))
                for start in range(0, total, chunk_size):
                    end = min(total, start + chunk_size)
                    s0 = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
                    s1 = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
                    actions0, _ = env.agents[0]._sample_action_batch(
                        s0, deterministic=True
                    )
                    actions1, _ = env.agents[1]._sample_action_batch(
                        s1, deterministic=True
                    )
                    mapping_flat[start:end, 0] = actions0.numpy().reshape(-1)
                    mapping_flat[start:end, 1] = actions1.numpy().reshape(-1)

                n = price_grid.size
                mapping = np.zeros((n, n, 2), dtype=np.float32)
                mapping[:, :, 0] = mapping_flat[:, 0].reshape(n, n, order="F")
                mapping[:, :, 1] = mapping_flat[:, 1].reshape(n, n, order="F")

                deltas = mapping - np.stack([p1_grid, p2_grid], axis=-1)
                dist_sq = (
                    deltas[:, :, 0] * deltas[:, :, 0]
                    + deltas[:, :, 1] * deltas[:, :, 1]
                )
                idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                fixed_point = (float(p1_grid[idx]), float(p2_grid[idx]))

                run_dir = self.sessions_dir / run.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                self._plot_phase_diagram(
                    mapping,
                    price_grid,
                    run_dir / "policy_phase.png",
                    self.phase_context,
                    fixed_point=fixed_point,
                )

        if runs_with_final:
            all_results = []
            for run in tqdm(runs_with_final, desc="Impulse response"):
                prices = np.asarray(run.prices, dtype=np.float32)
                if prices.ndim == 1:
                    prices = prices.reshape(-1, 1)
                idx = max(0, len(prices) - 1)
                start_idx = max(0, idx - IR_SETTLE_PERIODS)
                steady = np.mean(prices[start_idx : idx + 1], axis=0)
                env = self._init_env_with_weights(run.final_actor_paths)
                qualities = tuple(a.quality for a in env.agents)
                marginal_costs = np.array([a.marginal_cost for a in env.agents])
                eq = EquilibriumCalculator(demand=env.demand)
                for defector_idx, _ in enumerate(env.agents):
                    current_state_tf = tf.convert_to_tensor(
                        np.expand_dims(steady, axis=0),
                        dtype=tf.float32,
                    )
                    pre_prices = []
                    # Settling cycle
                    for _ in range(IR_SETTLE_PERIODS):
                        actions = []
                        actions_tf = []
                        for a in env.agents:
                            action_tf, _ = a._sample_action(
                                current_state_tf,
                                deterministic=True,
                                seed_step=None
                            )
                            actions.append(float(action_tf.numpy().reshape(-1)[0]))
                            actions_tf.append(action_tf)
                        real_prices = tuple(
                            env._denormalize_action(a) for a in actions
                        )
                        pre_prices.append(real_prices)
                        current_state_tf = tf.concat(actions_tf, axis=1)
                    # Non-deviation profit
                    base_profits = []
                    base_prices_list = []
                    state_base_tf = current_state_tf
                    for _ in range(END_PLOT_T):
                        base_actions = []
                        base_actions_tf = []
                        for a in env.agents:
                            action_tf, _ = a._sample_action(
                                state_base_tf,
                                deterministic=True,
                                seed_step=None
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
                        base_prices_list.append(base_real_prices)
                        state_base_tf = tf.concat(base_actions_tf, axis=1)

                    base_arr = np.asarray(base_profits, dtype=np.float32)
                    price_range = env.max_price - env.min_price
                    pre_prices_arr = np.asarray(pre_prices, dtype=np.float32)
                    br_price_t0 = eq.reaction_function(
                        prices=np.array(pre_prices_arr[-1]),
                        qualities=np.array(qualities),
                        marginal_costs=marginal_costs,
                        i=defector_idx,
                    )
                    br_action_norm_t0 = (
                        2 * (br_price_t0 - env.min_price) / price_range - 1
                    )
                    defect_t = 0
                    br_action_norm = br_action_norm_t0
                    dev_profits = []
                    dev_prices_list = []
                    state_dev_tf = current_state_tf
                    for t in range(END_PLOT_T):
                        dev_actions = []
                        dev_actions_tf = []
                        for i, a in enumerate(env.agents):
                            if t == defect_t and i == defector_idx:
                                dev_actions.append(br_action_norm)
                                dev_actions_tf.append(
                                    tf.constant([[br_action_norm]], dtype=tf.float32)
                                )
                            else:
                                action_tf, _ = a._sample_action(
                                    state_dev_tf,
                                    deterministic=True,
                                    seed_step=None
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
                                np.subtract(dev_real_prices, marginal_costs),
                                dev_qs,
                            )
                        )
                        dev_profits.append(dev_rews)
                        dev_prices_list.append(dev_real_prices)
                        state_dev_tf = tf.concat(dev_actions_tf, axis=1)

                    dev_arr = np.asarray(dev_profits, dtype=np.float32)
                    diff_col = dev_arr[:, defector_idx] - base_arr[:, defector_idx]
                    weights = np.power(
                        DEFAULT_DISCOUNT_FACTOR,
                        np.arange(diff_col.shape[0], dtype=np.float32),
                    )
                    discounted_gain = float(np.sum(diff_col * weights))
                    disc_base = float(np.sum(base_arr[:, defector_idx] * weights))
                    rel_discounted_gain = float(discounted_gain / disc_base)
                    all_results.append(
                        {
                            "run_id": run.run_id,
                            "defector_idx": defector_idx,
                            "defect_t": defect_t,
                            "pre_prices": np.asarray(pre_prices_arr, dtype=np.float32),
                            "dev_profits": dev_arr,
                            "base_profits": base_arr,
                            "dev_prices": np.asarray(dev_prices_list, dtype=np.float32),
                            "discounted_profit_gain": discounted_gain,
                            "rel_discounted_profit_gain": rel_discounted_gain,
                        }
                    )

            rel_disc_profits = (
                np.asarray(
                    [r["rel_discounted_profit_gain"] for r in all_results],
                    dtype=np.float32,
                )
                * 100.0
            )
            rel_disc_profits = np.round(rel_disc_profits, 1)
            if rel_disc_profits.size:
                plt.figure()
                ax = sns.histplot(rel_disc_profits, bins=15, stat="percent")
                ax.xaxis.set_major_formatter(StrMethodFormatter(r"{x:.1f}\%"))
                ax.yaxis.set_major_formatter(StrMethodFormatter(r"{x:.0f}\%"))
                plt.xlabel(
                    rf"Relative discounted gain $\frac{{\sum_t \gamma^t(\pi_t^{{\mathrm{{dev}}}}-\pi_t^{{\mathrm{{base}}}})}}{{\sum_t \gamma^t\pi_t^{{\mathrm{{base}}}}}}$ (\%), $\gamma={DEFAULT_DISCOUNT_FACTOR}$"
                )
                plt.ylabel("Percent")
                plt.savefig(self.summary_dir / "deviation_profit_rel_disc_hist.png")
                plt.close()
            # IR for Nash equilibria
            results_for_plots = [
                r for r in all_results if r["discounted_profit_gain"] < 0.0
            ]

            collected_def_dev = []
            collected_nondef_dev = []
            collected_def_pre = []
            collected_nondef_pre = []
            collected_def_dev_prices = []
            collected_nondef_dev_prices = []
            collected_def_pre_prices = []
            collected_nondef_pre_prices = []
            common_pre_len = None
            common_dev_len = None

            for r in results_for_plots:
                def_idx = r["defector_idx"]
                dev_prices_abs = np.asarray(r["dev_prices"])
                pre_prices_abs = np.asarray(r["pre_prices"])
                def_dev_abs = dev_prices_abs[:, def_idx]
                def_pre_abs = pre_prices_abs[:, def_idx]
                n_agents = dev_prices_abs.shape[1]
                others_mask = np.arange(n_agents) != def_idx
                if np.any(others_mask):
                    nondef_dev_abs = np.mean(dev_prices_abs[:, others_mask], axis=1)
                    nondef_pre_abs = np.mean(pre_prices_abs[:, others_mask], axis=1)
                else:
                    nondef_dev_abs = np.zeros_like(def_dev_abs)
                    nondef_pre_abs = np.zeros_like(def_pre_abs)

                baseline_def_price = def_pre_abs[-1]
                baseline_nondef_price = nondef_pre_abs[-1]
                def_dev_deviation = def_dev_abs - baseline_def_price
                def_pre_deviation = def_pre_abs - baseline_def_price
                nondef_dev_deviation = nondef_dev_abs - baseline_nondef_price
                nondef_pre_deviation = nondef_pre_abs - baseline_nondef_price

                collected_def_dev.append(def_dev_deviation)
                collected_nondef_dev.append(nondef_dev_deviation)
                collected_def_pre.append(def_pre_deviation)
                collected_nondef_pre.append(nondef_pre_deviation)
                collected_def_dev_prices.append(def_dev_abs)
                collected_nondef_dev_prices.append(nondef_dev_abs)
                collected_def_pre_prices.append(def_pre_abs)
                collected_nondef_pre_prices.append(nondef_pre_abs)
                if common_pre_len is None or def_pre_abs.shape[0] < common_pre_len:
                    common_pre_len = def_pre_abs.shape[0]
                if common_dev_len is None or def_dev_abs.shape[0] < common_dev_len:
                    common_dev_len = def_dev_abs.shape[0]

                xs = np.arange(-def_pre_abs.shape[0], def_dev_abs.shape[0])
                full_def_dev = np.concatenate([def_pre_deviation, def_dev_deviation])
                full_nondef_dev = np.concatenate(
                    [nondef_pre_deviation, nondef_dev_deviation]
                )
                out_path = self.sessions_dir / str(r["run_id"])
                out_path.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots()
                ax.plot(
                    xs,
                    full_def_dev,
                    label="Defector Deviation",
                    marker="o",
                    linestyle="--",
                )
                ax.plot(
                    xs,
                    full_nondef_dev,
                    label="Non-Defector Deviation",
                    linestyle="--",
                    marker="o",
                )
                ax.axvline(0, alpha=0.4)
                ax.set_xlim(START_PLOT_T, END_PLOT_T)
                ax.set_xlabel("Time")
                ax.set_ylabel("Deviation from Steady-State Price")
                ax.legend()
                fig.tight_layout()
                fig.savefig(
                    out_path / f"ir_{r['run_id']}_def{def_idx}.png"
                )
                plt.close(fig)

            xs_agg = np.arange(-common_pre_len, common_dev_len)
            idx_zero = common_pre_len
            def_pre_arr = np.array(
                [arr[:common_pre_len] for arr in collected_def_pre_prices]
            )
            nondef_pre_arr = np.array(
                [arr[:common_pre_len] for arr in collected_nondef_pre_prices]
            )
            def_dev_arr = np.array(
                [arr[:common_dev_len] for arr in collected_def_dev_prices]
            )
            nondef_dev_arr = np.array(
                [arr[:common_dev_len] for arr in collected_nondef_dev_prices]
            )

            fig, ax = plt.subplots()
            stat_def_pre = np.mean(def_pre_arr, axis=0)
            stat_nondef_pre = np.mean(nondef_pre_arr, axis=0)
            stat_def_dev = np.mean(def_dev_arr, axis=0)
            stat_nondef_dev = np.mean(nondef_dev_arr, axis=0)
            full_stat_def = np.concatenate([stat_def_pre, stat_def_dev])
            full_stat_nondef = np.concatenate([stat_nondef_pre, stat_nondef_dev])
            ax.plot(
                xs_agg,
                full_stat_def,
                label="Mean Defector Price",
                linestyle="--",
                marker="o",
            )
            ax.plot(
                xs_agg,
                full_stat_nondef,
                label="Mean Non-Defector Price",
                linestyle="--",
                marker="o",
            )
            ax.axvline(0, alpha=0.4)
            ax.set_xlim(START_PLOT_T, END_PLOT_T)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.summary_dir / "ir_aggregate_mean.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            def_pre_prices = np.array(
                [arr[:common_pre_len] for arr in collected_def_pre_prices]
            )
            def_dev_prices = np.array(
                [arr[:common_dev_len] for arr in collected_def_dev_prices]
            )
            nondef_pre_prices = np.array(
                [arr[:common_pre_len] for arr in collected_nondef_pre_prices]
            )
            nondef_dev_prices = np.array(
                [arr[:common_dev_len] for arr in collected_nondef_dev_prices]
            )
            def_baseline = def_pre_prices[:, -1]
            nondef_baseline = nondef_pre_prices[:, -1]
            def_denom = np.where(np.abs(def_baseline) < 1e-9, np.nan, def_baseline)
            nondef_denom = np.where(
                np.abs(nondef_baseline) < 1e-9, np.nan, nondef_baseline
            )
            def_pre_pct = (def_pre_prices - def_baseline[:, None]) / def_denom[:, None]
            def_dev_pct = (def_dev_prices - def_baseline[:, None]) / def_denom[:, None]
            nondef_pre_pct = (
                nondef_pre_prices - nondef_baseline[:, None]
            ) / nondef_denom[:, None]
            nondef_dev_pct = (
                nondef_dev_prices - nondef_baseline[:, None]
            ) / nondef_denom[:, None]
            def_pre_pct = np.nan_to_num(def_pre_pct * 100.0)
            def_dev_pct = np.nan_to_num(def_dev_pct * 100.0)
            nondef_pre_pct = np.nan_to_num(nondef_pre_pct * 100.0)
            nondef_dev_pct = np.nan_to_num(nondef_dev_pct * 100.0)
            full_def = np.hstack([def_pre_pct, def_dev_pct])
            full_nondef = np.hstack([nondef_pre_pct, nondef_dev_pct])
            full_def = np.round(full_def, 1)
            full_nondef = np.round(full_nondef, 1)
            slice_start = max(0, idx_zero + START_PLOT_T)
            slice_end = min(len(xs_agg), idx_zero + END_PLOT_T + 1)
            full_def_plot = full_def[:, slice_start:slice_end]
            full_nondef_plot = full_nondef[:, slice_start:slice_end]
            positions_plot = xs_agg[slice_start:slice_end]
            width = 0.3
            ax.boxplot(
                full_def_plot,
                positions=positions_plot - width / 2,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.6),
                showfliers=False,
            )
            ax.boxplot(
                full_nondef_plot,
                positions=positions_plot + width / 2,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor="orange", alpha=0.6),
                showfliers=False,
            )

            custom_lines = [
                Line2D([0], [0], color="lightblue", lw=4, alpha=0.6),
                Line2D([0], [0], color="orange", lw=4, alpha=0.6),
            ]
            ax.legend(custom_lines, ["Defector", "Non-Defector"])
            ax.axvline(0, alpha=0.4)
            ax.set_xlim(START_PLOT_T, END_PLOT_T)
            ax.set_xlabel("Time")
            ax.set_ylabel(r"Deviation from Steady-State Price (\%)")
            ax.yaxis.set_major_formatter(StrMethodFormatter(r"{x:.1f}\%"))
            tick_start = int(np.floor(positions_plot.min()))
            tick_end = int(np.ceil(positions_plot.max()))
            ax.xaxis.set_major_locator(
                FixedLocator(np.arange(tick_start, tick_end + 1, 1))
            )
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            fig.tight_layout()
            fig.savefig(self.summary_dir / "ir_aggregate_boxplot.png")
            plt.close(fig)
        return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path.cwd() / "artifacts")
    parser.add_argument("--n-agents", type=int, default=2)
    args = parser.parse_args(argv)

    return PlotSuite(args).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
