import copy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from price_simulator.src.algorithm.demand import LogitDemand
from price_simulator.src.algorithm.environment import ContSynchronEnvironment
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator
from price_simulator.src.algorithm.agents.sac import SACContinuous
from price_simulator.src.algorithm.agents.sac import build_sac_kwargs
import tensorflow as tf



def checkpoint_networks(
    environment: ContSynchronEnvironment,
    checkpoint_dir: Path,
    suffix: str = "",
    timestamp: str = "",
) -> List[Path]:
    if timestamp is None:
        raise ValueError("checkpoint_networks requires an explicit timestamp.")
    ts = timestamp
    checkpoint_paths: List[Path] = []
    for idx, agent in enumerate(environment.agents, start=1):
        suffix_str = f"_{suffix}" if suffix else ""
        actor = getattr(agent, "actor", None)
        critic1 = getattr(agent, "critic1", None)
        critic2 = getattr(agent, "critic2", None)
        actor_path = (checkpoint_dir / f"agent{idx}_{ts}{suffix_str}_actor.weights.h5")
        critic1_path = checkpoint_dir / f"agent{idx}_{ts}{suffix_str}_critic1.weights.h5"
        critic2_path = checkpoint_dir / f"agent{idx}_{ts}{suffix_str}_critic2.weights.h5"
        actor.save_weights(actor_path)
        critic1.save_weights(critic1_path)
        critic2.save_weights(critic2_path)
        checkpoint_paths.append(actor_path)
        checkpoint_paths.append(critic1_path)
        checkpoint_paths.append(critic2_path)

    return checkpoint_paths


def run(total_periods: int = 50_000):
    dt_now = datetime.now()
    timestamp = dt_now.strftime("%Y%m%d-%H%M%S")
    base_seed = int(dt_now.timestamp())
    tf.random.set_seed(base_seed)


    if total_periods <= 0:
        raise ValueError("total_periods must be a positive integer.")
    agent_seeds = [base_seed + agent_id for agent_id in range(2)]
    if len(set(agent_seeds)) != len(agent_seeds):
        raise ValueError("Agent seeds must be unique for each SAC agent.")

    sac_kwargs = build_sac_kwargs()
    env = ContSynchronEnvironment(
        markup=0.1,
        n_periods=total_periods,
        possible_prices=[],
        demand=LogitDemand(outside_quality=0.0, price_sensitivity=0.25),
        agents=[
            SACContinuous(seed=agent_seeds[0], **sac_kwargs),
            SACContinuous(seed=agent_seeds[1], **sac_kwargs),
        ],
    )
    artifacts_dir = Path.cwd() / "artifacts"
    checkpoint_dir = artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_freq = 10_000
    checkpoint_targets = sorted(
        {
            period
            for period in range(checkpoint_freq, total_periods + 1, checkpoint_freq)
        }
    )
    saved_steps = set()

    def periodic_checkpoint(step: int):
        if step not in saved_steps and step in checkpoint_targets:
            suffix = f"step{step:05d}"
            checkpoint_networks(
                env,
                checkpoint_dir,
                suffix=suffix,
                timestamp=timestamp,
            )
            print(f"Saved checkpoint artifacts at step {step}")
            saved_steps.add(step)

    env.play_game(checkpoint_callback=periodic_checkpoint, learn_start=512)

    # Persist training histories for each agent 
    saved_agents: List[Tuple[int, dict]] = []

    prices_history = np.asarray(env.price_history)
    profits_history = np.asarray(env.reward_history)

    for idx, agent in enumerate(env.agents, start=1):
        agent_index = idx - 1
        prices = (
            prices_history[:, agent_index]
            if prices_history.ndim > 1
            else prices_history
        )
        profits = (
            profits_history[:, agent_index]
            if profits_history.ndim > 1
            else profits_history
        )
        histories = {
            "profits": profits,
            "prices": prices,
            "q_loss": env.q_loss_history[agent_index],
            "q_baseline": env.q_baseline_history[agent_index],
            "grad_norm": env.grad_norm_history[agent_index],
            "temperature": env.temperature_history[agent_index],
            "average_reward": env.average_reward_history[agent_index],
            "policy_loss": env.policy_loss_history[agent_index],
            "policy_entropy": env.policy_entropy_history[agent_index],
            "policy_kl": env.policy_kl_history[agent_index],
        }
        paths = {}
        for name, history in histories.items():
            if history is None or len(history) == 0:
                continue
            path = artifacts_dir / f"agent{idx}_{name}_{timestamp}.npy"
            np.save(path, np.asarray(history))
            paths[name] = path

        if paths:
            saved_agents.append((idx, paths))
    checkpoint_networks(
        env, checkpoint_dir, suffix="final", timestamp=timestamp
    )
    print("Simulation done.")


if __name__ == "__main__":
    run()
