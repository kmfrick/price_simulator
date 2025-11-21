from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SimulationRun:
    run_id: str
    final_actor_paths: List[Path]
    final_critic1_paths: List[Path]
    final_critic2_paths: List[Path]
    prices: np.ndarray
    profits: np.ndarray
    grad_norm: np.ndarray
    average_reward: np.ndarray
    q_baseline: np.ndarray
    policy_loss: np.ndarray
    policy_entropy: np.ndarray
    q_loss: np.ndarray
    temperature: np.ndarray
    policy_kl: Optional[np.ndarray]
    checkpoints: List[Dict[str, object]]
