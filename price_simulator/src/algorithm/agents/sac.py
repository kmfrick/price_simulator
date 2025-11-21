import random
import attr
import uuid
import numpy as np
import tensorflow as tf
from cpprb import ReplayBuffer
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Activation,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.initializers import (
    RandomUniform,
    Orthogonal,
    Constant,
)
from typing import List, Tuple
import os

from price_simulator.src.algorithm.agents.simple import AgentStrategy
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_TWO = tf.constant(np.log(2.0), dtype=tf.float32)
LOG_TWO_PI = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)


def build_sac_kwargs() -> dict:
    return dict(
        actor_hidden_size=1024,
        critic_hidden_size=256,
        state_dim=2,
        action_dim=1,
        batch_size=128,
        lr_actor=3e-2,
        lr_critic=3e-3,
        lr_temperature=3e-3,
        reward_step_size=0.01,
        target_entropy=-1,
        tau=0.001,
        actor_hidden_layers=2,
        critic_hidden_layers=2,
        hidden_activation="leaky_relu",
        replay_buffer_size=100_000,
    )


@attr.s(eq=False)
class SACContinuous(AgentStrategy):
    """
    Soft Actor-Critic agent for continuous actions sampled in [-1, 1]
    """
    
    ### HYPERPARAMETERS ###
    # Nb and size of hidden layers in actor network
    actor_hidden_size: int = attr.ib(default=None)
    actor_hidden_layers: int = attr.ib(default=None)
    # Nb and size of hidden layers in critic network
    critic_hidden_layers: int = attr.ib(default=None)
    critic_hidden_size: int = attr.ib(default=None)
    # Minibatch size for SGD
    batch_size: int = attr.ib(default=None)
    # Learning rates
    lr_actor: float = attr.ib(default=None)
    lr_critic: float = attr.ib(default=None)
    lr_temperature: float = attr.ib(default=None)
    reward_step_size: float = attr.ib(default=None)
    # Max gradient norm
    clip_norm: float = attr.ib(default=10.0)
    # Activation function in hidden layers
    hidden_activation: str = attr.ib(default=None)
    # Entropy constraint
    target_entropy: float = attr.ib(default=None)
    # Polyak averaging coefficient for target updates
    tau: float = attr.ib(default=None)
    # Replay buffer size
    replay_buffer_size: int = attr.ib(default=None)

    ### ENVIRONMENT CHARACTERISTICS ###
    state_dim: int = attr.ib(default=None)
    action_dim: int = attr.ib(default=None)
    ### COMPONENTS ###
    seed: int = attr.ib(default=None)
    replay_memory: ReplayBuffer = attr.ib(default=None)
    # Networks
    actor: Model = attr.ib(default=None)
    critic1: Model = attr.ib(default=None)
    critic2: Model = attr.ib(default=None)
    target_critic1: Model = attr.ib(default=None)
    target_critic2: Model = attr.ib(default=None)
    # Optimizers
    actor_opt: Optimizer = attr.ib(default=None)
    critic_opt: Optimizer = attr.ib(default=None)
    temperature_opt: Optimizer = attr.ib(default=None)
    # Parameters
    log_temperature: tf.Variable = attr.ib(default=None)
    average_reward: tf.Variable = attr.ib(default=None)
    q_baseline: tf.Variable = attr.ib(default=None)
    training_step: tf.Variable = attr.ib(default=None)

    def __hash__(self):
        """Make instance hashable for tf.function caching."""
        return id(self)

    def __attrs_post_init__(self):
        """Create networks/optimizers when first called."""
        required_params = [
            "actor_hidden_size",
            "critic_hidden_size",
            "state_dim",
            "action_dim",
            "batch_size",
            "lr_actor",
            "lr_critic",
            "lr_temperature",
            "reward_step_size",
            "target_entropy",
        ]
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(
                    f"Hyperparameter '{param}' must be provided (cannot be None)."
                )
        if self.seed is None:
            raise ValueError("SACContinuous requires an explicit seed.")
        if int(self.actor_hidden_layers) < 1 or int(self.critic_hidden_layers) < 1:
            raise ValueError("Hidden layer counts must be >= 1.")

        activation_name = str(self.hidden_activation).lower().replace("-", "_")
        self.hidden_activation = activation_name

        state_dim = int(self.state_dim)
        action_dim = int(self.action_dim)
        if action_dim != 1:
            raise NotImplementedError(
                "SACContinuous currently supports only action_dim == 1."
            )
        buffer_size = self.replay_buffer_size
        env_dict = {
            "obs": {"shape": (state_dim,)},
            "act": {"shape": (action_dim,)},
            "rew": {},
            "next_obs": {"shape": (state_dim,)},
        }
        self.replay_memory = ReplayBuffer(size=int(buffer_size), env_dict=env_dict)
        self._base_seed = tf.constant(int(self.seed), dtype=tf.int32)
        def make_activation(name_prefix: str):
            return Activation(activation_name, name=f"{name_prefix}_{activation_name}")

        def build_actor():
            uid = uuid.uuid4().hex[:8]
            inp = Input(shape=(state_dim,), name=f"actor_input_{uid}")
            x = inp
            for layer_idx in range(int(self.actor_hidden_layers)):
                layer_name = f"actor_dense{layer_idx + 1}_{uid}"
                x = Dense(
                    self.actor_hidden_size,
                    activation=None,
                    name=layer_name,
                    kernel_initializer=Orthogonal()
                )(x)
                #x = LayerNormalization(name=f"actor_ln{layer_idx + 1}_{uid}")(x)
                x = make_activation(f"actor_act{layer_idx + 1}_{uid}")(x)
            gain = 0.1
            mu = Dense(
                action_dim,
                kernel_initializer=RandomUniform(minval=-gain, maxval=gain),
                name=f"actor_mu_{uid}",
            )(x)
            log_std = Dense(
                action_dim,
                kernel_initializer=RandomUniform(minval=-gain, maxval=gain),
                name=f"actor_std_{uid}",
            )(x)
            return Model(inputs=inp, outputs=[mu, log_std], name=f"actor_model_{uid}")

        def build_critic():
            uid = uuid.uuid4().hex[:8]
            inp = Input(shape=(state_dim + action_dim,), name=f"critic_input_{uid}")
            x = inp
            for layer_idx in range(int(self.critic_hidden_layers)):
                layer_name = f"critic_dense{layer_idx + 1}_{uid}"
                x = Dense(
                    self.critic_hidden_size,
                    activation=None,
                    name=layer_name,
                    kernel_initializer=Orthogonal()
                )(x)
                #x = LayerNormalization(name=f"critic_ln{layer_idx + 1}_{uid}")(x)
                x = make_activation(f"critic_act{layer_idx + 1}_{uid}")(x)
            out = Dense(
                1,
                name=f"critic_out_{uid}",
                bias_initializer=Constant(value=2.0),
            )(x)
            return Model(inputs=inp, outputs=out, name=f"critic_model_{uid}")

        self.actor = build_actor()
        self.critic1 = build_critic()
        self.critic2 = build_critic()
        self.target_critic1 = build_critic()
        self.target_critic2 = build_critic()
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.log_temperature = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.actor_opt = Adam(learning_rate=self.lr_actor)
        self.critic_opt = Adam(learning_rate=self.lr_critic)
        self.temperature_opt = Adam(learning_rate=self.lr_temperature)

        self.average_reward = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.q_baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.training_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._tau = tf.constant(float(self.tau), dtype=tf.float32)
        batch_dim = int(self.batch_size)
        sa_dim = int(self.state_dim) + int(self.action_dim)
        self._zero_sa = tf.zeros((1, sa_dim), dtype=tf.float32)

        # XLA-compiled sampling paths with fixed shapes (single and training batch)
        def _compiled_sample_action(state_tf, seed, deterministic):
            mu, log_std = self.actor(state_tf, training=False)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)

            noise = tf.random.stateless_normal(shape=tf.shape(mu), seed=seed)
            z = tf.where(deterministic, mu, mu + std * noise)

            action = tf.tanh(z)
            log_prob = -0.5 * (
                ((z - mu) / (std + 1e-8)) ** 2
                + 2 * log_std
                + LOG_TWO_PI
            )
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            # Mathematically equivalent to log(1 - tanh(z)^2) but more stable
            log_det_jacobian = 2 * (LOG_TWO - z - tf.math.softplus(-2. * z))
            logp = log_prob - tf.reduce_sum(log_det_jacobian, axis=1, keepdims=True)
            return action, logp

        self._sample_action_single_tf = tf.function(
            _compiled_sample_action,
            reduce_retracing=True,
            jit_compile=True,
            input_signature=[
                tf.TensorSpec(shape=(1, state_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.bool),
            ],
        )
        self._sample_action_batch_train_tf = tf.function(
            _compiled_sample_action,
            reduce_retracing=True,
            jit_compile=True,
            input_signature=[
                tf.TensorSpec(shape=(batch_dim, state_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.bool),
            ],
        )

        # Compile training step per instance to avoid global cache accumulation (memory leak)
        self._train_step = tf.function(
            self._train_step_impl,
            reduce_retracing=True,
            jit_compile=True,
            input_signature=[
                tf.TensorSpec(shape=(batch_dim, state_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_dim, action_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_dim, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_dim, state_dim), dtype=tf.float32),
            ],
        )

    def who_am_i(self) -> str:
        return (
            f"SACContinuous (actor_lr: {self.lr_actor}, " f"critic_lr: {self.lr_critic}"
        )

    def play_price(
        self,
        state: Tuple[float],
        n_period: int,
        t: int,
        use_target: bool = None,
    ) -> float:
        """Sample a price from the stochastic policy."""

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        # Use the environment timestep for the stateless RNG seed so action sampling
        # remains reproducible but not constant before training starts.
        step_seed = tf.constant(int(t), dtype=tf.int32)
        seed = tf.stack(
            [
                self._base_seed,
                self._base_seed + step_seed,
            ]
        )
        action, _ = self._sample_action_single_tf(state_tensor, seed, False)
        action_np = action.numpy().flatten()[0]
        action_np = np.clip(action_np, -1.0, 1.0)
        return float(action_np)

    def _train_step_impl(self, states_tf, actions_tf, rewards_tf, next_states_tf):
        """Compiled training step for better performance."""

        # Update critics
        with tf.GradientTape() as tape:
            step_seed = tf.cast(self.training_step, tf.int32)
            seed_next = tf.stack(
                [
                    self._base_seed,
                    self._base_seed + step_seed * 2,
                ]
            )
            next_actions, next_logp = self._sample_action_batch_train_tf(
                next_states_tf, seed_next, False
            )
            target_q1 = self.target_critic1(
                tf.concat([next_states_tf, next_actions], axis=-1), training=False
            )
            target_q2 = self.target_critic2(
                tf.concat([next_states_tf, next_actions], axis=-1), training=False
            )
            target_q = (target_q1 + target_q2) / 2.0
            zero_sa = tf.cast(self._zero_sa, next_states_tf.dtype)
            baseline_q1 = self.target_critic1(zero_sa, training=False)
            baseline_q2 = self.target_critic2(zero_sa, training=False)
            baseline = tf.reduce_mean((baseline_q1 + baseline_q2) / 2.0)
            self.q_baseline.assign(baseline)

            # Average-reward target
            temperature = tf.exp(tf.stop_gradient(self.log_temperature))
            target = rewards_tf - self.average_reward + (target_q - baseline) - temperature * next_logp

            q1 = self.critic1(
                tf.concat([states_tf, actions_tf], axis=-1), training=True
            )
            q2 = self.critic2(
                tf.concat([states_tf, actions_tf], axis=-1), training=True
            )
            critic1_loss = tf.reduce_mean(tf.square(q1 - target))
            critic2_loss = tf.reduce_mean(tf.square(q2 - target))
            critic_loss = critic1_loss + critic2_loss

        critic_vars = self.critic1.trainable_weights + self.critic2.trainable_weights
        critic_grads = tape.gradient(critic_loss, critic_vars)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.clip_norm)
        critic_grad_norm = tf.linalg.global_norm(critic_grads)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_vars))

        # Update actor
        with tf.GradientTape(persistent=True) as tape2:
            mu_old, log_std_old = self.actor(states_tf, training=True)
            log_std_old = tf.clip_by_value(log_std_old, LOG_STD_MIN, LOG_STD_MAX)
            std_old = tf.exp(log_std_old)
            seed_pi = tf.stack(
                [
                    self._base_seed,
                    self._base_seed + step_seed * 2 + 1,
                ]
            )
            noise = tf.random.stateless_normal(shape=tf.shape(mu_old), seed=seed_pi)
            z = mu_old + std_old * noise
            pi = tf.tanh(z)
            log_prob = -0.5 * (
                ((z - mu_old) / (std_old + 1e-8)) ** 2
                + 2 * log_std_old
                + LOG_TWO_PI
            )
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            log_det_jacobian = 2 * (LOG_TWO - z - tf.math.softplus(-2. * z))
            log_det_jacobian_sum = tf.reduce_sum(log_det_jacobian, axis=1, keepdims=True)
            logp_pi = log_prob - log_det_jacobian_sum
            q1_pi = self.critic1(tf.concat([states_tf, pi], axis=-1), training=False)
            q2_pi = self.critic2(tf.concat([states_tf, pi], axis=-1), training=False)
            q_pi = (q1_pi + q2_pi) / 2.0
            temperature_detached = tf.exp(tf.stop_gradient(self.log_temperature))
            actor_loss = tf.reduce_mean(temperature_detached * logp_pi - q_pi)

            # Calculate entropy
            entropy = -tf.reduce_mean(logp_pi)

            temperature_loss = -tf.reduce_mean(
                self.log_temperature * tf.stop_gradient(logp_pi + self.target_entropy)
            )

        actor_grads = tape2.gradient(actor_loss, self.actor.trainable_weights)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.clip_norm)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

        temperature_grad = tape2.gradient(temperature_loss, [self.log_temperature])
        self.temperature_opt.apply_gradients(
            zip(temperature_grad, [self.log_temperature])
        )
        del tape2

        mu_old = tf.stop_gradient(mu_old)
        log_std_old = tf.stop_gradient(log_std_old)
        z = tf.stop_gradient(z)
        log_det_jacobian_sum = tf.stop_gradient(log_det_jacobian_sum)
        logp_pi = tf.stop_gradient(logp_pi)
        mu_new, log_std_new = self.actor(states_tf, training=False)
        log_std_new = tf.clip_by_value(log_std_new, LOG_STD_MIN, LOG_STD_MAX)
        std_new = tf.exp(log_std_new)
        log_prob_new = -0.5 * (
            ((z - mu_new) / (std_new + 1e-8)) ** 2
            + 2 * log_std_new
            + LOG_TWO_PI
        )
        log_prob_new = tf.reduce_sum(log_prob_new, axis=1, keepdims=True)
        logp_new = log_prob_new - log_det_jacobian_sum
        policy_kl = tf.reduce_mean(logp_pi - logp_new)
        # Update training step for reproducible stateless RNG seeding
        self.training_step.assign_add(1)

        for target_var, source_var in zip(
            self.target_critic1.variables, self.critic1.variables
        ):
            target_var.assign(
                self._tau * source_var + (1.0 - self._tau) * target_var
            )
        for target_var, source_var in zip(
            self.target_critic2.variables, self.critic2.variables
        ):
            target_var.assign(
                self._tau * source_var + (1.0 - self._tau) * target_var
            )

        # Average-reward baseline update (EMA of entropy-adjusted reward + TD correction)
        with tf.name_scope("avg_reward_update"):
            cur_q1_targ = self.target_critic1(
                tf.concat([states_tf, actions_tf], axis=-1), training=False
            )
            cur_q2_targ = self.target_critic2(
                tf.concat([states_tf, actions_tf], axis=-1), training=False
            )
            cur_q_targ = (cur_q1_targ + cur_q2_targ) / 2.0

            next_q1_targ = self.target_critic1(
                tf.concat([next_states_tf, next_actions], axis=-1), training=False
            )
            next_q2_targ = self.target_critic2(
                tf.concat([next_states_tf, next_actions], axis=-1), training=False
            )
            next_q_targ = (next_q1_targ + next_q2_targ) / 2.0

            temperature_detached = tf.exp(tf.stop_gradient(self.log_temperature))
            new_avg = tf.reduce_mean(
                rewards_tf
                - temperature_detached * logp_pi
                + next_q_targ
                - cur_q_targ
            )
            self.average_reward.assign(
                (1.0 - self.reward_step_size) * self.average_reward
                + self.reward_step_size * new_avg
            )

        # Return metrics for logging
        return {
            "critic_loss": critic_loss,
            "critic_grad_norm": critic_grad_norm,
            "actor_loss": actor_loss,
            "entropy": entropy,
            "policy_kl": policy_kl,
            "q_baseline": baseline,
        }

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        previous_state: Tuple[float],
        state: Tuple[float],
        next_state: Tuple[float],
    ):
        """Learn from experience using compiled training step."""
        # Store experience
        self.replay_memory.add(
            obs=np.asarray(state, dtype=np.float32),
            act=np.asarray([action], dtype=np.float32),
            rew=float(reward),
            next_obs=np.asarray(next_state, dtype=np.float32),
        )

        if self.replay_memory.get_stored_size() < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = self.replay_memory.sample(self.batch_size)
        states = batch["obs"]
        actions = batch["act"]
        rewards = batch["rew"]
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        next_states = batch["next_obs"]
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # Perform compiled training step
        metrics = self._train_step(
            states_tf,
            actions_tf,
            rewards_tf,
            next_states_tf,
        )

        return metrics

    def _sample_action(
        self,
        state_tf: tf.Tensor,
        deterministic: bool = False,
        seed_step: int = 0,
    ):
        """Single-action (batch=1) XLA path; analysis uses _sample_action_batch for vectorized draws."""
        if seed_step is None:
            step_seed = tf.cast(self.training_step, tf.int32)
        else:
            step_seed = tf.cast(seed_step, tf.int32)
        seed = tf.stack(
            [self._base_seed, self._base_seed + step_seed]
        )
        return self._sample_action_single_tf(state_tf, seed, deterministic)

    def _sample_action_batch(self, state_tf: tf.Tensor, deterministic: bool = False):
        """Sample a batch of actions (non-compiled; used by analysis/plots)."""
        mu, log_std = self.actor(state_tf, training=False)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        step_seed = tf.cast(self.training_step, tf.int32)
        seed = tf.stack(
            [self._base_seed, self._base_seed + step_seed]
        )
        noise = tf.random.stateless_normal(shape=tf.shape(mu), seed=seed)
        z = tf.where(deterministic, mu, mu + std * noise)

        action = tf.tanh(z)
        # log prob with tanh correction
        log_prob = -0.5 * (
            ((z - mu) / (std + 1e-8)) ** 2
            + 2 * log_std
            + LOG_TWO_PI
        )
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        log_det_jacobian = 2 * (LOG_TWO - z - tf.math.softplus(-2. * z))
        logp = log_prob - tf.reduce_sum(log_det_jacobian, axis=1, keepdims=True)

        return action, logp
