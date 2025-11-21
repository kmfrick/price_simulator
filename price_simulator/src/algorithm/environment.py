import abc
import copy
import random
from typing import List, Tuple, Optional, Callable

import attr

import numpy as np
from tqdm import tqdm
from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.demand import (
    LogitDemand,
    MarketDemandStrategy,
    PrisonersDilemmaDemand,
)
from price_simulator.src.algorithm.equilibrium import EquilibriumCalculator


@attr.s
class EnvironmentStrategy(metaclass=abc.ABCMeta):
    """Top-level interface for Environment."""

    agents: List[AgentStrategy] = attr.ib(factory=list)
    possible_prices: List[float] = attr.ib(factory=list)
    demand: MarketDemandStrategy = attr.ib(factory=LogitDemand)
    nash_prices: np.array = attr.ib(init=False)
    monopoly_prices: np.array = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Compute Nash Price and Monopoly price after initialization."""
        if len(self.agents) > 0.0:
            if isinstance(self.demand, PrisonersDilemmaDemand):
                assert (
                    len(self.possible_prices) > 0.0
                ), "Priosoners Dilemma needs two possible prices"
                self.monopoly_prices = [
                    max(self.possible_prices),
                    max(self.possible_prices),
                ]
                self.nash_prices = np.array(
                    [min(self.possible_prices), min(self.possible_prices)]
                )
            else:
                marginal_costs = [agent.marginal_cost for agent in self.agents]
                qualities = [agent.quality for agent in self.agents]
                self.monopoly_prices = EquilibriumCalculator(
                    demand=self.demand
                ).get_monopoly_outcome(qualities, marginal_costs)
                self.nash_prices = EquilibriumCalculator(
                    demand=self.demand
                ).get_nash_equilibrium(qualities, marginal_costs)

    @abc.abstractmethod
    def play_game(self):
        raise NotImplementedError


@attr.s
class DiscreteSynchronEnvironment(EnvironmentStrategy):
    """Environment for discrete states and prices.

    Before the first iteration, prices are randomly initialized.
    Agents set prices at the same time.
    After choosing prices, demand and rewards are calculated.
    Then agents have the opportunity to learn.
    """

    n_periods: int = attr.ib(default=1)
    markup: float = attr.ib(default=0.1)
    n_prices: int = attr.ib(default=15)
    convergence_after: int = attr.ib(default=np.inf)
    history_after: int = attr.ib(default=np.inf)
    price_history: List = attr.ib(factory=list)
    quantity_history: List = attr.ib(factory=list)
    reward_history: List = attr.ib(factory=list)

    @n_periods.validator
    def check_n_periods(self, attribute, value):
        if not 0 < value:
            raise ValueError("Number of periods must be strictly positive")

    @markup.validator
    def check_markup(self, attribute, value):
        if not 0 <= value:
            raise ValueError("Price markup must be positive")

    @n_prices.validator
    def check_n_prices(self, attribute, value):
        if not 0 < value:
            raise ValueError("Number of prices must be strictly positive")

    def play_game(self) -> int:

        qualities = tuple(agent.quality for agent in self.agents)
        marginal_costs = tuple(agent.marginal_cost for agent in self.agents)

        # initialize first rounds
        if len(self.possible_prices) == 0:
            self.possible_prices = self.get_price_range(
                min(self.nash_prices),
                max(self.monopoly_prices),
                self.markup,
                self.n_prices,
            )
        previous_state = tuple(random.choices(self.possible_prices, k=len(self.agents)))
        state = tuple(
            agent.play_price(previous_state, self.possible_prices, self.n_periods, 0)
            for agent in self.agents
        )
        quantities = self.demand.get_quantities(state, qualities)
        previous_rewards = np.multiply(np.subtract(state, marginal_costs), quantities)

        for t in range(self.n_periods):

            # agents decide about there prices (hereafter is the state different)
            next_state = tuple(
                agent.play_price(state, self.possible_prices, self.n_periods, t)
                for agent in self.agents
            )

            # demand is estimated for prices
            quantities = self.demand.get_quantities(next_state, qualities)
            rewards = np.multiply(np.subtract(next_state, marginal_costs), quantities)

            # assert that everything is correct
            assert (np.array(quantities) >= 0.0).all(), "Quantities cannot be negative"
            assert (np.array(next_state) >= 0.0).all(), "Prices cannot be negative"

            # agents learn
            for agent, action, previous_action, reward, previous_reward in zip(
                self.agents, next_state, state, rewards, previous_rewards
            ):
                agent.learn(
                    previous_reward=previous_reward,
                    reward=reward,
                    previous_action=previous_action,
                    action=action,
                    action_space=self.possible_prices,
                    previous_state=previous_state,
                    state=state,
                    next_state=next_state,
                )

            # update variables
            previous_state = copy.deepcopy(state)
            state = copy.deepcopy(next_state)
            previous_rewards = copy.deepcopy(rewards)

            # save prices for the last periods
            if t > self.history_after:
                self.price_history.append(previous_state)
                self.quantity_history.append(quantities)
                self.reward_history.append(rewards)

        return t

    @staticmethod
    def get_price_range(
        nash_price: float, monopoly_price: float, markup: float, n_step: int
    ) -> List:
        increase = (monopoly_price - nash_price) * markup
        return list(
            np.linspace(nash_price - increase, monopoly_price + increase, n_step)
        )


@attr.s
class ReformulationEnvironment(DiscreteSynchronEnvironment):
    """Environment with reformulated state representation."""

    @staticmethod
    def reformulate(actions: Tuple) -> Tuple:
        return tuple([min(actions), max(actions), np.mean(actions)])

    def play_game(self) -> int:

        qualities = tuple(agent.quality for agent in self.agents)
        marginal_costs = tuple(agent.marginal_cost for agent in self.agents)

        # initialize first rounds
        if len(self.possible_prices) == 0:
            self.possible_prices = self.get_price_range(
                min(self.nash_prices),
                max(self.monopoly_prices),
                self.markup,
                self.n_prices,
            )
        previous_state = self.reformulate(
            tuple(random.choices(self.possible_prices, k=len(self.agents)))
        )
        previous_actions = tuple(
            agent.play_price(previous_state, self.possible_prices, self.n_periods, 0)
            for agent in self.agents
        )
        state = self.reformulate(previous_actions)
        quantities = self.demand.get_quantities(previous_actions, qualities)
        previous_rewards = np.multiply(
            np.subtract(previous_actions, marginal_costs), quantities
        )

        for t in range(self.n_periods):

            # agents decide about there prices (hereafter is the state different)
            actions = tuple(
                agent.play_price(state, self.possible_prices, self.n_periods, t)
                for agent in self.agents
            )
            next_state = self.reformulate(actions)

            # demand is estimated for prices
            quantities = self.demand.get_quantities(actions, qualities)
            rewards = np.multiply(np.subtract(actions, marginal_costs), quantities)

            # assert that everything is correct
            assert (np.array(quantities) >= 0.0).all(), "Quantities cannot be negative"
            assert (np.array(actions) >= 0.0).all(), "Prices cannot be negative"

            # agents learn
            for agent, action, previous_action, reward, previous_reward in zip(
                self.agents, actions, previous_actions, rewards, previous_rewards
            ):
                agent.learn(
                    previous_reward=previous_reward,
                    reward=reward,
                    previous_action=previous_action,
                    action=action,
                    action_space=self.possible_prices,
                    previous_state=previous_state,
                    state=state,
                    next_state=next_state,
                )

            # update variables
            previous_state = copy.deepcopy(state)
            state = copy.deepcopy(next_state)
            previous_rewards = copy.deepcopy(rewards)
            previous_actions = copy.deepcopy(actions)

            # save prices for the last periods
            if t > self.history_after:
                self.price_history.append(actions)
                self.quantity_history.append(quantities)
                self.reward_history.append(rewards)

        return t


@attr.s
class ContSynchronEnvironment(EnvironmentStrategy):
    """Environment for continuous states and prices.

    Before the first iteration, prices are randomly initialized.
    Agents set prices at the same time.
    After choosing prices, demand and rewards are calculated.
    Then agents have the opportunity to learn.
    """

    n_periods: int = attr.ib(default=1)
    markup: float = attr.ib(default=0.1)
    convergence_after: int = attr.ib(default=np.inf)
    price_history: List = attr.ib(factory=list)
    quantity_history: List = attr.ib(factory=list)
    reward_history: List = attr.ib(factory=list)
    q_loss_history: List = attr.ib(factory=list)
    q_baseline_history: List = attr.ib(factory=list)
    grad_norm_history: List = attr.ib(factory=list)
    temperature_history: List = attr.ib(factory=list)
    average_reward_history: List = attr.ib(factory=list)
    policy_loss_history: List = attr.ib(factory=list)
    policy_entropy_history: List = attr.ib(factory=list)
    policy_kl_history: List = attr.ib(factory=list)

    def __attrs_post_init__(self):
        # Run the parent's logic first
        super().__attrs_post_init__()

        # Then run the child's logic
        if not self.agents:
            raise ValueError("ContSynchronEnvironment requires at least one agent")

        qualities = np.asarray(
            [agent.quality for agent in self.agents], dtype=np.float64
        )
        marginal_costs = np.asarray(
            [agent.marginal_cost for agent in self.agents], dtype=np.float64
        )

        self.monopoly_price = np.min(self.monopoly_prices)
        self.nash_price = np.min(self.nash_prices)
        increase = (self.monopoly_price - self.nash_price) * self.markup
        self.min_price = self.nash_price - increase
        self.max_price = self.monopoly_price + increase
        self.price_range = self.max_price - self.min_price
        self.price_midpoint = (self.max_price + self.min_price) / 2
        self.price_sd = np.sqrt(
            (self.price_range**2) / 12
        )  # Under uniform random pricing

        nash_quantities = self.demand.get_quantities(self.nash_prices, qualities)
        self.nash_profits = np.multiply(
            np.subtract(self.nash_prices, marginal_costs), nash_quantities
        )
        monopoly_quantities = self.demand.get_quantities(
            self.monopoly_prices, qualities
        )
        self.monopoly_profits = np.multiply(
            np.subtract(self.monopoly_prices, marginal_costs), monopoly_quantities
        )
        denom = self.monopoly_profits - self.nash_profits
        if np.any(np.isclose(denom, 0.0)):
            raise ValueError(
                "Cannot normalize rewards: Nash and monopoly profits are (near) identical."
            )
        self.q_loss_history = [[] for _ in self.agents]
        self.q_baseline_history = [[] for _ in self.agents]
        self.grad_norm_history = [[] for _ in self.agents]
        self.temperature_history = [[] for _ in self.agents]
        self.average_reward_history = [[] for _ in self.agents]
        self.policy_loss_history = [[] for _ in self.agents]
        self.policy_entropy_history = [[] for _ in self.agents]
        self.policy_kl_history = [[] for _ in self.agents]

    @n_periods.validator
    def check_n_periods(self, attribute, value):
        if not 0 < value:
            raise ValueError("Number of periods must be strictly positive")

    @markup.validator
    def check_markup(self, attribute, value):
        if not 0 <= value:
            raise ValueError("Price markup must be positive")

    def _denormalize_action(self, action: float) -> float:
        """Map normalized action in [-1, 1] to price space."""
        return (action + 1) * 0.5 * (self.max_price - self.min_price) + self.min_price

    def _normalize_rewards(self, rewards: np.array) -> np.array:
        """Normalize rewards: -1 = Nash profit, 1 = monopoly profit."""
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        denom = self.monopoly_profits - self.nash_profits
        return (2 * (rewards_arr - self.nash_profits) / denom) - 1

    def play_game(
        self,
        learn_start: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
    ) -> int:

        qualities = tuple(agent.quality for agent in self.agents)
        marginal_costs = tuple(agent.marginal_cost for agent in self.agents)

        # initialize first rounds
        state = tuple(0 for agent in self.agents)  # normalized
        previous_state = tuple(0 for agent in self.agents)  # normalized
        previous_rewards = np.asarray([0 for agent in self.agents])  # normalized
        GRAD_STEPS = 1
        with tqdm(range(self.n_periods)) as tq:
            for t in tq:
                if t < learn_start:
                    actions = tuple(np.random.uniform(-1, 1) for agent in self.agents)
                else:
                    # agents decide about their prices (hereafter is the state different)
                    actions = tuple(
                        agent.play_price(state, self.n_periods, t)
                        for agent in self.agents
                    )  # normalized
                assert (
                    np.array(actions) >= -1.0
                ).all(), f"Actions have to be > -1, got {actions}"
                assert (
                    np.array(actions) <= 1.0
                ).all(), f"Actions have to be  < 1, got {actions}"

                # demand is estimated for prices
                prices = tuple(
                    self._denormalize_action(a) for a in actions
                )  # UNNORMALIZED
                assert (np.array(prices) >= 0.0).all(), "Prices cannot be negative"
                quantities = self.demand.get_quantities(
                    prices, qualities
                )  # UNNORMALIZED
                assert (
                    np.array(quantities) >= 0.0
                ).all(), "Quantities cannot be negative"
                profits = np.multiply(
                    np.subtract(prices, marginal_costs), quantities
                )  # UNNORMALIZED
                rewards = self._normalize_rewards(profits)  # normalized

                # agents learn
                if t > learn_start:
                    metrics_by_agent = None
                    for k in range(GRAD_STEPS):
                        metrics_by_agent = []
                        for (
                            agent,
                            action,
                            previous_action,
                            reward,
                            previous_reward,
                        ) in zip(
                            self.agents, actions, state, rewards, previous_rewards
                        ):
                            metrics_by_agent.append(
                                agent.learn(
                                    previous_reward=previous_reward,  # normalized
                                    reward=reward,  # normalized
                                    previous_action=previous_action,  # normalized
                                    action=action,  # normalized
                                    previous_state=previous_state,  # normalized
                                    state=state,  # normalized
                                    next_state=actions,  # normalized
                                )
                            )
                    if metrics_by_agent:
                        for agent_idx, metrics in enumerate(metrics_by_agent):
                            if not metrics:
                                continue
                            self.q_loss_history[agent_idx].append(
                                float(metrics["critic_loss"].numpy())
                            )
                            self.q_baseline_history[agent_idx].append(
                                float(metrics["q_baseline"].numpy())
                            )
                            self.grad_norm_history[agent_idx].append(
                                float(metrics["critic_grad_norm"].numpy())
                            )
                            self.temperature_history[agent_idx].append(
                                float(
                                    np.exp(
                                        self.agents[agent_idx].log_temperature.numpy()
                                    )
                                )
                            )
                            self.average_reward_history[agent_idx].append(
                                float(self.agents[agent_idx].average_reward.numpy())
                            )
                            self.policy_loss_history[agent_idx].append(
                                float(metrics["actor_loss"].numpy())
                            )
                            self.policy_entropy_history[agent_idx].append(
                                float(metrics["entropy"].numpy())
                            )
                            self.policy_kl_history[agent_idx].append(
                                float(metrics["policy_kl"].numpy())
                            )

                # update variables
                previous_state = copy.deepcopy(state)  # normalized
                state = copy.deepcopy(actions)  # normalized
                previous_rewards = copy.deepcopy(rewards)  # normalized

                # Save to history output
                self.price_history.append(actions)
                self.quantity_history.append(quantities)
                self.reward_history.append(rewards)

                if checkpoint_callback is not None:
                    checkpoint_callback(t + 1)
                postfix = {}
                if self.reward_history:
                    recent = np.asarray(self.reward_history[-1000:], dtype=np.float64)
                    if recent.ndim == 1:
                        recent = recent.reshape(-1, 1)
                    avg_gains = np.mean(recent, axis=0)
                    for idx, value in enumerate(avg_gains, start=1):
                        postfix[f"avgG_{idx}"] = f"{value:.3f}"
                if postfix:
                    tq.set_postfix(postfix, refresh=False)
        return t
