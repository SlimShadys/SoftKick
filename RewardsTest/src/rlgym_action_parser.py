from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import gym.spaces
import numpy as np

from rlgym_compat import GameState, PlayerData


class ActionParser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        """
        Function that returns the action space type. It will be called during the initialization of the environment.

        :return: The type of the action space
        """
        raise NotImplementedError

    @abstractmethod
    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        """
        Function that parses actions from the action space into a format that rlgym understands.
        The expected return value is a numpy float array of size (n, 8) where n is the number of agents.
        The second dimension is indexed as follows: throttle, steer, yaw, pitch, roll, jump, boost, handbrake.
        The first five values are expected to be in the range [-1, 1], while the last three values should be either 0 or 1.

        :param actions: An object of actions, as passed to the `env.step` function.
        :param state: The GameState object of the current state that were used to generate the actions.

        :return: the parsed actions in the rlgym format.
        """
        raise NotImplementedError


class DiscreteAction(ActionParser):
    """
    Simple discrete action space. All the analog actions have 3 bins by default: -1, 0 and 1.
    """

    def __init__(self, n_bins=3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([self._n_bins] * 5 + [2] * 3)

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 8)).astype(dtype=np.float32)

        # map all binned actions from {0, 1, 2 .. n_bins - 1} to {-1 .. 1}.
        actions[..., :5] = actions[..., :5] / (self._n_bins // 2) - 1

        return actions
