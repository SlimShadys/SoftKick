from abc import ABC, abstractmethod

from rlgym_compat import GameState


class TerminalCondition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, initial_state: GameState):
        """
        Function to be called each time the environment is reset.

        :param initial_state: The initial state of the reset environment.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, current_state: GameState) -> bool:
        """
        Function to determine if a game state is terminal. This will be called once per step, and must return either
        `True` or `False` if the current episode should be terminated at this state.

        :param current_state: The current state of the game.

        :return: Bool representing whether the current state is a terminal one.
        """
        raise NotImplementedError


class TimeoutCondition(TerminalCondition):
    """
    A condition that will terminate an episode after some number of steps.
    """

    def __init__(self, max_steps: int):
        super().__init__()
        self.steps = 0
        self.max_steps = max_steps

    def reset(self, initial_state: GameState):
        """
        Reset the step counter.
        """

        self.steps = 0

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Increment the current step counter and return `True` if `max_steps` have passed.
        """

        self.steps += 1
        return self.steps >= self.max_steps
