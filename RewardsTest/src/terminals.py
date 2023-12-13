import numpy as np

from rlgym_compat import GameState
from rlgym_terminals import TerminalCondition, TimeoutCondition


class KickoffTerminalCondition(TerminalCondition):
    def __init__(self, fps: int):
        super().__init__()

        self.radius = 1200**2

        self.fps = fps
        self.timeoutCondition = (
            4.35  # 6 seconds and we reset the game. Needed for kickoff terminal condition
        )
        self.timeoutCondition = TimeoutCondition(
            int(round(self.fps * self.timeoutCondition))
        )

    def reset(self, initial_state: GameState):
        self.timeoutCondition.reset(initial_state)
        return

    def is_terminal(self, current_state: GameState) -> bool:
        # ===============================
        # - Ball outside the circle of 1200 radius: (x^2 + y^2) > r^2
        # - Timeout reached: 6 seconds
        # ===============================
        if (
            (current_state.ball.position[0] ** 2 + current_state.ball.position[1] ** 2)
            > self.radius
        ) or self.timeoutCondition.is_terminal(current_state):
            return True
        else:
            return False
