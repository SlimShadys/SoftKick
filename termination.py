import numpy as np
import rlgym_sim as rlgym
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition

class KickoffTerminalCondition(TerminalCondition):
  def __init__(self, fps: int):
    super().__init__()
    self.fps = fps
    self.radius = 1200**2
    self.timeoutNoTouchSeconds = 100 # 100 seconds of no touch and we reset the game

    self.noTouchTimeoutCondition = NoTouchTimeoutCondition(self.fps * self.timeoutNoTouchSeconds)

  def reset(self, initial_state: GameState):
    self.noTouchTimeoutCondition.reset(initial_state)
    return
  
  def is_terminal(self, current_state: GameState) -> bool:

    # Ball outside the circle of 1200 radius
    if np.linalg.norm(np.array([current_state.ball.position[0], current_state.ball.position[1], current_state.ball.position[2]]) - np.array([0, 0, 97])) > self.radius:
      return True
    else:
      return False
