import numpy as np
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

class KickoffTerminalCondition(TerminalCondition):
  def __init__(self, fps: int):
    super().__init__()

    self.radius = 1200**2

    self.fps = fps
    self.timeoutCondition = 4.35 # 4.5 seconds and we reset the game. Needed for kickoff terminal condition
    self.timeoutCondition = TimeoutCondition(int(round(self.fps * self.timeoutCondition)))

  def reset(self, initial_state: GameState):
    self.timeoutCondition.reset(initial_state)
    return
  
  def is_terminal(self, current_state: GameState) -> bool:
    # ===============================
    # - Ball outside the circle of 1200 radius: (x^2 + y^2) > r^2
    # - Timeout reached: 4.5 seconds
    # ===============================
    ball_x, ball_y, ball_z = current_state.ball.position

    # Check if the ball is outside the radius or timeout condition is met
    if (ball_x**2 + ball_y**2 > self.radius) or (self.timeoutCondition.is_terminal(current_state)):
        return True
        
    return False
