from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState

# creates conditions for which the state will reset
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, \
                                                              NoTouchTimeoutCondition, \
                                                              GoalScoredCondition, \
                                                              BallTouchedCondition

class CustomTerminalCondition(TerminalCondition):
  def __init__(self, fps: int):
     super().__init__()
     self.fps = fps

     self.timeoutConditionSeconds = 300 # After 300 seconds, we reset the game
     self.timeoutNoTouchSeconds = 100 # 100 seconds of no touch

     self.timeoutCondition = TimeoutCondition(self.fps * self.timeoutConditionSeconds)
     self.noTouchTimeoutCondition = NoTouchTimeoutCondition(self.fps * self.timeoutNoTouchSeconds)
     self.ballTouchCondition = BallTouchedCondition()
     self.goalScoredCondition = GoalScoredCondition()

  def reset(self, initial_state: GameState):

    self.timeoutCondition.reset(initial_state)
    self.noTouchTimeoutCondition.reset(initial_state)
    self.goalScoredCondition.reset(initial_state)   
    self.ballTouchCondition.reset(initial_state)

  def is_terminal(self, current_state: GameState) -> bool:
    return self.timeoutCondition.is_terminal(current_state) or \
           self.noTouchTimeoutCondition.is_terminal(current_state) or \
           self.goalScoredCondition.is_terminal(current_state) or \
           self.ballTouchCondition.is_terminal(current_state)