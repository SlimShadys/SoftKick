from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward # allows to set reward for each event occurring
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward # gives the agent a reward for its velocity in the direction of the ball
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward

import numpy as np

class CustomReward(RewardFunction):
  def __init__(self) -> None:
    super().__init__()
    self.rewardWeights = {
        "ball_touched": 7.0,
        "velocity_player_to_ball": 2.0,
        "event": 1.0}

    self.ballTouchedByPlayer = TouchBallReward()
    self.velocityPlayerToBallReward = VelocityPlayerToBallReward()
    self.eventReward = EventReward(
    shot=10.0,
    boost_pickup=2.0,
    )

  def reset(self, initial_state: GameState):
    self.eventReward.reset(initial_state)
    self.velocityPlayerToBallReward.reset(initial_state)
    self.ballTouchedByPlayer.reset(initial_state)

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    event = self.eventReward.get_reward(player, state, previous_action) # Get reward for events
    velocityPlayerToBall = self.velocityPlayerToBallReward.get_reward(player, state, previous_action) # Get reward for velocity of player to ball
    ballTouched = self.ballTouchedByPlayer.get_reward(player, state, previous_action) # Get reward for touching the ball

    # Return the sum of the weighted rewards
    reward = event * self.rewardWeights["event"] + \
             ballTouched * self.rewardWeights["ball_touched"] + \
             velocityPlayerToBall * self.rewardWeights["velocity_player_to_ball"]

    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
         # Maybe we shouldn't return 0 when the episode ends.
         # Returning a positive reward if the agent is the last one to touch the ball?
        return 0.0