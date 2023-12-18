from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

import numpy as np

class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / CAR_MAX_SPEED

class CustomReward(RewardFunction):
  def __init__(self, gamma = 0.9908006132652293) -> None:
    super().__init__()
    self.rewardWeights = {
        "ball_touched": 4.00,
        "velocity_player_to_ball": 0.20,
        "naive_speed": 0.50,
        "event": 0.03
    }

    self.ballTouchedByPlayer = TouchBallReward()                          # Returns 1.0 if the player touches the ball       / Max 1.0
    self.velocityPlayerToBallReward = VelocityPlayerToBallReward()        # Returns the velocity of the player to the ball   / Max 1.0
    self.naiveSpeedReward = NaiveSpeedReward()                            # Returns the naive speed of the player            / Max 1.0
    self.eventReward = EventReward(                                       # Returns a reward for each event                  / Max 1.0 * 1.0 + 1.0 * 2.0 = 3.00
      touch = 2.0,
      boost_pickup = 1.0,
    )

    #self.gamma = gamma      # 0.9908006132652293
    #self.upperBound = 9.25  # Maximum reward per tick
    #self.finalUpperBound = (self.upperBound / (1 - self.gamma)) / 25 # 25 is a magic number that we found to work discretely well.
    
    # New calculation for the final reward (given by RLBot over 100 episodes)
    threshold_to_add = 0.5
    self.finalUpperBound = 0.3686379850539045 + threshold_to_add 

  def reset(self, initial_state: GameState):
    self.ballTouchedByPlayer.reset(initial_state)
    self.velocityPlayerToBallReward.reset(initial_state)
    self.naiveSpeedReward.reset(initial_state)
    self.eventReward.reset(initial_state)

  # // TODO
  # We could add a possible reward(s) as follows:
  # 1) Agent should learn to use boost properly
  # 2) Agent should learn to use dodge properly
  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

    ballTouched = self.ballTouchedByPlayer.get_reward(player, state, previous_action)                 # Get reward for touching the ball
    velocityPlayerToBall = self.velocityPlayerToBallReward.get_reward(player, state, previous_action) # Get reward for velocity of player to ball
    naiveSpeed = self.naiveSpeedReward.get_reward(player, state, previous_action)                     # Get reward for naive speed
    event = self.eventReward.get_reward(player, state, previous_action)                               # Get reward for events

    # Return the sum of the weighted rewards
    reward = ballTouched * self.rewardWeights["ball_touched"] + \
             velocityPlayerToBall * self.rewardWeights["velocity_player_to_ball"] + \
             naiveSpeed * self.rewardWeights["naive_speed"] + \
             event * self.rewardWeights["event"]
    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    # Final reward is given when ball is outside of the radius or when the timeout is reached
    # ==================================
    # Team:
    #   - 0 -> Blue team
    #   - 1 -> Orange team
    # Field:
    #   - Positive coord -> Orange field
    #   - Negative coord -> Blue field
    # If the ball is being sent towards the opposite field, give a positive reward
    # else, give a negative reward 
    # ==================================

    reward = 0.0
    reward += self.ballTouchedByPlayer.get_final_reward(player, state, previous_action) * self.rewardWeights["ball_touched"]
    reward += self.velocityPlayerToBallReward.get_final_reward(player, state, previous_action) * self.rewardWeights["velocity_player_to_ball"]
    reward += self.naiveSpeedReward.get_final_reward(player, state, previous_action) * self.rewardWeights["naive_speed"]
    reward += self.eventReward.get_final_reward(player, state, previous_action) * self.rewardWeights["event"]

    # We calculate the final reward adding the maximum reward per tick divided by the discount factor
    if (player.team_num == 0 and state.ball.position[1] > 0) or (player.team_num == 1 and state.ball.position[1] < 0):
        reward += self.finalUpperBound # Ball is being sent towards the opponent field
    else:
        reward -= self.finalUpperBound # Ball is being sent towards our field or is in the center of the field without any player touching it

    return reward