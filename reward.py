from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, FaceBallReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, SUPERSONIC_THRESHOLD

import numpy as np

class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / CAR_MAX_SPEED

class SustainedVelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.comulative_reward = 0

    def reset(self, initial_state: GameState):
        self.comulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        norm_scalar_vel_towards_ball = scalar_vel_towards_ball / SUPERSONIC_THRESHOLD

        if norm_scalar_vel_towards_ball > 0 and norm_scalar_vel_towards_ball < 1:
            reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) / 50
        else:
            if norm_scalar_vel_towards_ball < 0:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) * 1.5
            else:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED)

        self.comulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("VelocityPlayerToBall: Cumulative: ", self.comulative_reward)
        return 0.0

class CustomReward(RewardFunction):
  def __init__(self, gamma=0.990800) -> None:
    super().__init__()
    self.rewardWeights = {
        "face_ball": 0.02,
        "sustainedVelocity": 0.20,
        "ball_touched": 3.2,
        "velocity_player_to_ball": 0.9,
        "naive_speed": 1.0,
        "event": 0.06}

    self.faceBallReward = FaceBallReward()                                # Returns 1.0 if the player faces the ball         / Max 1.0
    self.ballTouchedByPlayer = TouchBallReward()                          # Returns 1.0 if the player touches the ball       / Max 1.0
    self.velocityPlayerToBallReward = VelocityPlayerToBallReward()        # Returns the velocity of the player to the ball   / Max 1.0
    self.naiveSpeedReward = NaiveSpeedReward()                            # Returns the naive speed of the player            / Max 1.0
    self.sustainedVelocityReward = SustainedVelocityPlayerToBallReward()  # Returns the sustainedVelocity of the player      / Max 1.0
    self.eventReward = EventReward(                                       # Returns a reward for each event                  / Max 1.0 * 1.0 + 1.0 * 2.0 = 3.00
      touch = 2.0,
      boost_pickup = 1.0,
    )

    self.upperBound = 5.50  # Maximum reward per tick
    self.gamma = gamma      # 0.9908006132652293
    self.finalUpperBound = (self.upperBound / (1 - self.gamma)) / 25 # 25 is a magic number that we found to work discretely well.

  def reset(self, initial_state: GameState):
    self.faceBallReward.reset(initial_state)
    self.ballTouchedByPlayer.reset(initial_state)
    self.velocityPlayerToBallReward.reset(initial_state)
    self.naiveSpeedReward.reset(initial_state)
    self.sustainedVelocityReward.reset(initial_state)
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
    sustainedVelocity = self.sustainedVelocityReward.get_reward(player, state, previous_action)       # Get reward for sustainedVelocity
    faceBall = self.faceBallReward.get_reward(player, state, previous_action)                         # Get reward for facing the ball

    # Return the sum of the weighted rewards
    reward = ballTouched * self.rewardWeights["ball_touched"] + \
             velocityPlayerToBall * self.rewardWeights["velocity_player_to_ball"] + \
             naiveSpeed * self.rewardWeights["naive_speed"] + \
             event * self.rewardWeights["event"] + \
             sustainedVelocity * self.rewardWeights["sustainedVelocity"] + \
             faceBall * self.rewardWeights["face_ball"]
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
    reward += self.faceBallReward.get_final_reward(player, state, previous_action) * self.rewardWeights["face_ball"]
    reward += self.sustainedVelocityReward.get_final_reward(player, state, previous_action) * self.rewardWeights["sustainedVelocity"]

    # We calculate the final reward adding the maximum reward per tick divided by the discount factor
    if (player.team_num == 0 and state.ball.position[1] > 0) or (player.team_num == 1 and state.ball.position[1] < 0):
        reward += self.finalUpperBound # Ball is being sent towards the opponent field
    else:
        reward -= self.finalUpperBound # Ball is being sent towards our field

    return reward