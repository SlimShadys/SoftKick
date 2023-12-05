from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward # allows to set reward for each event occurring
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward # gives the agent a reward for its velocity in the direction of the ball
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
from rlgym_sim.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward

import numpy as np

class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / 2300

class KickOffDistance(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for p in state.players:
            if p.car_id != player.car_id:
              positionOpponent = p.car_data.position
              dist = np.linalg.norm(positionOpponent - state.ball.position)

              if dist < player_dist:
                  return False

        return True

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
          return 1
        else:
          return 0

class CustomReward(RewardFunction):
  def __init__(self) -> None:
    super().__init__()
    self.rewardWeights = {
        "kick_off_distance": 0.025,
        "ball_touched": 1.0,
        "velocity_player_to_ball": 0.05,
        "naive_speed": 0.02,
        "first_touch": 1.0,
        "event": 0.01}

    self.kickOffDistance = KickOffDistance()
    self.ballTouchedByPlayer = TouchBallReward()
    self.velocityPlayerToBallReward = VelocityPlayerToBallReward()
    self.velocityBallToGoalReward = VelocityBallToGoalReward()
    self.naiveSpeedReward = NaiveSpeedReward()
    self.eventReward = EventReward(
      touch = 2.0,
      boost_pickup = 1.0,
      shot = 1.0,
    )

    self.touch_counter = 0

  def pre_step(self, state: GameState):
     if any([p.ball_touched for p in state.players]):
        self.touch_counter += 1
     return

  def reset(self, initial_state: GameState):
    self.kickOffDistance.reset(initial_state)
    self.ballTouchedByPlayer.reset(initial_state)
    self.velocityPlayerToBallReward.reset(initial_state)
    self.naiveSpeedReward.reset(initial_state)
    self.eventReward.reset(initial_state)
    self.velocityBallToGoalReward.reset(initial_state)

    self.touch_counter = 0

  # // TODO
  # We could add a possible reward(s) as follows:
  # 1) Do we need touch EventReward if we have ballTouched reward? We know we might end the game if the ball is outside the radius, so we might not need it.
  # 2) Agent should learn to use boost properly
  # 3) Agent should learn to use dodge properly
  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

    kickOffDistance = self.kickOffDistance.get_reward(player, state, previous_action)                 # Get reward for being the closest player to the ball at kick off
    ballTouched = self.ballTouchedByPlayer.get_reward(player, state, previous_action)                 # Get reward for touching the ball
    velocityPlayerToBall = self.velocityPlayerToBallReward.get_reward(player, state, previous_action) # Get reward for velocity of player to ball
    naiveSpeed = self.naiveSpeedReward.get_reward(player, state, previous_action)                     # Get reward for naive speed
    event = self.eventReward.get_reward(player, state, previous_action)                               # Get reward for events
    firstTouch = 0

    # First touch reward
    if(player.ball_touched and self.touch_counter == 1):
      firstTouch = 1.0

    # Return the sum of the weighted rewards
    reward = kickOffDistance * self.rewardWeights["kick_off_distance"] + \
             ballTouched * self.rewardWeights["ball_touched"] + \
             velocityPlayerToBall * self.rewardWeights["velocity_player_to_ball"] + \
             naiveSpeed * self.rewardWeights["naive_speed"] + \
             event * self.rewardWeights["event"] + \
             firstTouch * self.rewardWeights["first_touch"]
    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    # Final reward is given when ball is outside of the radius or when the timeout is reached
    # ==================================
    # Field:
    #   - Positive coord -> Orange field
    #   - Negative coord -> Blue field
    # If the ball is being sent towards the opposite field, give a reward summed with the velocity of the ball towards the goal (Max: +2.0)
    # else, give a negative reward summed with the velocity of the ball towards the goal (Max: -2.0)
    # ==================================

    # This is automatically positive/negative based on the player team
    reward_factor = self.velocityBallToGoalReward.get_reward(player, state, previous_action)
    if player.team_num == 0: # BLUE TEAM
        return 1.0 + reward_factor if state.ball.position[0] > 0 else -1.0
    else:                    # ORANGE TEAM
        return 1.0 + reward_factor if state.ball.position[0] < 0 else -1.0