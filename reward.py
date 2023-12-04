from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward # allows to set reward for each event occurring
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward # gives the agent a reward for its velocity in the direction of the ball
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward

import numpy as np

def distance(x: np.array, y: np.array) -> float:
    return np.linalg.norm(x - y)

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
        "ball_touched": 1.0,
        "velocity_player_to_ball": 2.0,
        "kick_off_distance": 1.0,
        "event": 1.0}

    self.kickOffDistance = KickOffDistance()
    self.ballTouchedByPlayer = TouchBallReward()
    self.velocityPlayerToBallReward = VelocityPlayerToBallReward()
    self.eventReward = EventReward(
      touch = 2.0,
      boost_pickup = 1.0,
    )

  def reset(self, initial_state: GameState):
    self.eventReward.reset(initial_state)
    self.velocityPlayerToBallReward.reset(initial_state)
    self.ballTouchedByPlayer.reset(initial_state)
    self.kickOffDistance.reset(initial_state)

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    event = self.eventReward.get_reward(player, state, previous_action) # Get reward for events
    velocityPlayerToBall = self.velocityPlayerToBallReward.get_reward(player, state, previous_action) # Get reward for velocity of player to ball
    ballTouched = self.ballTouchedByPlayer.get_reward(player, state, previous_action) # Get reward for touching the ball

    if state.ball.position[0] == 0 and state.ball.position[1] == 0:
      if(ballTouched == 1.0):
        firstTouch = 1.0
      else:
        firstTouch = 0.0

    kickOffDistance = self.kickOffDistance.get_reward(player, state, previous_action)

    # Return the sum of the weighted rewards
    reward = event * self.rewardWeights["event"] + \
             ballTouched * self.rewardWeights["ball_touched"] + \
             velocityPlayerToBall * self.rewardWeights["velocity_player_to_ball"] + \
             kickOffDistance * self.rewardWeights["kick_off_distance"] + \
             firstTouch * 7.0
    
    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
         # Maybe we shouldn't return 0 when the episode ends.
         # Returning a positive reward if the agent is the first one to touch the ball?
        return 0.0
