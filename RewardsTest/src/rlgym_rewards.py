from abc import ABC, abstractmethod

import numpy as np

from rlgym_compat import GameState, PlayerData
from rlgym_compat.common_values import BALL_RADIUS, BLUE_TEAM, CAR_MAX_SPEED


class RewardFunction(ABC):
    @abstractmethod
    def reset(self, initial_state: GameState):
        """
        Function to be called each time the environment is reset. This is meant to enable users to design stateful reward
        functions that maintain information about the game throughout an episode to determine a reward.

        :param initial_state: The initial state of the reset environment.
        """
        raise NotImplementedError

    def pre_step(self, state: GameState):
        """
        Function to pre-compute values each step. This function is called only once each step, before get_reward is
        called for each player.

        :param state: The current state of the game.
        """
        pass

    @abstractmethod
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Function to compute the reward for a player. This function is given a player argument, and it is expected that
        the reward returned by this function will be for that player.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: A reward for the player provided.
        """
        raise NotImplementedError

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Function to compute the reward for a player at the final step of an episode. This will be called only once, when
        it is determined that the current state is a terminal one. This may be useful for sparse reward signals that only
        produce a value at the final step of an environment. By default, the regular get_reward is used.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: A reward for the player provided.
        """
        return self.get_reward(player, state, previous_action)


class EventReward(RewardFunction):
    def __init__(
        self,
        goal=0.0,
        team_goal=0.0,
        concede=-0.0,
        touch=0.0,
        shot=0.0,
        save=0.0,
        demo=0.0,
        boost_pickup=0.0,
    ):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array(
            [goal, team_goal, concede, touch, shot, save, demo, boost_pickup]
        )

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array(
            [
                player.match_goals,
                team,
                opponent,
                player.ball_touched,
                player.match_shots,
                player.match_saves,
                player.match_demolishes,
                player.boost_amount,
            ]
        )

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(
                player, initial_state
            )

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.0):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return (
                (state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)
            ) ** self.aerial_weight
        return 0
