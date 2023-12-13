import math
import time
import torch
import pathlib
import sys

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.game_state_util import BallState, CarState, GameInfoState
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import GameState as RLBotGameState
from rlbot.utils.game_state_util import Physics, Rotator, Vector3
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_ppo.ppo import MultiDiscreteFF

from rewards import CustomReward
from rlgym_action_parser import DiscreteAction
from rlgym_compat import GameState as RLGymGameState
from rlgym_compat.common_values import BLUE_GOAL_BACK, ORANGE_GOAL_BACK
from rlgym_obs_builder import DefaultObs
from terminals import KickoffTerminalCondition


class MyBot(BaseAgent):
    def __del__(self):
        if self.fp:
            self.fp.close()
        if self.fp1:
            self.fp1.close()

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.fp = open("dataFirstPlayer.txt", "a")
        self.fp1 = open("dataSecondPlayer.txt", "a")
        self.tick_skip = 8
        self.half_life_seconds = 5
        self.include_final_reward = True

    def initialize_agent(self):
        # Start car in specific position
        # car_state = CarState(
        #     boost_amount=0,
        #     physics=Physics(
        #         location=Vector3(0, 0, 1500),
        #         velocity=Vector3(0, 0, 0),
        #         rotation=Rotator(0, 0, 0),
        #         angular_velocity=Vector3(0, 0, 0),
        #     ),
        # )
        # game_info_state = GameInfoState(world_gravity_z=0.0001)
        # game_state = GameState(cars={self.index: car_state}, game_info=game_info_state)
        # self.set_game_state(game_state)
        self.fps = 120 / self.tick_skip
        self.gamma = np.exp(np.log(0.5) / (self.fps * self.half_life_seconds))
        self.field_info = self.get_field_info()
        self.game_state = RLGymGameState(self.field_info)
        self.ticks = 0
        self.prev_tick = 0
        self.ticks_elapsed_since_update = 0
        self.done = False
        self.prev_action = np.zeros(8)
        self.reward_function = CustomReward(gamma=self.gamma)
        self.terminal_condition = KickoffTerminalCondition(fps=self.fps)
        self.obs_builder = DefaultObs()
        self.action_parser = DiscreteAction()
        self.started = False
        self.checked_kickoff = False
        self.policy = MultiDiscreteFF(89, (1024, 512, 512, 512), "cuda").to("cuda")
        _path = pathlib.Path(__file__).parent.resolve()
        sys.path.append(_path)
        self.policy.load_state_dict(torch.load(str(_path) + "/checkpoint/PPO_POLICY.pt"))
        self.controls = SimpleControllerState()
        self.ticks_since_tried_score = 0

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        cur_tick = packet.game_info.frame_num
        delta = cur_tick - self.prev_tick
        self.prev_tick = cur_tick
        self.ticks += delta
        self.ticks_elapsed_since_update += delta
        self.ticks_since_tried_score += delta

        self.game_state.decode(packet, delta)
        if packet.game_info.is_kickoff_pause and not self.checked_kickoff:
            self.checked_kickoff = True
            self.reward_function.reset(self.game_state)
            self.terminal_condition.reset(self.game_state)
            self.obs_builder.reset(self.game_state)
            self.ticks_elapsed_since_update = 0
            self.inverse_returns = 0
            self.done = False
            self.started = True

        if not packet.game_info.is_kickoff_pause:
            self.checked_kickoff = False

        if self.started:
            if self.ticks_elapsed_since_update >= self.tick_skip and not self.done:
                self.done = self.terminal_condition.is_terminal(self.game_state)
                player = self.game_state.players[self.index]
                self.reward_function.pre_step(self.game_state)

                if self.done:
                    reward = self.reward_function.get_final_reward(player, self.game_state, self.prev_action)
                else:
                    reward = self.reward_function.get_reward(player, self.game_state, self.prev_action)

                # Get observation
                obs = self.obs_builder.build_obs(player, self.game_state, self.prev_action)

                # Get action from Policy
                action_idx, _ = self.policy.get_action(obs)
                action = self.action_parser.parse_actions(action_idx.numpy(), self.game_state) # Parse action
                self.update_controls(action[0]) # Update controls with our action

                self.prev_action = action[0] # a = a'

                self.ticks_elapsed_since_update = 0

                # Write to file
                if(self.index == 0):
                    self.fp.write(f"{reward}\n")
                else:
                    self.fp1.write(f"{reward}\n")

                # If we are done, write DONE to file and we exit from the if-statement
                if self.done:
                    if(self.index == 0):
                        self.fp.write(f"DONE\n")
                    else:
                        self.fp1.write(f"DONE\n")

                # Just flush the buffer to make sure that we don't lose any data
                if(self.index == 0):
                    self.fp.flush()
                else:
                    self.fp1.flush()

            # Only for rendering purposes
            text = []
            if self.done:
                text.append("EPISODE DONE")

            self.renderer.begin_rendering()
            self.renderer.draw_string_2d(
                100, 50, 2, 2, "\n".join(text), self.renderer.white()
            )
            self.renderer.end_rendering()

        # If we are done, we want to reset the episode, so let's score a goal
        if (self.done and self.ticks_since_tried_score > 60 and packet.game_info.is_round_active):
            if(packet.game_ball.physics.location.y > 0):
                goal_pos = ORANGE_GOAL_BACK
            else:
                goal_pos = BLUE_GOAL_BACK
            vel_vec = goal_pos - np.array(
                [
                    packet.game_ball.physics.location.x,
                    packet.game_ball.physics.location.y,
                    packet.game_ball.physics.location.z,
                ]
            )
            vel_vec_normalized = vel_vec / np.linalg.norm(vel_vec)
            vel_vec = vel_vec_normalized * 10000
            self.set_game_state(
                RLBotGameState(
                    ball=BallState(
                        Physics(velocity=Vector3(vel_vec[0], vel_vec[1], vel_vec[2]))
                    )
                )
            )
            self.ticks_since_tried_score = 0

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
