import numpy as np
import rlgym_sim
from rlgym_ppo import Learner
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.state_setters import \
    DefaultState  # state at which each match starts (score = 0-0, time = 0:00, etc.)

from reward import CustomReward
from termination import CustomTerminalCondition


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        #return
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        # report = {"Cumulative Timesteps": cumulative_timesteps}
        # wandb_run.log(report)

        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)

if __name__ == "__main__":
    metrics_logger = ExampleLogger()

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    fps = game_tick_rate / tick_skip
    half_life_seconds = 5
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # calculating discount

    action_parser = DiscreteAction()

    terminal_conditions = CustomTerminalCondition(game_tick_rate / tick_skip)
    reward_fn = CustomReward()
    state_setter = DefaultState()
    obs_builder = DefaultObs()

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter,)

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(env=env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      exp_buffer_size=150_000,
                      ts_per_iteration=50_000,
                      ppo_batch_size=50_000,
                      ppo_minibatch_size=50_000,
                      policy_layer_sizes=(512, 512, 256, 256, 256),
                      critic_layer_sizes=(512, 512, 256, 256, 256),
                      ppo_epochs=1,
                      ppo_ent_coef=0.0001,
                      gae_gamma=0.996,
                      policy_lr=0.001,
                      critic_lr=0.001,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True)
    
    learner.learn()
