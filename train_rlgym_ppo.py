import numpy as np

#import rlgym
import rlgym_sim as rlgym

from rlgym_ppo import Learner
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState  # state at which each match starts (score = 0-0, time = 0:00, etc.)

from reward import CustomReward
from termination import KickoffTerminalCondition
from logger import Logger

def makeEnvironment():
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

    terminal_conditions = KickoffTerminalCondition(game_tick_rate / tick_skip)
    reward_fn = CustomReward()
    state_setter = DefaultState()
    obs_builder = DefaultObs()

    env = rlgym.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter,)
    return env

if __name__ == "__main__":
    metrics_logger = Logger()

    # 40 processes
    n_proc = 40

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(makeEnvironment,
                      n_proc=n_proc,
                      ppo_epochs=1,
                      ppo_batch_size=50_000,
                      ppo_minibatch_size=None,
                      exp_buffer_size=150_000,
                      ts_per_iteration=50_000,
                      min_inference_size=min_inference_size,
                      policy_layer_sizes=(1024, 512, 512, 512),
                      critic_layer_sizes=(1024, 512, 512, 512),
                      ppo_ent_coef=0.0001,
                      gae_gamma=0.996,
                      policy_lr=0.001,
                      critic_lr=0.001,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      metrics_logger=metrics_logger,
                      wandb_project_name="rlgym-ppo",
                      wandb_group_name="v0.2",
                      log_to_wandb=True)
    
    learner.learn()
