import numpy as np
import rlgym

from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym_ppo import Learner

from reward import CustomReward
from termination import KickoffTerminalCondition

def makeTestEnvironment():

    # For directly having ticks
    game_tick_rate = 120
    tick_skip = 8
    fps = game_tick_rate / tick_skip

    # RLGym match settings
    spawn_opponents = True
    team_size = 1
    action_parser = DiscreteAction()
    terminal_conditions = KickoffTerminalCondition(fps=fps)
    reward_fn = CustomReward()
    obs_builder = DefaultObs()
    state_setter = DefaultState()

    env = rlgym.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter,
                         game_speed=1)
    return env

if __name__ == "__main__":

    # RLGym-PPO gamma calculation
    game_tick_rate = 120
    tick_skip = 8
    fps = game_tick_rate / tick_skip
    half_life_seconds = 5
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # calculating discount

    # 1 process just for testing
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(makeTestEnvironment,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      exp_buffer_size=150_000,
                      ts_per_iteration=50_000,
                      ppo_batch_size=50_000,
                      ppo_minibatch_size=None,
                      policy_layer_sizes=(1024, 512, 512, 512),
                      critic_layer_sizes=(1024, 512, 512, 512),
                      ppo_epochs=1,
                      ppo_ent_coef=0.0001,
                      gae_gamma=gamma,
                      policy_lr=3e-4,
                      critic_lr=2.5e-4,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      load_wandb=False,
                      checkpoint_load_folder=None,
                      log_to_wandb=False)
    
    learner.learn()
