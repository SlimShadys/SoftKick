import rlgym

from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.state_setters import DefaultState

from reward import CustomReward
from termination import KickoffTerminalCondition
from rlgym_ppo import Learner

def makeTestEnvironment():
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8

    action_parser = DiscreteAction()

    terminal_conditions = KickoffTerminalCondition(game_tick_rate / tick_skip)

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
                      gae_gamma=0.996,
                      policy_lr=0.001,
                      critic_lr=0.001,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      load_wandb=True,
                      checkpoint_load_folder="C:/Users/gianm/Documents/Università-Git/SoftKick/data/checkpoints/rlgym-ppo-run-1701722416949958700/22359056",
                      log_to_wandb=False)
    
    learner.learn()
