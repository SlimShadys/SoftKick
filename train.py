from importlib.metadata import version # Used to get the version of Stable Baselines 3
import numpy as np 
import torch.nn as nn # Used for importing ReLU activation function

from stable_baselines3 import PPO # Importing the Proximal Policy Optimization (PPO) algorithm from Stable Baselines
from stable_baselines3.ppo import MlpPolicy # implements actor critic using a multi-layered perceptron
from stable_baselines3.common.callbacks import CheckpointCallback # Saves the model after each fixed number of steps
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
# vec_env - a method of stacking multiple independent environments into a single environment
# VecMonitor - used to record episode reward, length, time, and other data
# VecNormalize - a moving average, normalizing wrapper for a vectorized environment
# VecCheckNan - raises warnings for when a value is NaN

from rlgym.envs import Match # Keeps track of the settings and values of various instances of matches/games

from rlgym.utils.action_parsers import DiscreteAction # Makes things simpler conceptually and computationally
from rlgym.utils.obs_builders import AdvancedObs # creates an observation builder for the environment
from rlgym.utils.state_setters import DefaultState # state at which each match starts (score = 0-0, time = 0:00, etc.)

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv # allows for multiple instances to be running at the same time

from reward import CustomReward # custom reward function
from termination import CustomTerminalCondition # custom terminal condition

if __name__ == '__main__':  # Required for multiprocessing

    # === RLGym settings ===
    agents_per_match = 2  # 1-on-1s and there is an agent for both teams
    num_instances = 1 # number of instances of rocket league (higher value does not necessarily mean faster training)
    frame_skip = 8          # Number of ticks to repeat an action
    fps = 120 / frame_skip  # Frames per second
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    # === PPO hyperparameters ===
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # calculating discount
    learning_rate = 3e-4  # Around this is fairly common for PPO
    ent_coef = 0.005
    vf_coef = 1.00
    gae_lambda = 0.95
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        # The architecture takes insipration from Lucy-SKG
        # https://arxiv.org/pdf/2305.15801.pdf pag. 15
        net_arch=(
            512, 512, # We will have 2 hidden layers with 512 neurons each | 
            dict(
                pi=[256, 256, 256], # We will have 3 hidden layers with 256 neurons each for the policy network
                vf=[256, 256, 256]) # We will have 3 hidden layers with 256 neurons each for the value network
                ),
        ortho_init=True,)
    
    # === Training ===
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
    batch_size = 50000  # getting the batch size down to something more manageable - 100k in this case
    model_dirs = "models"   # Directory to save models to
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match( # match object with settings/parameters
            team_size=1, # means that the bot will be training with 1-on-1s
            tick_skip=frame_skip,
            reward_function=CustomReward(), # reward function to use
            spawn_opponents=True,  # this means that the agent will be on both teams, rather than our agent vs a rocket league bot
            terminal_conditions=CustomTerminalCondition(fps), # terminal condition to use
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous (less training time)
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)  # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                  # Optional
    env = VecMonitor(env)                                   # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards


    print("Stable Baselines 3 Version:", version('stable-baselines3'))
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=10,                 # PPO calls for multiple epochs
        policy_kwargs=policy_kwargs, # all of the arguments passed to the policy
        learning_rate=learning_rate, # Around this is fairly common for PPO
        ent_coef=ent_coef,              # From PPO Atari
        vf_coef=vf_coef,                  # From PPO Atari
        gamma=gamma,                 # Gamma as calculated using half-life
        verbose=3,                   # Print out all the info as we're going
        batch_size=batch_size,       # Batch size as high as possible within reason
        n_steps=steps,               # Number of steps to perform before optimizing network
        gae_lambda=gae_lambda,             # From PPO Atari
        tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="auto"                # Uses GPU if available
    )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(save_freq=round(1_000_000 / env.num_envs), save_path=model_dirs, name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            print("Training started!")
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
            model.save(F"{model_dirs}/exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency
    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save(F"{model_dirs}/exit_save")
    print("Save complete")