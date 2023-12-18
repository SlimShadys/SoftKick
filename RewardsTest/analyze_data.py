import numpy as np
import sys
import pathlib
import os

TICK_SKIP = 8
HALF_LIFE_SECONDS = 5

fps = 120 / TICK_SKIP
gamma = np.exp(np.log(0.5) / (fps * HALF_LIFE_SECONDS))

_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(_path)

def process_player_data(file_path, episodesBackInTime = 20):
    rewards_episodes = []
    current_episode = []
    
    full_path = str(_path) + "/" + file_path

    if(os.path.exists(full_path) == False):
        print(F"File '{file_path}' does not exist. Exiting...")
        exit()

    with open(full_path, "r") as f:
        for line in f:
            if line == "DONE\n":
                rewards_episodes.append(current_episode[-episodesBackInTime:-1])
                current_episode = []
            else:
                current_episode.append(float(line))

    returns_episodes = []
    current_episode = []

    for episode in rewards_episodes:
        returns = 0
        for reward in reversed(episode):
            current_episode.append(returns * gamma + reward)
        returns_episodes.append(list(reversed(current_episode)))

    episode_avgs = [np.mean(episode) for episode in returns_episodes]
    total_avg_player = np.mean(episode_avgs)

    if(file_path == 'dataFirstPlayer.txt'):
        print(F"Number of episodes: {len(episode_avgs)}")

    return total_avg_player

episodesBackInTime = 10
total_avg_first_player = process_player_data("dataFirstPlayer.txt", episodesBackInTime)
total_avg_second_player = process_player_data("dataSecondPlayer.txt", episodesBackInTime)

print("-------------------------")
print(F"Episodes back in time: {str(episodesBackInTime)}")
print(F"Total avg for first player: {total_avg_first_player}")
print(F"Total avg for second player: {total_avg_second_player}")
print("-------------------------")
print(F"Final mean: {(total_avg_first_player + total_avg_second_player) / 2}")
