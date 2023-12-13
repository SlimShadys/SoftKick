import numpy as np
from tqdm import tqdm

TICK_SKIP = 8
HALF_LIFE_SECONDS = 5

fps = 120 / TICK_SKIP
gamma = np.exp(np.log(0.5) / (fps * HALF_LIFE_SECONDS))

def process_player_data(file_path, player_name, episodesBackInTime = 20):
    rewards_episodes = []
    current_episode = []

    with open(file_path, "r") as f:
        for line in tqdm(f, desc=f"Reading data from {player_name} player"):
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

    return total_avg_player

total_avg_first_player = process_player_data("dataFirstPlayer.txt", "first", episodesBackInTime = 20)
total_avg_second_player = process_player_data("dataSecondPlayer.txt", "second", episodesBackInTime =  20)

print("-------------------------")
print(F"Total avg for first player: {total_avg_first_player}")
print(F"Total avg for second player: {total_avg_second_player}")
print("-------------------------")
print(F"Final mean: {(total_avg_first_player + total_avg_second_player) / 2}")
