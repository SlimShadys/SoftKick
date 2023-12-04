import numpy as np

from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState

class Logger(MetricsLogger):
    def __init__(self):
        self.blue_score = 0
        self.orange_score = 0
        self.logger_steps = 0
        
    def _collect_metrics(self, game_state: GameState) -> list:
        ball_stats = np.array([
            # Ball speed
            np.linalg.norm(game_state.ball.linear_velocity),
            # Ball height
            game_state.ball.position[2]
        ])
        
        p_stats = np.zeros(6)
        for p in game_state.players:
            p_stats += np.array([
                # Car speed
                np.linalg.norm(p.car_data.linear_velocity),
                # Car height
                p.car_data.position[2],
                # Boost held
                float(p.boost_amount),
                # On ground
                float(p.on_ground),
                # Ball touch
                float(p.ball_touched),
                # Is demoed
                float(p.is_demoed),
            ])
        p_stats /= len(game_state.players)

        return np.concatenate([ball_stats, p_stats])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        step_diff = cumulative_timesteps - self.logger_steps
        self.logger_steps = cumulative_timesteps

        metrics = np.array(collected_metrics)

        mean_metrics = metrics[:, :5, :].mean(axis=0).mean(axis=1)
        rate_metrics = metrics[:, 5:, :].sum(axis=0).sum(axis=1) / step_diff

        report = { # Mean metrics
                  "ball_speed": mean_metrics[0],
                  "ball_height": mean_metrics[1],
                  "car_speed": mean_metrics[2],
                  "car_height": mean_metrics[3],
                  "boost_held": mean_metrics[4],
                   
                   # Rate metrics
                  "on_ground": rate_metrics[0],
                  "touch_rate": rate_metrics[1],
                  "demoed_rate": rate_metrics[2],

                  "Cumulative Timesteps":cumulative_timesteps
                }
        wandb_run.log(report)
