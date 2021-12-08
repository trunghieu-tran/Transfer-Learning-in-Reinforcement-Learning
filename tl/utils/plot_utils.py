import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    if window == -1: # average from begining to current position
        moving_averages = []
        i = 0
        s = 0
        while i < len(values):
            s += values[i]
            moving_averages.append(s / (i + 1))
            i += 1
        return moving_averages

    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', moving_window=-1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'episodes')
    y = moving_average(y, window=moving_window)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

def plot_multiple_results(log_dir_w_TL, log_dir_w_TL_rs, log_dir_wo_TL, log_dir_w_full_TL_rs, title='Learning Curve', moving_window=-1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # With TL
    x_w_TL, y_w_TL = ts2xy(load_results(log_dir_w_TL), 'episodes')
    y_w_TL = moving_average(y_w_TL, window=moving_window)
    x_w_TL = x_w_TL[len(x_w_TL) - len(y_w_TL):]

    # With TL and reward shaping
    x_w_TL_rs, y_w_TL_rs = ts2xy(load_results(log_dir_w_TL_rs), 'episodes')
    y_w_TL_rs = moving_average(y_w_TL_rs, window=moving_window)
    x_w_TL_rs = x_w_TL_rs[len(x_w_TL_rs) - len(y_w_TL_rs):]

    # Without TL
    x_wo_TL, y_wo_TL = ts2xy(load_results(log_dir_wo_TL), 'episodes')
    y_wo_TL = moving_average(y_wo_TL, window=moving_window)
    x_wo_TL = x_wo_TL[len(x_wo_TL) - len(y_wo_TL):]

    # With full TL and reward shaping
    x_w_full_TL_rs, y_w_full_TL_rs = ts2xy(load_results(log_dir_w_full_TL_rs), 'episodes')
    y_w_full_TL_rs = moving_average(y_w_full_TL_rs, window=moving_window)
    x_w_full_TL_rs = x_w_full_TL_rs[len(x_w_full_TL_rs) - len(y_w_full_TL_rs):]


    plt.figure(title)

    plt.plot(x_wo_TL, y_wo_TL, marker='x', markersize=8, linestyle='-', color='b', label='Without TL', linewidth=3)
    plt.plot(x_w_TL, y_w_TL, marker='o', markersize=8, linestyle='-', color='g', label='With TL',  linewidth=3)
    plt.plot(x_w_TL_rs, y_w_TL_rs, marker='s', markersize=8, linestyle='-', color='r', label='With TL rs', linewidth=3)
    plt.plot(x_w_full_TL_rs, y_w_full_TL_rs, marker='*', markersize=8, linestyle='-', color='y', label='With full TL rs', linewidth=3)

    plt.legend(loc='upper left', title='Approaches', fontsize=14)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True