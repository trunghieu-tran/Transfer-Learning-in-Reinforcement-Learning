import os
import math
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


def get_avg_std(list_of_lists):
    # Calculate average list
    avg_list = None
    for a_list in list_of_lists:
        if avg_list is None:
            avg_list = a_list
        else:
            avg_list = [(x + y) for (x, y) in zip(avg_list, a_list)]
    num_lists = len(list_of_lists)
    avg_list = [(x / num_lists) for x in avg_list]

    if num_lists == 1:
        std_list = [0.0 for x in range(len(avg_list))]
        return avg_list, std_list

    # Calculate sample standard dev. list
    std_list = None
    for a_list in list_of_lists:
        to_add_list = [(x - y) * (x - y) for (x, y) in zip(a_list, avg_list)]
        if std_list is None:
            std_list = to_add_list
        else:
            std_list = [(x + y) for (x, y) in zip(to_add_list, std_list)]
    std_list = [math.sqrt(x / (num_lists - 1)) for x in std_list]

    return avg_list, std_list



def loading_all_exp_result_from_directory(dir, running_time_num=1):
    x = []
    y = []
    for i in range(running_time_num):
        new_dir = dir + str(i) + "/"
        xx, yy = ts2xy(load_results(new_dir), 'episodes')
        x = xx
        y.append(yy)

    return x, y


def extract_xy_for_plotting(dir, running_time_num, moving_window):
    x, y = loading_all_exp_result_from_directory(dir, running_time_num)
    y_ave, y_std = get_avg_std(y)

    y_ave = moving_average(y_ave, window=moving_window)
    y_std = moving_average(y_std, window=moving_window)

    lower_list = [(x - y) for (x, y) in zip(y_ave, y_std)]
    upper_list = [(x + y) for (x, y) in zip(y_ave, y_std)]

    y_plot = y_ave
    x_plot = x[len(x) - len(y_plot):]

    return x_plot, y_plot, lower_list, upper_list

def plot_multiple_results_with_multiple_runing_time(log_dir_w_TL,
                                                    log_dir_w_TL_rs,
                                                    log_dir_wo_TL,
                                                    title='Learning Curve', moving_window=-1,
                                                    running_time_num=1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # With TL
    x_w_TL,y_w_TL,w_TL_lower_list, w_TL_upper_list = extract_xy_for_plotting(log_dir_w_TL, running_time_num, moving_window)


    # With TL and reward shaping
    x_w_TL_rs, y_w_TL_rs, w_TL_rs_lower_list, w_TL_rs_upper_list = extract_xy_for_plotting(log_dir_w_TL_rs, running_time_num,
                                                                               moving_window)

    # Without TL
    x_wo_TL, y_wo_TL, wo_TL_lower_list, wo_TL_upper_list = extract_xy_for_plotting(log_dir_wo_TL, running_time_num,
                                                                               moving_window)

    plt.figure(title)

    plt.plot(x_wo_TL, y_wo_TL, marker='x', markersize=8, linestyle='-', color='b', label='Without TL', linewidth=3)
    plt.fill_between(x_wo_TL, wo_TL_lower_list, wo_TL_upper_list, color='lightblue', alpha=0.2)

    plt.plot(x_w_TL, y_w_TL, marker='o', markersize=8, linestyle='-', color='g', label='With TL',  linewidth=3)
    plt.fill_between(x_w_TL, w_TL_lower_list, w_TL_upper_list, color='lightgreen', alpha=0.2)

    plt.plot(x_w_TL_rs, y_w_TL_rs, marker='s', markersize=8, linestyle='-', color='r', label='With TL rs', linewidth=3)
    plt.fill_between(x_w_TL_rs, w_TL_rs_lower_list, w_TL_rs_upper_list, color='lightcoral', alpha=0.2)

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