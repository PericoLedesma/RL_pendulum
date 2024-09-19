import matplotlib.pyplot as plt
import os
import numpy as np

from datetime import datetime

def plot_rewards(reward_history, max_episodes,agent,plot, save_plot):

    average_reward = []
    for idx in range(len(reward_history)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = reward_history[:idx + 1]
        else:
            avg_list = reward_history[idx - 49:idx + 1]
        average_reward.append(np.average(avg_list))

    plt.figure(figsize=(25, 15))

    plt.plot(reward_history)
    plt.plot(average_reward)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward ')
    plt.axhline(y=0, color='red', linestyle='--', label=f'Max reward (y=0)')
    plt.title(f'{agent.agent_name}_Total Reward/steps per episode. Agent level {agent.n_games}.')

    if save_plot:
        current_date = datetime.now().strftime("%Y%m%d_%H-%M")
        file_name = f"plots/{current_date}_rewards_{agent.agent_name}_level{agent.n_games}.png"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_name)
    if plot:
        plt.show()