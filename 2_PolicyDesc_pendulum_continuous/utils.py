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

def plot_actions(action_history, name, n_games, plot, save_plot):
    average_action = []
    for idx in range(len(average_action)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = average_action[:idx + 1]
        else:
            avg_list = average_action[idx - 49:idx + 1]
        average_action.append(np.average(avg_list))

    plt.figure(figsize=(25, 15))

    plt.plot(action_history)

    plt.xlabel('Episode')
    plt.ylabel('Action ')
    plt.axhline(y=-2, color='red', linestyle='--')
    plt.axhline(y=2, color='red', linestyle='--')
    plt.title(f'{name}_Action(torque) evolution. Agent level {n_games}')

    if save_plot:
        current_date = datetime.now().strftime("%Y%m%d_%H-%M")
        file_name = f"plots/{current_date}_actions_{name}_level{n_games}.png"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_name)
    if plot:
        plt.show()


def plot_state_evolution(state_evolution, max_ep_steps, name, save_plot):
    print('Plotting state evolution ...')  # todo theta_dot

    eps, steps, states = zip(*state_evolution)  # Unzip to tuple
    states = np.array(states)

    new_episode = np.array(eps) * max_ep_steps
    steps = np.array(eps) * max_ep_steps + np.array(steps)

    theta = np.degrees(np.arccos(states[:, 0]))
    theta_dot = np.degrees(states[2])

    plt.figure(figsize=(50, 15))

    plt.plot(steps, theta, label='Theta (rad)', color='b', linestyle='-', marker=None)
    plt.vlines(x=new_episode, ymin=0, ymax=180, colors='r', linestyle='--', label='New Episode')

    plt.xlabel('Steps')
    plt.ylabel('Theta (radians)')
    plt.title(f'{name}_Evolution of Theta Over Steps')

    if save_plot:
        current_date = datetime.now().strftime("%Y%m%d_%H-%M")
        file_name = f"plots/{current_date}_state_evolution_{name}.png"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(file_name)

    plt.legend()
    plt.grid(True)
    plt.show()
