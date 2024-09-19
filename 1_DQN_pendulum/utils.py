import matplotlib.pyplot as plt
import os
import numpy as np

from datetime import datetime

def plot_rewards(reward_history, epsilon_history, max_episodes, agent, plot, save_plot):
    average_reward = []
    for idx in range(len(reward_history)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = reward_history[:idx + 1]
        else:
            avg_list = reward_history[idx - 49:idx + 1]
        average_reward.append(np.average(avg_list))

    plt.figure(figsize=(25, 15))


    # First subplot for reward history
    plt.subplot(2, 1, 1)  # (2 rows, 1 column, first subplot)

    plt.plot(reward_history, label='Reward')
    plt.plot(average_reward, label='Average Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward ')

    plt.axhline(y=max_episodes, color='red', linestyle='--', label=f'Max reward (y={max_episodes})')
    plt.title(f'{agent.agent_name}_Total Reward/steps per episode. Agent level {agent.n_games}. Final epsilon = {agent.epsilon:.2f}')
    plt.legend()

    # Second subplot for epsilon history
    plt.subplot(2, 1, 2)  # (2 rows, 1 column, second subplot)
    plt.plot(epsilon_history, label='Epsilon', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Over Time')
    plt.legend()

    if save_plot:
        current_date = datetime.now().strftime("%Y%m%d_%H-%M")
        file_name = f"plots_DQN/{current_date}_rewards_{agent.agent_name}_level{agent.n_games}.png"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(file_name)

    if plot:
        plt.show()

def plot_actions(action_history, agent, plot, save_plot):
    plt.figure(figsize=(25, 15))

    plt.plot(action_history)

    plt.xlabel('Episode')
    plt.ylabel('Action ')
    plt.axhline(y=-2, color='red', linestyle='--')
    plt.axhline(y=2, color='red', linestyle='--')
    plt.title(f'{agent.agent_name}_Action(torque) evolution. Agent level {agent.n_games}')

    if save_plot:
        current_date = datetime.now().strftime("%Y%m%d_%H-%M")
        file_name = f"plots/{current_date}_actions_{agent.agent_name}_level{agent.n_games}.png"

        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_name)
    if plot:
        plt.show()

def plot_state_evolution(state_evolution, max_ep_steps, name):
    print('Plotting state evolution ...') # todo theta_dot

    eps, steps, states = zip(*state_evolution) # Unzip to tuple
    states = np.array(states)

    new_episode = np.array(eps) * max_ep_steps
    steps = np.array(eps) * max_ep_steps + np.array(steps)

    theta = np.degrees(np.arccos(states[:, 0]))
    theta_dot = np.degrees(states[2])

    plt.figure(figsize=(50, 15))

    plt.plot(steps, theta, label='Theta (rad)', marker='o')
    plt.vlines(x=new_episode, ymin=0, ymax=180, colors='r', linestyle='--', label='New Episode')

    plt.xlabel('Steps')
    plt.ylabel('Theta (radians)')
    plt.title('Evolution of Theta Over Steps')

    # Get the current date and time in the desired format (e.g., YYYY-MM-DD_HH-MM-SS)
    current_date = datetime.now().strftime("%Y%m%d_%H-%M")
    file_name = f"data/{current_date}_state_evolution_{name}.png"

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


    plt.legend()
    plt.grid(True)
    plt.show()