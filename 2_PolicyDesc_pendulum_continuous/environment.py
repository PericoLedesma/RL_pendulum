# Libraries
import numpy as np
import gymnasium as gym
import time

# Files
from utils import *


class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)  # 'human ' or 'rgb_array', 'ansi'
        self.max_episode_steps = max_ep_steps
        # print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, max_ep_steps, plot_eps_inf_every, plot, save_plot, note) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPOCHS\n')
        start_time = time.perf_counter()

        reward_history, state_evolution = [], []
        action_history = []

        try:
            for eps in range(n_episodes):
                eps_data = []

                truncated, done = False, False
                step, score = 0, 0

                state, _ = self.env.reset()

                while not truncated or not done:
                    action = agent.get_action(state)

                    state_, reward, terminated, truncated, info = self.env.step([action])

                    state_evolution.append((eps, step, state))
                    eps_data.append((eps, state, action, reward))
                    action_history.append(action)

                    state = state_
                    score += reward
                    step += 1

                    if self.max_episode_steps < step:
                        done = True

                # At the end of each episode
                agent.n_games += 1
                reward_history.append(score)

                if eps % plot_eps_inf_every == 0:
                    print(f"[Epoch {eps}] Reward = {score:.0f}| avg_score_100={np.mean(reward_history[-100:]):.2f}| avg_score_50={np.mean(reward_history[-50:]):.2f}")

                # Policy descent
                agent.policy_update(eps_data)


        except KeyboardInterrupt:
            print('\n*Training Interrupted', '\n')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print(f'\nTraining Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)
        agent.store_agent_parameters(np.mean(reward_history[-500:]), np.mean(reward_history[-100:]), note)

        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #
        plot_rewards(reward_history, max_ep_steps, agent, plot, save_plot)
        plot_actions(action_history, agent.agent_name, agent.n_games, plot, save_plot)
        # plot_state_evolution(state_evolution, max_ep_steps, agent.agent_name, save_plot)

    def close(self):
        self.env.close()
        print('\n ****** Environment Closed ******')
