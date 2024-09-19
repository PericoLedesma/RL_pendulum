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
                            render_mode=render_mode)
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, batch_size, max_ep_steps,save_model, plot, save_plot) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPOCHS\n')
        start_time = time.perf_counter()

        reward_history, epsilon_history = [], []
        state_evolution = []
        action_history = []

        try:
            for eps in range(n_episodes):
                truncated = False
                step, score = 0, 0
                state, _ = self.env.reset()

                while not truncated:
                    # message = ('\tStep %d: theta=%0.2f deg |theta_dot=%.2f deg/s |eps=%.2f'
                    #       %(step, np.degrees(np.arccos(state[0])), np.degrees(state[2]),agent.epsilon))

                    # print('\r', message, end='')

                    action = agent.get_action(state)

                    state_, reward, done, truncated, info = self.env.step([action])

                    # Short memory, just this step
                    agent.learning(state, [action], [reward], state_, [truncated])

                    # Store in replay memory
                    # print('Storing in replay memory ... Number of memories ', len(agent.memory) + 1)
                    agent.memory.append((state, action, reward, state_, truncated))

                    state_evolution.append((eps, step, state))
                    action_history.append(action)

                    # Next steps loop
                    state = state_
                    score += reward
                    step += 1


                # At the end of each episode
                agent.n_games += 1
                agent.linear_epsilon_decay(eps, n_episodes)
                epsilon_history.append(agent.epsilon)
                reward_history.append(score)

                message = f"[Epoch {eps + 1}] Reward = {score}| ave_score_50 = {np.mean(reward_history[-50:]):.2f}| Epsilon = {agent.epsilon:.2f}"
                # print('\r', message, end='')
                print(message)

                # Long memory, replay memory
                if len(agent.memory) > batch_size:
                    agent.memory_replay(batch_size)

                # Update target network
                # Optionally update target network periodically
                # if episode % 10 == 0:
                agent.Q_target.load_state_dict(agent.Q_model.state_dict())


        except KeyboardInterrupt:
            print('\n*Training Interrupted', '\n')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print(f'\nTraining Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)

        if save_model:
            # agent.save()
            agent.store_agent_parameters(np.mean(reward_history[-500:]))
        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #
        plot_rewards(reward_history, epsilon_history, max_ep_steps, agent, plot, save_plot)
        plot_actions(action_history, agent, plot, save_plot)
        # plot_state_evolution(state_evolution, max_ep_steps, agent.agent_name, save_plot)

    def close(self):
        self.env.close()
        print('\n ****** Environment Closed ******')
