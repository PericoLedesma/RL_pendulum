# Libraries
import time
import gymnasium as gym

# Files
from utils import *

class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, max_ep_steps, plot_eps_inf_every, plot, save_plot) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPOCHS\n')
        start_time = time.perf_counter()

        reward_history, state_evolution = [], []

        try:
            for eps in range(n_episodes):
                eps_data = [] # for storing data from the episode
                score = 0
                truncated, terminated = False, False

                state, _ = self.env.reset()

                while not truncated and not terminated:
                    action = agent.get_action(state)
                    assert -2 <= action <= 2, f"Torque {action} is out of bounds! Expected between -2 and 2."

                    state_, reward, terminated, truncated, info = self.env.step([action.item()])

                    state_evolution.append((eps, state))
                    eps_data.append((eps, state, action, reward))

                    # Next steps loop
                    state = state_
                    score += reward

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

        # agent.save() # todo save periodically to not lose if crashes
        # agent.store_agent_parameters(np.mean(reward_history[-mean_batch:])) # todo store model metadata to keep track experiments. Librry of soft to keep track?
        print('=' * 60)

        # ------------------ PLOTTING RESULTS ------------------ #
        plot_rewards(reward_history, max_ep_steps, agent, plot, save_plot)

    def close(self):
        self.env.close()
        print('\n\n ****** Environment Closed ******\n\n')
