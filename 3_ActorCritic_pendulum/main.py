from agent import AC_Agent
from environment import EnvironmentClass

# -----------------------------
'''
Notes: 
action_space = Discrete(2)  # Example: action = 0 (left), action = 1 (right)
Reward = steps, binary, where the agent receives +1 for every step it balances the pole.

# To change thresold: /Users/pedrorodriguezdeledesmajimenez/anaconda3/envs/general/lib/python3.11/site-packages/gym/envs/classic_control/cartpole.py
'''

# TODO Truncation not working  - there is not truncation in this environment
# todo visualize state sapce evolution
# todo visualize value function convergence
# todo some simulations with different hyperparameters gamma, space action discretization, epsilon decay, hidden layers
# TODO normalize rewards

# -----------------------------
MAX_EPISODE_STEPS = 200  # Default should be 500


def main():
    # ------------------ ENVIRONMENT  ------------------ #
    env_class = EnvironmentClass(env_id='Pendulum-v1',
                                 render_mode='rgb_array',  # 'human ' or 'rgb_array', 'ansi'
                                 max_ep_steps=MAX_EPISODE_STEPS)

    # ------------------ AGENTS  ------------------ #
    agents = {}
    for layers in [[128, 128], [256, 256]]:  # 2 layers, one shared, and other for each parameter. [128, 256]
        agents[f"model_{layers}"] = AC_Agent(env_class=env_class,
                                             lr=0.001,
                                             gamma=0.99,
                                             hidden_layers=layers) # todo Divide between hidden layers for mu and std

    # ------------------ TRAINING  ------------------ #
    for agent in agents.values():
        env_class.run_env(agent,
                          n_episodes=7000,
                          max_ep_steps=MAX_EPISODE_STEPS,
                          plot_eps_inf_every=10,
                          plot=False,
                          save_plot=True)

    env_class.close()


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
