from environment import EnvironmentClass
from agent import GD_Agent

'''
Notes: 
action_space = Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32) ==> continuous 
DQN is designed for discrete action spaces, so you’ll need to adapt your algorithm to handle the continuous action space. 
Reward
The reward is based on how closely the pendulum aligns with the upright position, with penalties for deviation. 
The reward is negative and is a function of the angle and speed of the pendulum:
reward = -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)

https://towardsdatascience.com/reinforcement-learning-explained-visually-part-6-policy-gradients-step-by-step-f9f448e73754
'''

MAX_EPISODE_STEPS = 200  # Default should be 500, not working


def main():
    # ------------------ ENVIRONMENT  ------------------ #
    env_class = EnvironmentClass(env_id='Pendulum-v1',
                                 render_mode='rgb_array',  # 'human ' or 'rgb_array'
                                 max_ep_steps=MAX_EPISODE_STEPS)

    # ------------------ AGENTS  ------------------ #
    agents = {}
    for layers in [[256,256], [256,128]]:  # first for shared, second for each
        agents[f"model_{layers}"] = GD_Agent(env_class=env_class,
                                             lr=0.001,
                                             gamma=0.99,
                                             hidden_layers=layers)

    # # ------------------ TRAINING  ------------------ #
    for agent in agents.values():
        env_class.run_env(agent,
                          n_episodes=1200,
                          max_ep_steps=MAX_EPISODE_STEPS,
                          plot_eps_inf_every=50,
                          plot=False,
                          save_plot=True,
                          note='None')

    env_class.close()


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
