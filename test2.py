'''
Note: More sophisticated methods like n-step A2C or PPO typically yield better performance.
 This code, however, demonstrates the core idea of an on-policy Actor-Critic update step.
'''

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --------------------
# Hyperparameters
# --------------------
GAMMA = 0.99  # Discount factor
LR = 1e-3  # Learning rate
EPISODES = 2000  # Number of episodes
MAX_STEPS = 200  # Max steps per episode
HIDDEN_DIM = 128  # Number of hidden units
RENDER = False  # Whether to render the environment


# --------------------
# Actor-Critic Network
# --------------------
class ActorCritic(nn.Module):
    """
    A single network that has a shared backbone and two "heads":
    1) Policy head (actor) -> outputs mean and log_std for a Normal action distribution
    2) Value head (critic) -> outputs a single scalar for V(s)
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared backbone
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        # We'll keep log_std as a learnable parameter vector
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the shared trunk, then produce:
          - mean and log_std for the policy
          - state-value
        Returns: (dist, value)
        where dist is a Normal distribution object.
        """
        # Shared layers
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))

        # Actor: mean and log_std
        mean = self.policy_mean(x)
        std = torch.exp(self.log_std)  # shape: [action_dim]
        dist = Normal(mean, std)

        # Critic: value estimate
        value = self.value_head(x)  # shape: [batch_size, 1]

        return dist, value


# --------------------
# Actor-Critic Agent
# --------------------
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        self.ac = ActorCritic(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state):
        """
        Given a state, this returns:
         - action (as a numpy array, for the environment)
         - log probability of the selected action
         - predicted state-value (V(s))
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # shape [1, state_dim]
        dist, value = self.ac(state_t)

        # Sample an action from the policy distribution
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum across action dimensions if >1

        return action.detach().numpy()[0], log_prob, value

    def update(self, log_prob, value, reward, next_value, done):
        """
        Perform a single-step actor-critic update:

        advantage = r + gamma * V(s_{t+1}) - V(s_t)
        actor_loss = -log_pi(a_t|s_t) * advantage
        critic_loss = advantage^2

        total_loss = actor_loss + critic_loss
        """
        # If the episode ended, there's no next_value to bootstrap from
        # We multiply next_value by (1 - done) so that if done = True, next_value=0
        td_target = reward + GAMMA * next_value * (1 - done)
        advantage = td_target - value

        # Actor loss (negative because we do gradient descent on this)
        actor_loss = -log_prob * advantage.detach()

        # Critic loss (MSE)
        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --------------------
# Main Training Loop
# --------------------
def main():
    env = gym.make('Pendulum-v1')
    torch.manual_seed(42)
    np.random.seed(42)

    state_dim = env.observation_space.shape[0]  # e.g. 3: [cos(theta), sin(theta), theta_dot]
    action_dim = env.action_space.shape[0]  # e.g. 1 (torque)

    agent = ActorCriticAgent(state_dim, action_dim, hidden_dim=HIDDEN_DIM, lr=LR)

    reward_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            if RENDER:
                env.render()

            # Select action (and get log prob + value)
            action, log_prob, value = agent.select_action(state)

            # Step in the environment
            next_state, reward, done, _ , _= env.step(action)
            episode_reward += reward

            # Estimate value of the next state (for bootstrapping)
            with torch.no_grad():
                _, next_value = agent.ac(torch.FloatTensor(next_state).unsqueeze(0))

            # Convert reward and done to Torch
            reward_t = torch.FloatTensor([reward])
            done_t = torch.FloatTensor([float(done)])

            # Update the actor-critic
            agent.update(log_prob, value, reward_t, next_value, done_t)

            state = next_state
            if done:
                break

        reward_history.append(episode_reward)

        # Print intermediate results
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode + 1}/{EPISODES}, Average Reward (last 10): {avg_reward:.2f}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
