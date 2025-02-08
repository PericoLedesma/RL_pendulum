import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --------------------
# Hyperparameters
# --------------------
LR_POLICY = 1e-3
LR_VALUE = 3e-3
GAMMA = 0.99
EPISODES = 2000
MAX_STEPS = 200
HIDDEN_DIM = 128

RENDER_ENV = False


# --------------------
# Policy Network (Actor)
# --------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)

        # Log std parameter (could also be produced by a network head)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc_mean(x)
        return mean, self.log_std


# --------------------
# Value Network (Critic)
# --------------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc_out(x)
        return value


# --------------------
# A2C Agent
# --------------------
class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # Actor (Policy) and Critic (Value Function)
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_POLICY)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR_VALUE)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)  # shape: [1, state_dim]
        mean, log_std = self.policy(state_t)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy()[0], log_prob, mean, std

    def compute_returns(self, rewards, dones, values, next_value):
        """
        Compute discounted returns (bootstrapping from next_value if not done).
        """
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + GAMMA * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        return returns, values

    def update(self, log_probs, values, returns):
        """
        A2C update step.
        advantage = returns - values
        1) Update policy: maximize log_prob * advantage
        2) Update value: minimize MSE(returns - values)
        """
        advantages = returns - values

        # Policy loss (negative because we do gradient descent)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss (mean squared error)
        value_loss = advantages.pow(2).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


def main():
    env = gym.make("Pendulum-v1")
    np.random.seed(42)
    torch.manual_seed(42)

    state_dim = env.observation_space.shape[0]  # e.g. 3 for Pendulum: [cos(theta), sin(theta), theta_dot]
    action_dim = env.action_space.shape[0]  # e.g. 1 for Pendulum (torque)

    agent = A2CAgent(state_dim, action_dim, hidden_dim=HIDDEN_DIM)
    reward_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()

        log_probs = []
        values = []
        rewards = []
        dones = []

        episode_reward = 0

        for t in range(MAX_STEPS):
            if RENDER_ENV:
                env.render()

            # Select action
            action, log_prob, mean, std = agent.select_action(state)

            # Step in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Store log prob and value for the current state
            value = agent.value(torch.FloatTensor(state).unsqueeze(0))

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor([reward]))
            dones.append(float(done))

            state = next_state
            episode_reward += reward

            if done:
                break

        # For the terminal state (or if we reached max_steps), we bootstrap the value
        if done:
            next_value = torch.zeros(1)  # no bootstrapping if the episode truly ended
        else:
            next_value = agent.value(torch.FloatTensor(next_state).unsqueeze(0)).detach()

        # Compute the returns
        returns, values = agent.compute_returns(rewards, dones, values, next_value)

        # Convert list of log_probs to a tensor
        log_probs = torch.stack(log_probs)

        # Update policy and value function
        agent.update(log_probs, values, returns)

        reward_history.append(episode_reward)

        # Print running average reward
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode + 1}/{EPISODES}, Average Reward (last 10): {avg_reward:.2f}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()