import numpy as np
import json
from collections import deque
import torch as T  # PyTorch library for ML and DL
import torch.nn.functional as F  # PyTorch's functional module
from torch.distributions import Categorical


from model import PolicyNetwork, ValueNetwork
from utils import *

# -----------------------------
MAX_MEMORY = 100_000
METADATA_FILE = 'data/agent_metadata'


class AC_Agent:
    def __init__(self, lr, gamma, env_class, hidden_layers):
        # MODEL NAME
        self.agent_name = "model"
        for hidden_layer in hidden_layers:
            self.agent_name += f"_{hidden_layer}"

        print(f'\n ****** Creating Agent {self.agent_name}... ******')
        print(f'\tInput/Observation = {env_class.env.observation_space.shape} | Output/action = {env_class.env.action_space.shape[0]}')
        print(f'\tAction space: {env_class.env.action_space.low[0]}-{env_class.env.action_space.high[0]}')

        self.hidden_layers = hidden_layers

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.n_games = 0

        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.gpu_use()

        # Actor network (Policy)
        self.ActorNet = PolicyNetwork('Actor',
                                env_class.env.observation_space.shape[0],
                                hidden_layers,
                                self.lr).to(self.device)

        # Critic network (Value)
        self.CriticNet = ValueNetwork('Critic',
                                 env_class.env.observation_space.shape[0],
                                 hidden_layers,
                                 self.lr).to(self.device)

        #self.action_space = T.tensor(self.action_space, dtype=T.float).to(self.device)

    def get_action(self, observation: np.ndarray) -> int:
        with T.no_grad():
            n_distrib = self.ActorNet(T.tensor(observation, dtype=T.float).to(self.device))
            action = n_distrib.sample().item()
            return np.clip(action, -2., 2.)


    def policy_update(self, eps_data: list) -> None:
        eps, eps_states, eps_actions, eps_rewards = zip(*eps_data)
        eps_states = T.tensor(np.array(eps_states), dtype=T.float).to(self.device)
        eps_actions = T.tensor(np.array(eps_actions), dtype=T.int).to(self.device)
        eps_rewards = T.tensor(np.array(eps_rewards), dtype=T.float).to(self.device)

        # # REWARDS OF EACH STEP
        cum_rewards = T.zeros_like(eps_rewards)
        reward_len = len(eps_rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = eps_rewards[j] + (cum_rewards[j + 1] * self.gamma if j + 1 < reward_len else 0)

        # CRITIC - Optimize value loss (Critic)
        self.CriticNet.optimizer.zero_grad()

        values = self.CriticNet(eps_states)
        values = values.squeeze(dim=1)

        vf_loss = F.mse_loss(values, cum_rewards, reduction="none")
        vf_loss.sum().backward()
        self.CriticNet.optimizer.step()

        # ACTOR - Optimize policy loss (Actor)
        self.ActorNet.optimizer.zero_grad()

        with T.no_grad():
            values = self.CriticNet(eps_states)

        advantages = cum_rewards - values

        n = self.ActorNet(eps_states)
        log_prob = n.log_prob(eps_actions)
        pi_loss = - log_prob * advantages
        pi_loss = pi_loss.mean().backward()

        self.ActorNet.optimizer.step()


    def store_agent_parameters(self, mean_score_500, mean_score_100, note='None'):
        # print(f"Storing agent {self.agent_name} metadata ... ==> N_games= {self.n_games} | Mean_score={self.init_mean_score:.2f} > {mean_score}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            if self.agent_name not in metadata.keys():
                metadata[self.agent_name] = {}

            metadata[self.agent_name][current_time] = {}

            metadata[self.agent_name][current_time]['n_games'] = self.n_games
            metadata[self.agent_name][current_time]['mean_score_500'] = mean_score_500
            metadata[self.agent_name][current_time]['mean_score_100'] = mean_score_100
            metadata[self.agent_name][current_time]['note'] = 'NOTE'

        else:
            directory = os.path.dirname(METADATA_FILE)
            if not os.path.exists(directory):
                os.makedirs(directory)

            metadata = {self.agent_name: {
                current_time: {
                    'n_games': self.n_games,
                    'mean_score_500': mean_score_500,
                    'mean_score_100': mean_score_100,
                    'note': 'NOTE'}
            }}

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f'*** Model {self.agent_name} metadata saved | mean_score_500: {mean_score_500} | mean_score_100: {mean_score_100} ')


    def save(self):
        print(f'*** Saving Models {self.agent_name} parameters ...')
        self.ActorNet.save()
        self.CriticNet.save()


    def gpu_use(self):
        if not T.backends.mps.is_available():
            print("\tCHECK: CPU training")
            self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else:
            self.device = T.device("mps")
