import numpy as np
import json
from collections import deque
import torch as T  # PyTorch library for ML and DL
import torch.nn.functional as F  # PyTorch's functional module
from torch.distributions import Categorical

from model import PolicyNetwork
from utils import *

# -----------------------------
MAX_MEMORY = 100_000
MODEL_FILE = '1_DQN_pendulum'
METADATA_FILE = 'data/agent_PG_discrete_metadata'


class GD_Agent:
    def __init__(self, env_class, lr, gamma, hidden_layers):
        # MODEL NAME
        self.agent_name = f'model'
        for hidden_layer in hidden_layers:
            self.agent_name += f'_{hidden_layer}'

        print(f'\n ****** Creating Agent {self.agent_name}... ******')
        print(f'\tInput/Observation = {env_class.env.observation_space.shape[0]} | Output/action ={env_class.env.action_space.shape[0]}')


        self.hidden_layers = hidden_layers

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.n_games = 0

        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.gpu_use()
        self.PolicyPi = PolicyNetwork(env_class.env.observation_space.shape[0],
                                      hidden_layers,
                                      self.lr).to(self.device)

    def get_action(self, observation: np.ndarray) -> int:
        with T.no_grad():
            n_distrib = self.PolicyPi(T.tensor(observation, dtype=T.float).to(self.device))
            return np.clip(n_distrib.sample().item(), -2., 2.)

    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = T.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def policy_update(self, eps_data: list) -> None:
        eps, eps_states, eps_actions, eps_rewards = zip(*eps_data)
        eps_states = T.tensor(np.array(eps_states), dtype=T.float).to(self.PolicyPi.device)
        eps_actions = T.tensor(np.array(eps_actions), dtype=T.int).to(self.PolicyPi.device)
        eps_rewards = T.tensor(np.array(eps_rewards), dtype=T.float).to(self.PolicyPi.device)

        cum_reward = self.reward_to_go(eps_rewards)

        n = self.PolicyPi(eps_states)
        log_prob = n.log_prob(eps_actions)

        batch_loss = -(log_prob * cum_reward).mean()

        self.PolicyPi.optimizer.zero_grad()
        batch_loss.backward()
        self.PolicyPi.optimizer.step()

    def store_agent_parameters(self, mean_score_500, mean_score_100, note):
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
            metadata[self.agent_name][current_time]['note'] = note

        else:
            directory = os.path.dirname(METADATA_FILE)
            if not os.path.exists(directory):
                os.makedirs(directory)

            metadata = {self.agent_name: {
                current_time: {
                    'n_games': self.n_games,
                    'mean_score_500': mean_score_500,
                    'mean_score_100': mean_score_100,
                    'note': note}
            }}

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f'*** Model {self.agent_name} metadata saved | mean_score_500: {mean_score_500} | mean_score_100: {mean_score_100} ')


    def gpu_use(self):
        if not T.backends.mps.is_available():
            print("\tCHECK: CPU training")
            self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else:
            self.device = T.device("mps")

    def print_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            print('\tModels metadata... ')
            for model, date in metadata.items():
                print(f"\t\t-->{model} : ")
                for date, data in date.items():
                    print(f"\t\t\t]{date}] : {data}")