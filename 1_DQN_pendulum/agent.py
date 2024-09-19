import random
import json
from collections import deque
import torch as T  # PyTorch library for ML and DL

import time
from datetime import datetime

from model import DeepQNetwork
from utils import *

# -----------------------------
MAX_MEMORY = 100_000
METADATA_FILE = 'data/agent_DQN_metadata'


class DQN_Agent:
    def __init__(self, lr, gamma, env_class, epsilon_max, epsilon_decay, epsilon_min, action_interval, hidden_layers):
        self.agent_name = f'model'
        for hidden_layer in hidden_layers:
            self.agent_name = self.agent_name + f'_{hidden_layer}'

        print(f'\n ****** Creating Agent {self.agent_name}... ******')
        print(f'\tInput/Observation = {env_class.env.observation_space.shape[0]} | Output/action ={env_class.env.action_space.shape[0]}')

        # self.action_space = [i for i in range(env_class.env.action_space.shape[0])]
        n = (env_class.env.action_space.high[0] - env_class.env.action_space.low[0]) / action_interval + 1
        self.action_space = np.linspace(env_class.env.action_space.low[0], env_class.env.action_space.high[0], int(n))
        # self.action_space = np.float32(self.action_space)

        self.epsilon = epsilon_max  # Exploration rate
        self.epsilon_max = epsilon_max  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        self.hidden_layers = hidden_layers

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor

        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.agent_parameters()

        self.Q_model = DeepQNetwork(env_class.env.observation_space.shape[0], hidden_layers, self.action_space.shape[0], self.lr)

        self.Q_target = DeepQNetwork(env_class.env.observation_space.shape[0], hidden_layers, self.action_space.shape[0], self.lr)
        self.Q_target.load_state_dict(self.Q_model.state_dict())
        _ = self.Q_target.requires_grad_(False)  # target q-network doen't need grad

        self.action_space = T.tensor(self.action_space, dtype=T.float).to(self.Q_model.device)

    def get_action(self, observation: np.ndarray) -> int:
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation), dtype=T.float).to(self.Q_model.device)
            q_values_actions = self.Q_model.forward(state)
            return T.argmax(q_values_actions).item()
        else:
            return T.randint(0, self.action_space.shape[0], (1,)).item()

    def linear_epsilon_decay(self, episode, n_episodes):
        """
        Linearly decay epsilon from epsilon_max to epsilon_min over n_episodes.
        """
        decay_rate = (self.epsilon_max - self.epsilon_min) / n_episodes
        epsilon = self.epsilon_max - episode * decay_rate
        self.epsilon = max(epsilon, self.epsilon_min)

    def memory_replay(self, batch_size):
        # print('Memory replay...')
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.learning(states, actions, rewards, next_states, dones)

    def learning(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, state_: np.ndarray, done: np.ndarray) -> None:
        # if len(action) > 1:
        #     print(' ---------- Learning... batch size:', len(action), '| Memory size:', len(self.memory),' -------')

        states = T.tensor(np.array(state), dtype=T.float).to(self.Q_model.device)
        actions = T.tensor(action, dtype=T.int).to(self.Q_model.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.Q_model.device)
        states_ = T.tensor(np.array(state_), dtype=T.float).to(self.Q_model.device)
        dones = T.tensor(done, dtype=T.bool).to(self.Q_model.device)
        # (n, x)
        # print('states', states.shape, 'actions', actions.shape, 'rewards', rewards.shape, 'states_', states_.shape, 'done', done)

        # We check if the state is a single state or a batch of states
        if len(states.shape) == 1:
            #             print('Single state')
            # Size (1, x), unsqueeze turns an n.d. tensor into an (n+1).d. one by adding an extra dimension of depth 1.
            states = T.unsqueeze(states, 0)
            states_ = T.unsqueeze(states_, 0)
            actions = T.unsqueeze(actions, 0)
            rewards = T.unsqueeze(rewards, 0)
            dones = T.unsqueeze(dones, 0)
        #             print('states', states.shape, 'actions', actions.shape, 'rewards', rewards.shape, 'states_', states_.shape, 'dones', dones)

        # 1: predicted Q values with current state. We just have reward of the action we took
        q_pred = self.Q_model.forward(states)  # To check (actions)

        # print('===> q_predicted', q_pred)
        # print('===>  Actions:', actions)
        # print('\n')

        batch_indices = T.arange(q_pred.size(0))  # Tensor representing batch indices
        selected_q_values = q_pred[batch_indices, actions]

        #         print("Q-values for selected actions:", selected_q_values)

        # 1.2 Same size that we will use to store the target Q values
        q_target_pred = T.zeros_like(selected_q_values)

        # 2. We need to get the Q values of the next state
        for idx in range(len(states)):
            if not done[idx]:  # -> only do this if not done
                # 3.1 Q(s_{t+1}). 2 outputs, one for each action : size=[, 2]
                Q_target_output = self.Q_target(states_[idx])

                # We need to get the action that maximizes the Q value, action i that has a q_value = T.max(Q_target_output)
                Q_target_actions_max = T.max(Q_target_output)

                # 3.4 Q_target = r + gamma * max( Q(s_{t+1})  )
                q_target_pred[idx] = rewards[idx] + self.gamma * Q_target_actions_max
            else:
                q_target_pred[idx] = rewards[idx]

        self.Q_model.optimizer.zero_grad()  # Reset the gradients from the previous iteration

        loss = self.Q_model.loss(q_target_pred, selected_q_values)  # Calculate the loss
        loss.backward()  # Calculate the gradients

        self.Q_model.optimizer.step()

    def agent_parameters(self):

        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            print('\tModels metadata loaded: ')
            for key, value in metadata.items():
                print(f"\t\t{key} : {value}")
            #
            # if self.agent_name not in metadata.keys():
            #     print(f"\tNo metadata for {self.agent_name}")
            #     self.n_games = 0
            #     self.init_mean_score = 0
            # else:
            #     print(f"\tLoading metadata for {self.agent_name}...", end=" ")
            #
            #     required_keys = {'n_games', 'mean_score'}
            #     if required_keys <= metadata[self.agent_name].keys():
            #         self.n_games = metadata[self.agent_name]['n_games']
            #         self.init_mean_score = metadata[self.agent_name]['mean_score']
            #         print(f"Agent metadata loaded successfully ==> N_games= {self.n_games} | Mean_score={self.init_mean_score:.2f}")
            #     else:
            #         print(f"The file {METADATA_FILE} is missing some required keys. ERROR")

        else:
            print(f"\tNo metadata of the agent found.")
        self.n_games = 0
        # self.init_mean_score = 0

    def store_agent_parameters(self, mean_score):
        # print(f"Storing agent {self.agent_name} metadata ... ==> N_games= {self.n_games} | Mean_score={self.init_mean_score:.2f} > {mean_score}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            if self.agent_name not in metadata.keys():
                metadata[self.agent_name] = {}

            metadata[self.agent_name][current_time] = {}

            metadata[self.agent_name][current_time]['n_games'] = self.n_games
            metadata[self.agent_name][current_time]['mean_score'] = mean_score
            metadata[self.agent_name][current_time]['note'] = 'NOTE'

        else:
            directory = os.path.dirname(METADATA_FILE)
            if not os.path.exists(directory):
                os.makedirs(directory)

            metadata = {self.agent_name: {
                current_time: {
                    'n_games': self.n_games,
                    'mean_score': mean_score,
                    'note': 'NOTE'}
            }}

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f'*** Model {self.agent_name} metadata saved. Mean score: {mean_score}. ')
        for key, value in metadata.items():
            print(f"\t-> {key} : {value}")

    def save(self):
        print(f'*** Saving model {self.agent_name} parameters ...', end=" ")
        self.Q_model.save()

