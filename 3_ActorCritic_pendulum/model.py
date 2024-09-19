import os
import torch as T
import torch.nn as nn


# --------------------------------------------------------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, name, input_size, hidden_layers, lr):
        super(ValueNetwork, self).__init__()
        # self.type = name
        # if len(hidden_layers) == 1:
        #     self.hidden_layers = str(hidden_layers[0])
        # else:
        #     self.hidden_layers = "_".join(map(str, hidden_layers))

        print(f'\t*Creating {name} Network... {input_size}x{hidden_layers}x1')

        layers = []
        layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                    nn.ReLU()))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Sequential(nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                                            nn.ReLU()))
        layers.append(nn.Linear(hidden_layers[-1], 1))  # Value function, it can take all values

        # Register all layers using nn.ModuleList
        self.layers = nn.ModuleList(layers)

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        # self.load_model()

    def forward(self, state) -> T.tensor:
        input_weights = state
        for layer in self.layers[:-1]:
            input_weights = T.relu(layer(input_weights))
        return self.layers[-1](input_weights)  # Value function


class PolicyNetwork(nn.Module):
    def __init__(self, name, input_size, hidden_layers, lr):
        super(PolicyNetwork, self).__init__()
        # self.type = name
        # if len(hidden_layers) == 1:
        #     self.hidden_layers = str(hidden_layers[0])
        # else:
        #     self.hidden_layers = "_".join(map(str, hidden_layers))

        print(f'\t*Creating {name} Network... {input_size}x{hidden_layers}x2')

        self.shared_layer = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                          nn.ReLU())
        self.shared_layer2 = nn.Sequential(nn.Linear(hidden_layers[0], hidden_layers[1]),
                                           nn.ReLU())

        # Output layers: one for mean and one for standard deviation
        self.fc_mu = nn.Sequential(nn.Linear(hidden_layers[1], 1),
                                   nn.Tanh())  # Output for the mean.
        self.fc_std = nn.Sequential(nn.Linear(hidden_layers[1], 1),
                                    nn.Softplus())  # Output for the standard deviation

        # Optimizer
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        # self.load_model() # todo separate mu and std

    def forward(self, state):
        shared = self.shared_layer(state)
        shared2 = self.shared_layer2(shared)

        # Output layers
        mu = 2 * self.fc_mu(shared2)  # todo no need to be cliped here
        std = self.fc_std(shared2) + 1e-3  # avoid zero values

        return T.distributions.Normal(mu, std)
