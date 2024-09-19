import os
import torch as T
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, lr):
        """
        Args:
            input_size (int): The size of the input features.
            hidden_layers (list of int): A list where each element represents the number of units in a hidden layer.
            output_size (int): The size of the output layer.
        """
        super(PolicyNetwork, self).__init__()
        if len(hidden_layers) == 1:
            self.hidden_layers = str(hidden_layers[0])
        else:
            self.hidden_layers = "_".join(map(str, hidden_layers))

        print(f'\t*Creating Policy Network... {input_size}x{self.hidden_layers}x2')
        # Shared parameters
        self.shared_layer = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                          nn.ReLU())
        self.shared_layer2 = nn.Sequential(nn.Linear(hidden_layers[0], hidden_layers[1]),
                                          nn.ReLU())
        # Output layers: one for mean and one for standard deviation
        self.fc_mu = nn.Sequential(nn.Linear(hidden_layers[1], 1),
                                   nn.Tanh())  # Output for the mean.
        self.fc_std = nn.Sequential(nn.Linear(hidden_layers[1], 1),
                                    nn.Softplus())  # Output for the standard deviation

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        #
        if not T.backends.mps.is_available():
            print("\tCHECK: CPU training")
            self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else:
            self.device = T.device("mps")
        self.to(self.device)

        # self.load_model()

    def forward(self, state):
        shared = self.shared_layer(state)
        shared2 = self.shared_layer2(shared)

        # Output layers
        mu = 2 * self.fc_mu(shared2)  # todo no need to be cliped here
        std = self.fc_std(shared2) + 1e-3  # avoid zero values

        return T.distributions.Normal(mu, std)
