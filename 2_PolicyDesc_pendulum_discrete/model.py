import os
import torch as T
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, lr):
        """
        Args:
            input_size (int): The size of the input features.
            hidden_layers (list of int): A list where each element represents the number of units in a hidden layer.
            output_size (int): The size of the output layer.
        """
        super(Network, self).__init__()
        print(f'\t*Creating Network... {input_size}x{hidden_layers}x{output_size}')

        if len(hidden_layers) == 1:
            self.hidden_layers = str(hidden_layers[0])
        else:
            self.hidden_layers = "_".join(map(str, hidden_layers))

        layers = []

        layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                    nn.ReLU()))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Sequential(nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                                            nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(hidden_layers[-1], output_size),
                      nn.Tanh()))

        # Register all layers using nn.ModuleList
        self.layers = nn.ModuleList(layers)

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


    def forward(self, state) -> T.tensor:
        input_weights = state
        for layer in self.layers[:-1]:
            input_weights = T.relu(layer(input_weights))
        return self.layers[-1](input_weights) # logits


