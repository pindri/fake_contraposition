from torch import nn


class FFNetwork(nn.Module):
    """
    This class defines a basic feed-forward neural network.
    It can be specified by input size, output size and a list of hidden layer sizes.
    During forward passes, the last hidden layer will be the output without application of an activation function.
    """
    def __init__(self, input_dim: int, output_dim: int, layer_sizes: [int], activation: nn.Module = nn.ReLU()):
        super(FFNetwork, self).__init__()
        self.activation = activation

        self.layers = nn.ModuleList()
        current_layer = input_dim
        for i, size in enumerate(layer_sizes + [output_dim]):
            self.layers.append(nn.Linear(current_layer, size))
            current_layer = size

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
