import torch
import torch.nn as nn


class BigConvOnnx(nn.Module):
    def __init__(self):
        super(BigConvOnnx, self).__init__()
        # Hard-coded normalization constants (from Constant_15 and Constant_17)

        # Convolutional layers (from Conv_19, Conv_21, Conv_23, Conv_25)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers (from Gemm_28, Gemm_30, Gemm_output)
        self.fc1 = nn.Linear(in_features=3136, out_features=512, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        # Normalize input: (x - 0.1307) / 0.3081

        x = self.conv1(x.view(-1, 1, 28, 28)
)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        x = self.relu6(x)

        x = self.fc3(x)
        return x


def load_weights_from_onnx_state_dict(onnx_state_dict, model):
    """
    Map weights from an ONNX-converted state_dict to the custom model.
    Adjust the keys if necessary.
    """
    # Mapping from ONNX node names to our module's parameter names
    mapping = {
        "Conv_19.weight": "conv1.weight",
        "Conv_19.bias": "conv1.bias",
        "Conv_21.weight": "conv2.weight",
        "Conv_21.bias": "conv2.bias",
        "Conv_23.weight": "conv3.weight",
        "Conv_23.bias": "conv3.bias",
        "Conv_25.weight": "conv4.weight",
        "Conv_25.bias": "conv4.bias",
        "Gemm_28.weight": "fc1.weight",
        "Gemm_28.bias": "fc1.bias",
        "Gemm_30.weight": "fc2.weight",
        "Gemm_30.bias": "fc2.bias",
        "Gemm_output.weight": "fc3.weight",
        "Gemm_output.bias": "fc3.bias"
    }

    new_state = {}
    for onnx_name, model_name in mapping.items():
        if onnx_name in onnx_state_dict:
            new_state[model_name] = onnx_state_dict[onnx_name]
        else:
            print(f"Warning: {onnx_name} not found in the ONNX state_dict.")
    model.load_state_dict(new_state, strict=False)
    return model

# Example usage:
# Assuming you have already converted your ONNX file using onnx2pytorch, e.g.:
#
#   import onnx
#   from onnx2pytorch import ConvertModel
#
#   onnx_model = onnx.load("model.onnx")
#   converted_model = ConvertModel(onnx_model)
#   onnx_state_dict = converted_model.state_dict()
#
# Then you can build your custom module and load the weights:

# custom_model = CustomONNXModel()
# custom_model = load_weights_from_onnx_state_dict(onnx_state_dict, custom_model)
#
# # Test with a dummy input (batch size 8, 1 channel, e.g., 28x28 image if applicable)
# dummy_input = torch.randn(8, 1, 28, 28)
# output = custom_model(dummy_input)
# print("Output shape:", output.shape)
