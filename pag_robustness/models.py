import torch
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

class _CopyModulesWrapper(nn.Module):
    """
    • Copies every sub-module from `backbone` into *this* module with the same
      names → state-dict stays byte-identical.
    • Registers mean / std as *non-persistent* buffers so they follow .cuda()
      but never appear in the checkpoint.
    """

    def __init__(self, backbone: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()

        # 1) replicate the backbone’s child hierarchy verbatim
        for name, module in backbone.named_children():
            self.add_module(name, module)

        # 2) non-persistent buffers for normalisation
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        # Call the original forward *as if we were the backbone*.
        # Most torchvision nets just: return self.classifier(self.features(x))
        # so we replicate that behaviour here.
        if hasattr(self, "classifier"):          # VGG / AlexNet style
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        else:                                   # generic safety net
            return super().forward(x)


# ────────────────────────────────────────────────────────────────────────────
def CifarNormalizedNetwork(backbone: nn.Module) -> nn.Module:
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return _CopyModulesWrapper(backbone, mean, std)

class _NormalizedWrapper(nn.Module):
    """
    Generic wrapper that transparently normalizes inputs before the backbone
    and *denormalizes* its outputs afterwards.
    • `state_dict()` and `load_state_dict()` are passed straight through to the
      backbone → checkpoints stay compatible.
    • Any unknown attribute is looked-up on the backbone, so code that expects
      e.g. `model.conv1.weight` still works.
    """
    def __init__(self, backbone: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", mean, persistent=False)           # follow .cuda(), .to(device) …
        self.register_buffer("std",  std, persistent=False)

    # ------------------------------------------------------------------ I/O
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)

    # ---------------------------------------------------------------- state-dict compat
    def state_dict(self, *args, **kwargs):
        """Save exactly what the backbone would save (no prefix, no buffers)."""
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Accept both
        • new-style   keys: backbone.features.*
        • legacy keys:      features.*
        Buffers (mean/std) are ignored.
        """
        remap = {}
        for k, v in state_dict.items():
            if k.endswith((".mean", ".std")):           # 1) drop wrapper buffers
                continue
            if k.startswith("backbone."):               # 2) new checkpoint → keep
                remap[k[len("backbone."):]] = v
            else:                                       # 3) old checkpoint → add prefix
                remap[k] = v

        missing, unexpected = self.backbone.load_state_dict(remap, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Missing keys: {missing}  Unexpected: {unexpected}")
        return missing, unexpected
    # --------------------------------------------------------------- attribute passthrough
    def __getattr__(self, name):
        # if attr not found on wrapper, delegate to backbone
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)



class CCifarNormalizedNetwork(_NormalizedWrapper):
    """Wrap any 3-channel CNN to operate on CIFAR-10-normalized inputs."""
    def __init__(self, backbone: nn.Module):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        super().__init__(backbone, mean, std)


class MNISTNormalizedNetwork(nn.Module):
    """Same idea for greyscale MNIST (1×28×28)."""
    MNIST_MEAN = torch.tensor([0.1307]).view(1, 1, 1, 1)
    MNIST_STD  = torch.tensor([0.3081]).view(1, 1, 1, 1)

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", self.MNIST_MEAN)
        self.register_buffer("std",  self.MNIST_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        y      = self.backbone(x_norm)
        return y * self.std + self.mean