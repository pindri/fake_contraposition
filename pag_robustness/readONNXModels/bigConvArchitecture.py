import math

import torch.nn as nn
import torch.nn.functional as F

class ConvBigMNIST(nn.Module):
    """
    Faithful PyTorch reproduction of DiffAI's convBig for 1×28×28 inputs.
    Output: logits for `num_classes` (default = 10).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # -------- convolutional backbone --------
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # After two stride-2 layers:
        # 28 → 14 → 7     ⇒  64 × 7 × 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self._init_weights()

    # -------- weight initialisation identical to DiffAI --------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

    # -------- forward pass --------
    def forward(self, x):
        #x = (x - 0.1307) / 0.3081  # transforms.Normalize((0.1307,), (0.3081,)) # TODO
        x = F.relu(self.conv1(x))   # 32 × 28 × 28
        x = F.relu(self.conv2(x))   # 32 × 14 × 14
        x = F.relu(self.conv3(x))   # 64 × 14 × 14
        x = F.relu(self.conv4(x))   # 64 ×  7 ×  7
        x = x.flatten(1)            # 3136
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)