import torch
import torch.nn as nn


class TemperatureScaledNetwork(nn.Module):
    def __init__(self, model):
        super(TemperatureScaledNetwork, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initialize T = 1

    def forward(self, inputs):
        # Get logits from the original model
        logits = self.model(inputs)
        # Apply temperature scaling
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Scale logits by temperature and apply softmax
        temperature = self.temperature.expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, val_loader):
        """
        Tune the temperature parameter (T) on validation data to minimize negative log-likelihood (NLL).
        """
        self.model.eval()

        # Collect logits and labels from the validation set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        # Concatenate collected logits and labels
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature using NLL loss
        nll_criterion = nn.CrossEntropyLoss()

        # Use an optimizer to adjust the temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=5000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self
