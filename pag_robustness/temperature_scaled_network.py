import torch
import torch.nn as nn


def sample_from_dataloader(loader, num_points, std=0.1):
    #TODO: build in caching
    #TODO: bounds of the data after normalization
    """
    returns a tensor that is sampled from the given dataset with gaussian noise with std
    """
    all_inputs = []
    all_labels = []

    # Iterate through the DataLoader
    for inputs, labels in loader:
        all_inputs.append(inputs)
        all_labels.append(labels)

    # Concatenate all inputs and labels into two big tensors
    dataset = torch.cat(all_inputs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(dataset.shape)
    n, d = dataset.shape
    idx = torch.floor(torch.rand(num_points) * n).int()
    dataset_resampled = dataset[idx,]
    return torch.clamp(dataset_resampled + torch.randn_like(dataset_resampled) * std, 0, 1), labels_tensor[idx,]


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

    def set_temperature(self, val_loader, reg=0):
        """
        Tune the temperature parameter (T) on validation data to minimize negative log-likelihood (NLL).
        """
        self.model.eval()

        # Collect logits and labels from the validation set
        logits_list = []
        labels_list = []
        # points, labels = sample_from_dataloader(val_loader, num_points,std=0.01)

        with torch.no_grad():
            # logits = self.model(points)
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

        def eval_opt():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            # Regularization against overconfidence
            reg_strength = reg
            regularization_term = reg_strength * self.temperature
            loss = loss - regularization_term
            loss.backward()
            return loss

        optimizer.step(eval_opt)
        return self
