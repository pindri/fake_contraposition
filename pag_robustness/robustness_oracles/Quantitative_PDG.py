import numpy as np
import torch
from mair.attacks.attack import Attack
from torch import nn


class Quantitative_PGD:
    """
    adapted from the MAIR package

    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255h)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, device=torch.device("cpu")):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.device = device

    def __call__(self, data_loader):
        output_vectors = []
        for (X, y) in data_loader:
            y = self.model(X.to(device=self.device)).argmax(dim=1)
            output_vectors.append(self.forward(X, y))
        return torch.cat(output_vectors, dim=0)

    def forward(self, inputs, labels):
        self.model.eval()
        batch_size = 16192
        num_batches = inputs.size(0) // batch_size + int(inputs.size(0) % batch_size != 0)
        outputs = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, inputs.size(0))
            batch_points = inputs[start_idx:end_idx].to(self.device)  # Get the batch
            batch_labels = labels[start_idx:end_idx].to(self.device)  # Get the batch

            # Forward pass
            batch_outputs = self._forward(batch_points, batch_labels)

            # Move the outputs back to the CPU if you're using a GPU
            outputs.append(batch_outputs.cpu())

            # Concatenate all the outputs into a single tensor
        return torch.cat(outputs, dim=0)

    def _forward(self, inputs, labels):

        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_inputs = inputs.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(
                -self.eps, self.eps
            )
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()
        break_vector = torch.ones(adv_inputs.shape[0]).to(self.device) * self.steps

        idx = torch.Tensor(range(adv_inputs.shape[0])).int().to(self.device)
        # print(idx)
        for i in range(self.steps):
            adv_inputs.requires_grad = True
            outputs = self.model(adv_inputs)
            # Calculate loss
            cost = loss(outputs.to(self.device), labels[idx])

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs[idx,], min=-self.eps, max=self.eps)
            # Originally clamped in [0, 1] (to obtain a sample in the data space), but we don't like that.
            adv_inputs = torch.clamp(inputs[idx,] + delta, min=0, max=1).detach()

            # for all indices in breakdown vector where: the value is self.steps and the model label is NOT the
            # original label, set the breakdown vector index to i print(f"break_steps {torch.where((break_vector ==
            # self.steps) & (self.model(adv_inputs).argmax(dim=1) != labels))[0]}")
            # print(self.model(adv_inputs).argmax(dim=1))
            # print(self.model(inputs).argmax(dim=1))
            # print(inputs)
            # print(adv_inputs)



            # robust indices
            resistant_samples = ((self.model(adv_inputs).argmax(dim=1) == self.model(inputs[idx]).argmax(dim=1))
                                 .to(self.device))
            break_vector[idx[~resistant_samples]] = i
            adv_inputs = adv_inputs[resistant_samples]
            # print(i, inputs[idx,].shape)
            idx = idx[resistant_samples]
            if len(idx) == 0:
                break
        return break_vector


def batched_forward(model, points, batch_size = 1024, device = torch.device("cuda")):
    num_batches = points.size(0) // batch_size + int(points.size(0) % batch_size != 0)
    outputs = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, points.size(0))
            batch_points = points[start_idx:end_idx].to(device)  # Get the batch

            # Forward pass
            batch_outputs = model(batch_points)

            # Move the outputs back to the CPU if you're using a GPU
            outputs.append(batch_outputs.cpu())

            # Concatenate all the outputs into a single tensor
    return torch.cat(outputs, dim=0)