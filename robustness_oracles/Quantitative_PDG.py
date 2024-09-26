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
            output_vectors.append(self.forward(X, y))
        return torch.cat(output_vectors, dim=0)

    def forward(self, inputs, labels):

        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_inputs = inputs.clone().detach()

        if self.random_start:  # what does THIS do?
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(
                -self.eps, self.eps
            )
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()

        break_vector = torch.ones(adv_inputs.shape[0]) * self.steps

        for i in range(self.steps):
            adv_inputs.requires_grad = True
            outputs = self.model(adv_inputs)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            # Originally clamped in [0, 1] (to obtain a sample in the data space), but we don't like that.
            adv_inputs = torch.clamp(inputs + delta, min=-np.inf, max=np.inf).detach()

            # for all indices in breakdown vector where: the value is self.steps and the model label is NOT the
            # original label, set the breakdown vector index to i print(f"break_steps {torch.where((break_vector ==
            # self.steps) & (self.model(adv_inputs).argmax(dim=1) != labels))[0]}")
            break_vector[torch.where((break_vector == self.steps) & (self.model(adv_inputs).argmax(dim=1) != labels))[0]] = i

        return break_vector
