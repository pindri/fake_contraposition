import concurrent.futures
import tempfile
import auto_LiRPA
import numpy as np
import torch
from maraboupy import Marabou

from auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule
from torch import nn

# git clone https://github.com/Verified-Intelligence/auto_LiRPA
# cd auto_LiRPA
# pip install .


def find_robustness_radius(model, point, max_radius=0.1, step_num=5):
    """ Worker function to process a single point. """
    radius = 0
    increment = max_radius / 2

    for _ in range(step_num):
        perturbation = PerturbationLpNorm(norm=np.inf,
                                          x_L=torch.clip(point.unsqueeze(0) - (radius+increment), min=-0.4242),
                                          x_U=torch.clip(point.unsqueeze(0) + (radius+increment), max=2.8215))
        bounded_input = BoundedTensor(point.unsqueeze(0), perturbation)
        output_bounds = model.compute_bounds(x=(bounded_input,), method="backward")
        if max(output_bounds[1][0]) <= 0:
            radius += increment
        increment /= 2
    return radius


def fdind_robustness_radius(model, point, max_radius=0.1, step_num=5):
    low, high = 0, max_radius
    best_eps = 0

    for _ in range(step_num):
        mid = (low + high) / 2
        perturbation = PerturbationLpNorm(norm=np.inf, eps=mid)
        bounded_input = BoundedTensor(point.unsqueeze(0), perturbation)
        output_bounds = model.compute_bounds(x=(bounded_input,), method="backward")
        if torch.max(output_bounds[0]) == torch.argmax(output_bounds[1]):
            best_eps = mid  # Still robust
            low = mid
        else:
            high = mid  # Not robust
    return best_eps


def quantitative_lirpa(model, step_num, max_radius, points, classes):
    radii = []
    class_model_list = {cls: get_bounded_class_model(model, cls, torch.empty_like(points[0].unsqueeze(0))) for cls in np.unique(classes)}
    for idx, point in enumerate(points.cuda()):
        # print("test")
        radius = find_robustness_radius(class_model_list[classes[idx].item()], point, max_radius, step_num)
        radii.append(radius)
        if idx % 100 == 0:
            print(f"Processed {idx} points")
    return torch.tensor(radii)


class WrappedModel(nn.Module):
    def __init__(self, pretrained_model, logit_index):
        super(WrappedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.logit_index = logit_index

    def forward(self, x):
        logits = self.pretrained_model(x)
        return logits - logits[:, self.logit_index].unsqueeze(1)


def get_bounded_class_model(model, cls, point):
    return BoundedModule(WrappedModel(model, cls), point)
