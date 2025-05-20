import torch
from torch import nn

from pag_robustness.temperature_scaled_network import denormalize_data, renormalize_data



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
        print("sample-checking in progress")
        self.model.eval()
        batch_size = 4096
        num_batches = inputs.size(0) // batch_size + int(inputs.size(0) % batch_size != 0)
        outputs = []
        dists = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, inputs.size(0))

                batch_points = inputs[start_idx:end_idx]  # Get the batch
                batch_labels = labels[start_idx:end_idx]  # Get the batch

                # Forward pass
                print(f"attacking batch {i}")
                batch_outputs, batch_dists = self._forward(batch_points, batch_labels)

                # Move the outputs back to the CPU if you're using a GPU
                outputs.append(batch_outputs.cpu())
                dists.append(batch_dists.cpu())

            # Concatenate all the outputs into a single tensor
        return torch.cat(outputs, dim=0), torch.cat(dists, dim=0)

    def _forward(self, inputs, labels) -> (torch.Tensor, torch.Tensor):
        with torch.enable_grad():
            inputs = inputs.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            loss = nn.CrossEntropyLoss()
            adv_inputs = inputs.clone().detach().to(self.device)
            denormalized_inputs = denormalize_data(inputs.clone().detach())
            denormalized_inputs.requires_grad = False

            if self.random_start:
                # Starting at a uniformly random point
                adv_inputs = denormalize_data(adv_inputs) + torch.rand_like(inputs)*0.2*self.eps-0.1*self.eps
                adv_inputs = renormalize_data(torch.clamp(adv_inputs, min=0, max=1).detach())

            break_vector = torch.ones(adv_inputs.shape[0]).to(self.device) * self.steps
            distance_vector = torch.ones(adv_inputs.shape[0]).to(self.device) * float('inf')

            idx = torch.Tensor(range(adv_inputs.shape[0])).int().to(self.device)
            # print(idx)
            for i in range(self.steps):

                adv_inputs.requires_grad = True
                outputs = self.model(adv_inputs)
                # Calculate loss
                cost = loss(outputs.to(self.device), labels[idx])

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_inputs, retain_graph=False, create_graph=False)[0]

                delta = torch.clamp(denormalize_data(adv_inputs) + self.alpha * grad.sign() - denormalized_inputs[idx,],
                                    min=-self.eps, max=self.eps)
                # Originally clamped in [0, 1] (to obtain a sample in the data space), but we don't like that.

                adv_inputs = renormalize_data(torch.clamp(denormalized_inputs[idx,] + delta, min=0, max=1).detach())
                # print(adv_inputs.min(), adv_inputs.max())
                # adv_inputs = torch.clamp(inputs[idx,] + delta, min=0, max=1).detach()

                # for all indices in breakdown vector where: the value is self.steps and the model label is NOT the
                # original label, set the breakdown vector index to i print(f"break_steps {torch.where((break_vector ==
                # self.steps) & (self.model(adv_inputs).argmax(dim=1) != labels))[0]}")
                # print(self.model(adv_inputs).argmax(dim=1))
                # print(self.model(inputs).argmax(dim=1))
                # print(inputs)
                # print(adv_inputs)

                resistant_samples = ((self.model(adv_inputs).argmax(dim=1) == labels[idx])
                                     .to(self.device))
                break_vector[idx[~resistant_samples]] = i

                distance_vector[idx[~resistant_samples]] = (torch.norm(inputs[idx]-adv_inputs,
                                                                       p=float('inf'),
                                                                       dim=list(range(1, inputs.ndim)))
                                                            )[~resistant_samples]
                adv_inputs = adv_inputs[resistant_samples]
                # adv_inputs = adv_inputs[resistant_samples]
                # print(i, inputs[idx,].shape)
                idx = idx[resistant_samples]
                # print(len(idx), i)
                if len(idx) == 0:
                    break
        return break_vector, distance_vector


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