import mair
import numpy as np
import seaborn as sns
import torch
from mair import AT, Standard
from matplotlib import pyplot as plt, gridspec

from datasets import get_loaders
from models import FFNetwork
from pag_robustness.robustness_oracles.Quantitative_PDG import Quantitative_PGD


def train_and_store_networks(seed, path, robust):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dim_input, dim_output, train_loader, _, val_loader, test_loader = (
        get_loaders('cifar10', val_split=0.2, batch_size=512, flatten=False))

    # resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=False)
    normal_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    robust_model = mair.RobModel(normal_model, n_classes=dim_output)

    # return robust_model, val_loader, test_loader
    if torch.cuda.is_available():
        robust_model = robust_model.cuda()
    if robust:
        trainer = AT(robust_model, eps=0.5,
                     alpha=.1,
                     steps=10)
    else:
        trainer = Standard(robust_model)
    trainer.record_rob(train_loader, val_loader, eps=.5, alpha=.1,
                       steps=10, std=1)
    trainer.setup(optimizer=f"SGD(lr={10**-4}, momentum={0.1})",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=5,
                  )
    trainer.fit(train_loader=train_loader,
                n_epochs=5,
                save_path='../../rob/',
                save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
                save_type="Epoch",
                save_overwrite=True,
                record_type="Epoch"
                )
    print(
        {"train-acc": robust_model.eval_accuracy(train_loader),
         "train-GN-rob": robust_model.eval_rob_accuracy_gn(train_loader, std=1),
         "train-PGD robustness": robust_model.eval_rob_accuracy_pgd(train_loader, eps=0.5, alpha=0.1, steps=10),
         "train-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(train_loader, eps=0.5),
         "val-acc": robust_model.eval_accuracy(val_loader),
         "val-GN-rob": robust_model.eval_rob_accuracy_gn(val_loader, std=1),
         "val-PGD robustness": robust_model.eval_rob_accuracy_pgd(val_loader, eps=0.5, alpha=0.1, steps=10),
         "val-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(val_loader, eps=0.5),
         "test-acc": robust_model.eval_accuracy(test_loader),
         "test-GN-rob": robust_model.eval_rob_accuracy_gn(test_loader, std=1),
         "test-PGD robustness": robust_model.eval_rob_accuracy_pgd(test_loader, eps=0.5, alpha=0.1, steps=10),
         "test-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(test_loader, eps=0.5)})
    return robust_model, val_loader, test_loader


def sample_from_dataloader(loader, num_points, std=0.1):
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
    return torch.clamp(dataset_resampled + torch.randn_like(dataset_resampled) * std, 0, 1)


def get_confidences(logits: torch.tensor) -> torch.tensor:
    return torch.max(torch.softmax(logits, dim=-1), dim=-1)[0]


def get_classes(logits: torch.tensor) -> torch.tensor:
    return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)


def scatter_with_density(preds, robs):
    fig = plt.figure(figsize=(8, 8))
    x = get_confidences(preds).detach().numpy()
    y = robs
    gs = gridspec.GridSpec(4, 4, width_ratios=[0.2, 1, 1, 0.2], height_ratios=[0.2, 1, 1, 0.2])

    # Scatter plot (placed in the middle)
    ax_main = plt.subplot(gs[1:3, 1:3])
    ax_main.scatter(x, y, c=get_classes(preds).detach().numpy(), alpha=0.5)
    ax_main.set_xlabel("Confidence Score (softmax)")
    ax_main.set_ylabel("Robustness (PGD Steps)")

    # Density plot for X-axis (above scatter plot)
    ax_top = plt.subplot(gs[0, 1:3], sharex=ax_main)
    sns.kdeplot(x, ax=ax_top, fill=True)
    ax_top.set_ylabel('Density')
    ax_top.get_xaxis().set_visible(False)

    # Density plot for Y-axis (right of scatter plot)
    ax_right = plt.subplot(gs[1:3, 3], sharey=ax_main)
    sns.kdeplot(y, ax=ax_right, fill=True, vertical=True)
    ax_right.set_xlabel('Density')
    ax_right.get_yaxis().set_visible(False)

    # Hide tick labels for density plots
    ax_top.tick_params(axis='x', which='both', bottom=False)
    ax_right.tick_params(axis='y', which='both', left=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, val_loader, test_loader = train_and_store_networks(3,3,True)
    validation_sample = sample_from_dataloader(val_loader, 300000, std=8 / 255)
    preds = model(validation_sample)

    rob_pgd = Quantitative_PGD(model, eps=127 / 255, alpha=1 / 255, steps=100, random_start=False)
    robs = rob_pgd.forward(validation_sample, get_classes(model(validation_sample)))

    scatter_with_density(preds, robs)
    print()

    print(get_confidences(preds).shape)
