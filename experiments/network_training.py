import mair
import numpy as np
import pandas as pd
import wandb
from mair import Standard, AT, RobModel
import torch

from datasets import get_loaders
from network_utils import get_classes, get_confidences
from models import FFNetwork
from pag_robustness.robustness_oracles.Quantitative_Marabou import quantitative_Marabou
from pag_robustness.robustness_oracles.Quantitative_PDG import Quantitative_PGD
from pag_robustness.sample_complexities import complexity
from pag_robustness.temperature_scaled_network import TemperatureScaledNetwork, sample_from_dataloader


def get_base_model(dim_input: int, dim_output: int, network_type: str) -> RobModel:
    match network_type:
        case "feed_forward":
            normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
        case "resnet18":
            normal_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        case _:
            raise "unknown network type"
    robust_model = mair.RobModel(normal_model, n_classes=dim_output)
    if torch.cuda.is_available():
        robust_model = robust_model.cuda()
    return robust_model


def setup_and_get_data(dataset, network_type) -> (str, RobModel, dict):
    wandb.init(entity="peter-blohm-tu-wien", project=f"pag_experiment_{dataset}_{network_type}")
    name = f'{dataset}_{network_type}_{wandb.config.optimization_function}_{wandb.config.seed}'
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, train_loader, scaler_loader, val_loader, test_loader = (
        get_loaders(dataset, scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=dataset != "cifar10"))
    robust_model = get_base_model(dim_input, dim_output, network_type)
    return name, robust_model, {"train": train_loader, "scale": scaler_loader, "val": val_loader, "test": test_loader}


def train_network(dataset: str, network_type: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)

    # adversarial training or standard
    if wandb.config.optimization_function == "AT":
        trainer = AT(robust_model, eps=wandb.config.EPS,
                     alpha=wandb.config.ALPHA,
                     steps=wandb.config.STEPS)
    else:
        trainer = Standard(robust_model)

    trainer.record_rob(loaders["train"], loaders["val"], eps=wandb.config.EPS, alpha=wandb.config.ALPHA,
                       steps=wandb.config.STEPS, std=wandb.config.STD)
    trainer.setup(optimizer=f"SGD(lr={wandb.config.lr}, "
                            f"    momentum={wandb.config.momentum}, "
                            f"    weight_decay={wandb.config.weight_decay})",
                  scheduler="MultiStepLR(milestones=[100, 150], gamma=0.1)",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=wandb.config.n_epochs,
                  clip_grad_norm=1.0
                  )
    trainer.fit(train_loader=loaders["train"],
                n_epochs=wandb.config.n_epochs,
                save_path=f'../rob/{name}/',
                save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
                save_type="Epoch",
                save_overwrite=True,
                record_type="Epoch"
                )

    def log_loader_metric(loader_name, loader):
        return {
            f"{loader_name}-acc": robust_model.eval_accuracy(loader),
            f"{loader_name}-GN-rob": robust_model.eval_rob_accuracy_gn(loader, std=wandb.config.STD),
            f"{loader_name}-PGD-robustness": robust_model.eval_rob_accuracy_pgd(loader,
                                                                                eps=wandb.config.EPS,
                                                                                alpha=wandb.config.ALPHA,
                                                                                steps=wandb.config.STEPS),
            f"{loader_name}-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(loader, eps=wandb.config.EPS)
        }

    wandb.log({
        **log_loader_metric("train", loaders["train"]),
        **log_loader_metric("val", loaders["val"]),
        **log_loader_metric("test", loaders["test"])})


def temperature_scale_network(dataset: str, network_type: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)
    robust_model.load_state_dict(torch.load(f'../rob/{name}/best.pth', weights_only=False))

    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.set_temperature(loaders["scale"], reg=wandb.config.temp_scaling_regularization)
    torch.save(scaled_model.state_dict(), f'../rob/scaled/{name}/best.pth')


def sampling(dataset: str, network_type: str, method: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)
    robust_model.load_state_dict(torch.load(f'../rob/{name}/best.pth', weights_only=False))

    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.load_state_dict(torch.load(f'../rob/scaled/{name}/best.pth', weights_only=False))

    num_points = complexity(wandb.config.epsilon * (1 - wandb.config.kappa_max_quantile), wandb.config.delta)
    validation_sample = sample_from_dataloader(loader=loaders["val"], num_points=num_points,
                                               std=wandb.config.SAMPLING_GN_STD)
    preds = robust_model(validation_sample)

    match method:
        case "pgd":
            rob_pgd = Quantitative_PGD(robust_model, eps=wandb.config.PGD_EPS, alpha=wandb.config.PGD_ALPHA,
                                       steps=wandb.config.PGD_STEPS, random_start=False)

            classes = get_classes(robust_model(validation_sample))
            steps, distances = rob_pgd.forward(validation_sample, classes)
        case "marabou":
            distances = quantitative_Marabou(robust_model, points=validation_sample,
                                             step_num=wandb.config.MARABOU_STEPS,
                                             max_radius=wandb.config.MARABOU_MAX_RADIUS)
            steps = torch.zeros_like(distances)
            classes = get_classes(robust_model(validation_sample))
        case _:
            raise "unknown robustness oracle"
    print("Robustness done")
    confs = get_confidences(preds)
    scaled_confs = get_confidences(preds / scaled_model.temperature)

    table = wandb.Table(columns=["PGD_robustness", "confidence", "scaled_confidence", "class"])
    df = pd.DataFrame({"PGD_robustness_steps": steps.tolist(),
                       "PGD_robustness_distances": distances.tolist(),
                       "confidence": confs.tolist(),
                       "scaled_confidence": scaled_confs.tolist(),
                       "class": classes.tolist()})
    # Save the DataFrame to CSV locally
    df.to_csv(f"../results/{name}.csv", index=False)
    # Add data points to the table
    for i in range(num_points):
        table.add_data(steps[i], distances[i], confs[i], scaled_confs[i], classes[i])
    wandb.log({"sampling_table": table})
    # wandb.log({"scatter_plot": wandb.plot.scatter(table, "x", "y", title="Scatter Plot Example")})


def testing():
    wandb.init(entity="peter-blohm-tu-wien", project="pag_mnist_test")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, _, _, _, test_loader = (
        get_loaders('mnist', scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=True))

    normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
    robust_model = mair.RobModel(normal_model, n_classes=10)
    robust_model.load_state_dict(torch.load(f"../models/{wandb.config.optimization_function}_{wandb.config.seed}.torch",
                                            weights_only=False))
    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.load_state_dict(
        torch.load(f"../models/scaled/{wandb.config.optimization_function}_{wandb.config.seed}.torch",
                   weights_only=False))
    num_points = complexity(wandb.config.epsilon * (1 - wandb.config.kappa_max_quantile), wandb.config.delta)
    all_inputs = []
    all_labels = []

    # Iterate through the DataLoader
    for inputs, labels in test_loader:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    num_points = len(all_inputs)

    # validation_sample = sample_from_dataloader(loader=sampler_loader, num_points=num_points, std=wandb.config.SAMPLING_GN_STD)
    preds = robust_model(all_inputs)
    print("Sampling done")
    rob_pgd = Quantitative_PGD(robust_model, eps=wandb.config.PGD_EPS, alpha=wandb.config.PGD_ALPHA,
                               steps=wandb.config.PGD_STEPS, random_start=False)

    classes = get_classes(robust_model(all_inputs))
    robs = rob_pgd.forward(all_inputs, classes)
    print("Robustness done")
    confs = get_confidences(preds)
    scaled_confs = get_confidences(preds / scaled_model.temperature)

    table = wandb.Table(columns=["PGD_robustness", "confidence", "scaled_confidence", "class"])
    df = pd.DataFrame({"PGD_robustness": robs.tolist(),
                       "confidence": confs.tolist(),
                       "scaled_confidence": scaled_confs.tolist(),
                       "class": classes.tolist()})
    # Save the DataFrame to CSV locally
    df.to_csv(f"../results/test_{wandb.config.optimization_function}_{wandb.config.seed}.csv", index=False)
    # Add data points to the table
    for i in range(num_points):
        table.add_data(robs[i], confs[i], scaled_confs[i], classes[i])
    wandb.log({"sampling_table": table})
    print(f"accuracy: {(all_labels == classes).sum()}")
    # wandb.log({"scatter_plot": wandb.plot.scatter(table, "x", "y", title="Scatter Plot Example")})
