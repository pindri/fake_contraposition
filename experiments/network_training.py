import gc
import os
import time

import torch
from pag_robustness.readONNXModels.bigConvArchitecture import ConvBigMNIST

from pag_robustness.robustness_oracles.Quantitative_LiRPA import quantitative_lirpa

if not torch.cuda.is_available():
    raise Exception("cuda is not available")

import mair
import numpy as np
import pandas as pd
from torch import Tensor

import wandb
from mair import Standard, RobModel, TRADES

from pag_robustness.datasets import get_kfold_loaders
from network_utils import get_classes, get_confidences
from pag_robustness.models import FFNetwork, CifarNormalizedNetwork, CCifarNormalizedNetwork
from pag_robustness.robustness_oracles.Quantitative_Marabou import quantitative_marabou
from pag_robustness.robustness_oracles.Quantitative_PDG import QuantitativePGD
from pag_robustness.sample_complexities import complexity
from pag_robustness.temperature_scaled_network import sample_from_dataloader


def get_base_model(dim_input: int, dim_output: int, network_type: str) -> RobModel:
    match network_type:
        case "feed_forward":
            normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
        case "resnet20":
            normal_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=False)
        case "vgg11_bn":
            normal_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg11_bn', pretrained=False)
        case "convBig":
            normal_model = ConvBigMNIST()
        case _:
            raise "unknown network type"
    if torch.cuda.is_available():
        normal_model = normal_model.to(device="cuda")
    robust_model = mair.RobModel(normal_model, n_classes=dim_output)

    return robust_model


def setup_and_get_data(dataset, network_type) -> (str, RobModel, dict):
    wandb.init()
    name = f'{dataset}_{network_type}_{wandb.config.optimization_function}_{wandb.config.seed}_{wandb.config.robust_beta}'

    # eliminate all sources of randomness
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)
    torch.manual_seed(wandb.config.seed)  # Set PyTorch seed for CPU operations
    torch.cuda.manual_seed(wandb.config.seed)  # Set PyTorch seed for current GPU
    torch.cuda.manual_seed_all(wandb.config.seed)  # Set seed for all GPUs if using multi-GPU
    #TODO this breaks determinism but speeds up training
    torch.backends.cudnn.deterministic = not True  # Enforce deterministic algorithms
    torch.backends.cudnn.benchmark = not False  # Disable automatic optimizations that could introduce randomness

    dim_input, dim_output, train_loader, val_loader, test_loader = (
        get_kfold_loaders(dataset, split_index=wandb.config.seed, batch_size=wandb.config.batch_size, flatten= (network_type != "convBig") and (dataset != "cifar10")))
    robust_model = get_base_model(dim_input, dim_output, network_type)
    return name, robust_model, {"train": train_loader, "val": val_loader, "test": test_loader}


def train_network(dataset: str, network_type: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)

    # adversarial training or standard
    if wandb.config.optimization_function == "TRADES":
        trainer = TRADES(robust_model, eps=wandb.config.EPS,
                         alpha=wandb.config.ALPHA,
                         steps=wandb.config.STEPS, beta=wandb.config.robust_beta)
    elif  wandb.config.optimization_function == "Standard":
        trainer = Standard(robust_model)
    else:
        Exception("Unknown optimization function")
    trainer.record_rob(loaders["train"], loaders["val"], eps=wandb.config.EPS, alpha=wandb.config.ALPHA,
                       steps=wandb.config.STEPS, std=wandb.config.STD)
    trainer.setup(optimizer=f"SGD(lr={wandb.config.lr}, "
                            f"    momentum={wandb.config.momentum}, "
                            f"    weight_decay={wandb.config.weight_decay})",
                  scheduler=f"CosineAnnealingLR(T_max={wandb.config.n_epochs})",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=wandb.config.n_epochs,
                  )
    # with torch.autocast("cuda"):
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


def attack_model(name: str, model: RobModel, data: Tensor, method: str, labels: Tensor = None):
    start_time = time.time()
    batch_size = 16186//4
    num_batches = data.size(0) // batch_size + int(data.size(0) % batch_size != 0)
    preds = []
    data.requires_grad = False

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, data.size(0))
            batch_points = data[start_idx:end_idx].cuda()  # Get the batch

            # Forward pass
            print(f"predicting batch {i}")
            batch_preds = model(batch_points)

            # Move the outputs back to the CPU if you're using a GPU
            preds.append(batch_preds.cpu())
            del batch_points, batch_preds
            torch.cuda.empty_cache()
        # Concatenate all the outputs into a single tensor
    preds = torch.cat(preds, dim=0)
    # print(confs[1])
    match method:
        case "pgd":
            rob_pgd = QuantitativePGD(model, eps=wandb.config.PGD_EPS, alpha=wandb.config.PGD_ALPHA,
                                      steps=wandb.config.PGD_STEPS, random_start=False, device="cuda")
            classes = get_classes(preds)
            classes.requires_grad = False
            steps, distances = rob_pgd.forward(data, classes)
        case "marabou":
            distances = quantitative_marabou(model.cpu(), points=data,
                                             step_num=wandb.config.MARABOU_STEPS,
                                             max_radius=wandb.config.MARABOU_MAX_RADIUS)
            steps = torch.zeros_like(distances)
            classes = get_classes(preds)
        case "lirpa":
            with torch.no_grad():
                distances = quantitative_lirpa(model, points=data.cuda(),
                                                 step_num=wandb.config.MARABOU_STEPS,
                                                 max_radius=wandb.config.MARABOU_MAX_RADIUS,
                                                 classes = get_classes(preds))
            steps = torch.zeros_like(distances)
            classes = get_classes(preds)
        case _:
            raise "unknown robustness oracle"
    print("Robustness done")
    confs = get_confidences(preds)
    if labels is None:
        labels = classes
    df = pd.DataFrame({f"{method}_robustness_steps": steps.tolist(),
                       f"{method}_robustness_distances": distances.tolist(),
                       "confidence": confs.tolist(),
                       "pred_class": classes.tolist(),
                       "true class": labels.tolist(),
                       "runtime": time.time() - start_time})
    # Save the DataFrame to CSV locally

    df.to_csv(f"../results/sampling_{name}.csv", index=False)
    del df
    # Add data points to the table
    # for i in range(len(data)):
    #     table.add_data(steps[i], distances[i], confs[i], scaled_confs[i], classes[i])
    # wandb.log({"sampling_table": table})
    return classes


def sampling(dataset: str, network_type: str, method: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)

    robust_model.load_state_dict(
        torch.load(f'../rob/{name}/last.pth', weights_only=False, map_location="cuda")["rmodel"])
    robust_model = (robust_model).cuda()

    np.random.seed(wandb.config.SAMPLING_SEED)
    torch.random.manual_seed(wandb.config.SAMPLING_SEED)
    torch.manual_seed(wandb.config.SAMPLING_SEED)  # Set PyTorch seed for CPU operations
    torch.cuda.manual_seed(wandb.config.SAMPLING_SEED)  # Set PyTorch seed for current GPU
    torch.cuda.manual_seed_all(wandb.config.SAMPLING_SEED)  # Set seed for all GPUs if using multi-GPU

    num_points = complexity(wandb.config.epsilon * (1 - wandb.config.kappa_max_quantile), wandb.config.delta)
    validation_sample = sample_from_dataloader(loader=loaders["val"], num_points=num_points,
                                               std=wandb.config.SAMPLING_GN_STD)

    for param in robust_model.parameters():
        param.requires_grad = False
    robust_model.eval()

    print("val", validation_sample[0].min(), validation_sample[0].max())
    attack_model(f"{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_{wandb.config.SAMPLING_SEED}", robust_model, validation_sample[0], method)
    del robust_model
    validation_sample[0].detach().cpu()
    del validation_sample
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.empty_cache()
    wandb.finish()


def testing(dataset: str, network_type: str, method: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)

    robust_model.load_state_dict(
        torch.load(f'../rob/{name}/last.pth', weights_only=False, map_location="cuda")["rmodel"])
    robust_model = (robust_model).cuda()

    for param in robust_model.parameters():
        param.requires_grad = False
    robust_model.eval()

    all_inputs = []
    all_labels = []
    for inputs, labels in loaders["val"]:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_inputs = (torch.cat(all_inputs, dim=0))
    print(all_inputs.min(), all_inputs.max())
    classes = attack_model(f"validation_set_{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_best_shiny", robust_model,
                           all_inputs.clone(), method, all_labels.clone())
    print(f"accuracy: {(all_labels.cuda() == classes.cuda()).sum()}", len(classes))

    all_inputs = []
    all_labels = []

    # Iterate through the DataLoader
    for inputs, labels in loaders["test"]:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_inputs = (torch.cat(all_inputs, dim=0))
    print(all_inputs.min(), all_inputs.max())
    classes = attack_model(f"test_{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_best_shiny", robust_model, all_inputs,
                           method, all_labels)
    print(f"accuracy: {(all_labels.cuda() == classes.cuda()).sum()}", len(classes))
