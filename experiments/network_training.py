import gc
import os
import time

import onnx
import torch
from onnx2pytorch import ConvertModel

from pag_robustness.readONNXModels.bigConvONNX import BigConvOnnx, load_weights_from_onnx_state_dict
from pag_robustness.robustness_oracles.Quantitative_LiRPA import quantitative_lirpa

if not torch.cuda.is_available():
    raise Exception("cuda is not available")
import mair
import numpy as np
import pandas as pd
from torch import Tensor

import wandb
from mair import Standard, RobModel, TRADES

from pag_robustness.datasets import get_loaders
from network_utils import get_classes, get_confidences
from pag_robustness.models import FFNetwork
from pag_robustness.robustness_oracles.Quantitative_Marabou import quantitative_Marabou
from pag_robustness.robustness_oracles.Quantitative_PDG import Quantitative_PGD
from pag_robustness.sample_complexities import complexity
from pag_robustness.temperature_scaled_network import TemperatureScaledNetwork, sample_from_dataloader


def get_base_model(dim_input: int, dim_output: int, network_type: str) -> RobModel:
    match network_type:
        case "feed_forward":
            normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
        case "resnet18":
            # normal_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            normal_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
        case "vgg11_bn":
            # normal_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            normal_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg11_bn', pretrained=True)
        case "convSmall":
            onnx_model_path = "/home/pblohm/pag/fake_contraposition/rob/onnx_models/mnist_convSmallRELU__Point.onnx"
            onnx_model = onnx.load(onnx_model_path)
            normal_model = load_weights_from_onnx_state_dict(
                ConvertModel(onnx_model,experimental=True).state_dict(),
                BigConvOnnx())
        case "convBig":
            onnx_model_path = "/home/pblohm/pag/fake_contraposition/rob/onnx_models/mnist_convBigRELU__DiffAI.onnx"
            onnx_model = onnx.load(onnx_model_path)
            normal_model = load_weights_from_onnx_state_dict(
                ConvertModel(onnx_model,experimental=True).state_dict(),
                BigConvOnnx())
        case _:
            raise "unknown network type"
    robust_model = mair.RobModel(normal_model, n_classes=dim_output)
    if torch.cuda.is_available():
        robust_model = robust_model.cuda()
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
    torch.backends.cudnn.deterministic = True  # Enforce deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable automatic optimizations that could introduce randomness

    dim_input, dim_output, train_loader, scaler_loader, val_loader, test_loader = (
        get_loaders(dataset, scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=dataset != "cifar10"))
    robust_model = get_base_model(dim_input, dim_output, network_type)
    return name, robust_model, {"train": train_loader, "scale": scaler_loader, "val": val_loader, "test": test_loader}


def train_network(dataset: str, network_type: str):
    name, robust_model, loaders = setup_and_get_data(dataset, network_type)

    # adversarial training or standard
    if wandb.config.optimization_function == "AT":
        trainer = TRADES(robust_model, eps=wandb.config.EPS,
                         alpha=wandb.config.ALPHA,
                         steps=wandb.config.STEPS, beta=wandb.config.robust_beta)
    else:
        trainer = Standard(robust_model)

    trainer.record_rob(loaders["train"], loaders["val"], eps=wandb.config.EPS, alpha=wandb.config.ALPHA,
                       steps=wandb.config.STEPS, std=wandb.config.STD)
    trainer.setup(optimizer=f"SGD(lr={wandb.config.lr}, "
                            f"    momentum={wandb.config.momentum}, "
                            f"    weight_decay={wandb.config.weight_decay})",
                  scheduler="Step(milestones=[10, 15], gamma=0.1)",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=wandb.config.n_epochs
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
    print(name)
    robust_model.load_state_dict(
        torch.load(f'../rob/{name}/best.pth', weights_only=False, map_location="cpu")["rmodel"])
    robust_model = robust_model.cuda()
    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model = scaled_model.cuda()
    scaled_model.set_temperature(loaders["scale"], reg=wandb.config.temp_scaling_regularization)

    os.makedirs(f'../rob/scaled/{name}', exist_ok=True)
    torch.save(scaled_model.state_dict(), f'../rob/scaled/{name}/best.pth')


def attack_model(name: str, model: RobModel, data: Tensor, method: str, scaling_temp: int = 1, labels: Tensor = None):
    start_time = time.time()
    batch_size = 16186//2
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
    confs = get_confidences(preds)
    print(confs[1])
    match method:
        case "pgd":
            rob_pgd = Quantitative_PGD(model, eps=wandb.config.PGD_EPS, alpha=wandb.config.PGD_ALPHA,
                                       steps=wandb.config.PGD_STEPS, random_start=False, device="cuda:0")

            classes = get_classes(preds)
            classes.requires_grad = False
            steps, distances = rob_pgd.forward(data, classes)
        case "marabou":
            distances = quantitative_Marabou(model.cpu(), points=data,
                                             step_num=wandb.config.MARABOU_STEPS,
                                             max_radius=wandb.config.MARABOU_MAX_RADIUS)
            steps = torch.zeros_like(distances)
            classes = get_classes(preds)
        case "lirpa":
            distances = quantitative_lirpa(model, points=data,
                                             step_num=wandb.config.MARABOU_STEPS,
                                             max_radius=wandb.config.MARABOU_MAX_RADIUS,
                                             classes = get_classes(preds))
            steps = torch.zeros_like(distances)
            classes = get_classes(preds)
        case _:
            raise "unknown robustness oracle"
    print("Robustness done")
    confs = get_confidences(preds)
    scaled_confs = get_confidences(preds / scaling_temp.cpu())
    if labels is None:
        labels = classes
    df = pd.DataFrame({f"{method}_robustness_steps": steps.tolist(),
                       f"{method}_robustness_distances": distances.tolist(),
                       "confidence": confs.tolist(),
                       "scaled_confidence": scaled_confs.tolist(),
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
        torch.load(f'../rob/{name}/best.pth', weights_only=False, map_location="cpu")["rmodel"])

    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.load_state_dict(torch.load(f'../rob/scaled/{name}/best.pth', weights_only=False, map_location="cpu"))

    num_points = complexity(wandb.config.epsilon * (1 - wandb.config.kappa_max_quantile), wandb.config.delta)
    validation_sample = sample_from_dataloader(loader=loaders["val"], num_points=num_points,
                                               std=wandb.config.SAMPLING_GN_STD)

    for param in robust_model.parameters():
        param.requires_grad = False
    robust_model.eval()

    print("val", validation_sample[0].min(), validation_sample[0].max())
    attack_model(f"{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_best", robust_model, validation_sample[0], method,
                 scaled_model.temperature)
    del robust_model
    del scaled_model
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
        torch.load(f'../rob/{name}/best.pth', weights_only=False, map_location="cpu")["rmodel"])
    robust_model.cuda()
    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.load_state_dict(torch.load(f'../rob/scaled/{name}/best.pth', weights_only=False, map_location="cpu"))

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
    classes = attack_model(f"validation_set_{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_best", robust_model,
                           all_inputs.clone(), method, scaled_model.temperature, all_labels.clone())
    print(f"accuracy: {(all_labels.cuda() == classes.cuda()).sum()}")

    all_inputs = []
    all_labels = []

    # Iterate through the DataLoader
    for inputs, labels in loaders["test"]:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_inputs = (torch.cat(all_inputs, dim=0))
    print(all_inputs.min(), all_inputs.max())
    classes = attack_model(f"test_{name}_{method}_std{wandb.config.SAMPLING_GN_STD}_best", robust_model, all_inputs,
                           method, scaled_model.temperature, all_labels)
    print(f"accuracy: {(all_labels.cuda() == classes.cuda()).sum()}")
