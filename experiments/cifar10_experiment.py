import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("what the heeeell")
from network_training import train_network, sampling, temperature_scale_network, testing


# print(torch.cuda.is_available())
if __name__ == "__main__":

    with open("cifar10.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    #
    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_cifar10_train_normal", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: train_network("cifar10", "resnet18"))
    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_cifar10_scale_normal", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: temperature_scale_network("cifar10", "resnet18"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_cifar10_sample_train_normal2_last",
                             sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("cifar10", "resnet18","pgd"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_cifar10_test_normal2_last", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("cifar10", "resnet18","pgd"))
