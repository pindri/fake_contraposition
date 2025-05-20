import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, temperature_scale_network, testing

WANDB_ENTITY = "peter-blohm-tu-wien"

# print(torch.cuda.is_available())
if __name__ == "__main__":

    with open("big_net.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_cifar10_training_best", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("cifar10", "vgg11_bn"))
    #
    # sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_cifar10_scaling_best", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: temperature_scale_network("cifar10", "vgg11_bn"))
    #
    # sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_cifar10_sampling_best", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: sampling("cifar10", "vgg11_bn", "pgd"))

    # sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_cifar10_test_best", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: testing("cifar10", "resnet18","pgd"))
