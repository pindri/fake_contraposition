import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, testing

WANDB_ENTITY = "peter-blohm-tu-wien"

# print(torch.cuda.is_available())
if __name__ == "__main__":

    with open("cifar10_new.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_camera_ready_cifar10_train", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("cifar10", "resnet20"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_camera_ready_cifar10_sampling", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("cifar10", "resnet20","pgd"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_camera_ready_cifar10_test", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("cifar10", "resnet20","pgd"))
