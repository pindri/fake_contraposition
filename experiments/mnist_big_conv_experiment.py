import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, testing

WANDB_ENTITY = "peter-blohm-tu-wien"

if __name__ == "__main__":
    with open("mnist_big_conf.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)


    # sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_mnist_training_best_big_conv", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: train_network("mnist", "convBig"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_mnist_sampling_best_big_conv", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("mnist", "convBig","pgd"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="pag_mnist_test_best_big_conv_norm", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("mnist", "convBig","pgd"))