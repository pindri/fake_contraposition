import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("what the heeeell")
from network_training import train_network, sampling, temperature_scale_network, testing
#


if __name__ == "__main__":
    with open("mnist.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_train_normal", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("mnist", "feed_forward"))
    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_scale_normal", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: temperature_scale_network("mnist", "feed_forward"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_sample_train_normal",
                           sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("mnist", "feed_forward", "pgd"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_test_normal", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("mnist", "feed_forward", "pgd"))
