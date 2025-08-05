import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, temperature_scale_network, testing
#


if __name__ == "__main__":
    with open("mnist_marabou.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_train_normal2", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("mnist", "feed_forward"))

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_sample_normal_marabou_test", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("mnist", "feed_forward", "marabou"))

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_test_normal_marabou_test", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("mnist", "feed_forward", "marabou"))
