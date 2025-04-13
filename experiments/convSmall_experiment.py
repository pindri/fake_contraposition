import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, temperature_scale_network, testing

WANDB_ENTITY = "peter-blohm-tu-wien"

# print(torch.cuda.is_available())
if __name__ == "__main__":

    with open("mnist_marabou.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="convBig_lirpa_mnist_mnist_train_normal2",
    #                        sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: train_network("mnist", "convBig"))
    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_scale_normal2",
    #                        sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: temperature_scale_network("mnist", "convBig"))
    # sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="convBig_lirpa_mnist_mnist_sample_normal_marabou_test",
    #                        sweep=sweep_configuration)

    # sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="convBig_lirpa_mnist_sampling_best", sweep=sweep_configuration)
    # wandb.agent(sweep_id, function=lambda: sampling("mnist", "convBig","lirpa"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="convBig_lirpa_mnist_test_best", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("mnist", "convBig", "lirpa"))
