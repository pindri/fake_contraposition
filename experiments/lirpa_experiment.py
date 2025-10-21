import yaml
import wandb
import torch
if not torch.cuda.is_available():
    raise Exception("Cuda is not available")
from network_training import train_network, sampling, testing

WANDB_ENTITY = "<WANDB_ENTITY>"

if __name__ == "__main__":
    with open("mnist_lirpa.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="PAG", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("mnist", "feed_forward"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="PAG", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: sampling("mnist", "feed_forward","lirpa"))

    sweep_id = wandb.sweep(entity=WANDB_ENTITY, project="PAG", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: testing("mnist", "feed_forward","lirpa"))
