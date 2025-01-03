import yaml
import wandb

from network_training import train_network, sampling, temperature_scale_network, testing

if __name__ == "__main__":
    with open("mnist.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    #
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_train_normal", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: train_network("mnist", "feed_forward"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_scale_normal", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=lambda: temperature_scale_network("mnist", "feed_forward"))
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_sample_train_normal",
                           sweep=sweep_configuration)
    wandb.agent(sweep_id, function=sampling)
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_test_normal", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=testing)
