import warnings

import mair
import numpy as np
import torch
import yaml
import wandb
from mair import AT, Standard

from datasets import get_loaders
from models import FFNetwork


def training():
    wandb.init(entity="peter-blohm-tu-wien", project="pag_mnist_test")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, train_loader, val_loader, test_loader = get_loaders('mnist', val_split=0.2,
                                                                               batch_size=wandb.config.batch_size)

    normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)

    robust_model = mair.RobModel(normal_model, n_classes=dim_output)
    if torch.cuda.is_available():
        robust_model = robust_model.cuda()
    if wandb.config.optimization_function == "AT":
        trainer = AT(robust_model, eps=wandb.config.EPS,
                     alpha=wandb.config.ALPHA,
                     steps=wandb.config.STEPS)
    else:
        trainer = Standard(robust_model)
    trainer.record_rob(train_loader, val_loader, eps=wandb.config.EPS, alpha=wandb.config.ALPHA,
                       steps=wandb.config.STEPS, std=wandb.config.STD)
    trainer.setup(optimizer=f"SGD(lr={wandb.config.lr}, momentum={wandb.config.momentum})",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)",
                  scheduler_type="Epoch",
                  minimizer=None,  # or "AWP(rho=5e-3)",
                  n_epochs=wandb.config.n_epochs,
                  )
    trainer.fit(train_loader=train_loader,
                n_epochs=wandb.config.n_epochs,
                save_path='../rob/',
                save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
                save_type="Epoch",
                save_overwrite=True,
                record_type="Epoch"
                )
    wandb.log(
        {"train-acc": robust_model.eval_accuracy(train_loader),
         "train-GN-rob": robust_model.eval_rob_accuracy_gn(train_loader, std=wandb.config.STD),
         "train-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(train_loader, eps=wandb.config.EPS),
         "val-acc": robust_model.eval_accuracy(val_loader),
         "val-GN-rob": robust_model.eval_rob_accuracy_gn(val_loader, std=wandb.config.STD),
         "val-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(val_loader, eps=wandb.config.EPS),
         "test-acc": robust_model.eval_accuracy(test_loader),
         "test-GN-rob": robust_model.eval_rob_accuracy_gn(test_loader, std=wandb.config.STD),
         "test-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(test_loader, eps=wandb.config.EPS)})
    torch.save(robust_model.state_dict(), f"{wandb.run.dir}.h5")
    artifact = wandb.Artifact(sweep_id, type='model')  # wandb.run.id
    artifact.add_file(f"{wandb.run.dir}.h5")
    wandb.run.log_artifact(artifact, aliases=[wandb.run.id])


if __name__ == "__main__":
    with open("mnist_experiment/sweep.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)

    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_test-orphans", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=training)
