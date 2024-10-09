import warnings

import mair
import numpy as np
import pandas as pd
import torch
import yaml
import wandb
from mair import AT, Standard

from datasets import get_loaders
from mnist_scratch import sample_from_dataloader, get_classes, get_confidences
from models import FFNetwork
from pag_robustness.robustness_oracles.Quantitative_Marabou import quantitative_Marabou
from pag_robustness.sample_complexities import complexity
from pag_robustness.temperature_scaled_network import TemperatureScaledNetwork


def training():
    wandb.init(entity="peter-blohm-tu-wien", project="pag_mnist_test")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, train_loader, _, val_loader, test_loader = (
        get_loaders('mnist', scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=True))

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
         "test-acc": robust_model.eval_accuracy(test_loader),
         "test-GN-rob": robust_model.eval_rob_accuracy_gn(test_loader, std=wandb.config.STD),
         "test-FGSM robustness": robust_model.eval_rob_accuracy_fgsm(test_loader, eps=wandb.config.EPS)})

    torch.save(robust_model.state_dict(), f"{wandb.run.dir}.h5")
    torch.save(robust_model.state_dict(), f"../models/{wandb.config.optimization_function}_{wandb.config.seed}.torch")
    artifact = wandb.Artifact(sweep_id, type='model')  # wandb.run.id
    artifact.add_file(f"{wandb.run.dir}.h5")
    wandb.run.log_artifact(artifact, aliases=[wandb.config.optimization_function])


def scaling():
    wandb.init(entity="peter-blohm-tu-wien", project="pag_mnist_test")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, _, scaler_loader, _, _ = (
        get_loaders('mnist', scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=True))

    normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
    robust_model = mair.RobModel(normal_model, n_classes=10)
    robust_model.load_state_dict(torch.load(f"../models/{wandb.config.optimization_function}_{wandb.config.seed}.torch",
                                            weights_only=False))

    # print(normal_model.state_dict())
    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.set_temperature(scaler_loader)
    torch.save(scaled_model.state_dict(), f"{wandb.run.dir}.h5")
    torch.save(scaled_model.state_dict(),
               f"../models/scaled/{wandb.config.optimization_function}_{wandb.config.seed}.torch")
    artifact = wandb.Artifact(sweep_id, type='model')  # wandb.run.id
    artifact.add_file(f"{wandb.run.dir}.h5")
    wandb.run.log_artifact(artifact, aliases=[wandb.config.optimization_function])
    wandb.log({"temperature": scaled_model.temperature.detach().numpy()})


def sampling():
    wandb.init(entity="peter-blohm-tu-wien", project="pag_mnist_test")
    np.random.seed(wandb.config.seed)
    torch.random.manual_seed(wandb.config.seed)

    dim_input, dim_output, _, _, sampler_loader, _ = (
        get_loaders('mnist', scaler_split=wandb.config.scaler_split, sampler_split=wandb.config.sampler_split,
                    batch_size=wandb.config.batch_size, flatten=True))

    normal_model = FFNetwork(dim_input, dim_output, layer_sizes=wandb.config.layer_sizes)
    robust_model = mair.RobModel(normal_model, n_classes=10)
    robust_model.load_state_dict(torch.load(f"../models/{wandb.config.optimization_function}_{wandb.config.seed}.torch",
                                            weights_only=False))
    scaled_model = TemperatureScaledNetwork(robust_model)
    scaled_model.load_state_dict(
        torch.load(f"../models/scaled/{wandb.config.optimization_function}_{wandb.config.seed}.torch",
                   weights_only=False))
    num_points = complexity(wandb.config.epsilon * (1-wandb.config.kappa_max_quantile), wandb.config.delta)
    validation_sample = sample_from_dataloader(loader=sampler_loader, num_points=num_points, std=wandb.config.SAMPLING_GN_STD)
    preds = robust_model(validation_sample)
    print("Sampling done")
    # rob_pgd =

    # classes = get_classes(robust_model(validation_sample))
    robs = quantitative_Marabou(robust_model, points=validation_sample, step_num=wandb.config.MARABOU_STEPS,
                                   max_radius=wandb.config.MARABOU_MAX_RADIUS)
    print("Robustness done")
    confs = get_confidences(preds)
    scaled_confs = get_confidences(preds/scaled_model.temperature)
    classes = get_classes(preds)
    table = wandb.Table(columns=["PGD_robustness", "confidence", "scaled_confidence", "class"])
    df = pd.DataFrame({"PGD_robustness": robs.tolist(),
                       "confidence": confs.tolist(),
                       "scaled_confidence": scaled_confs.tolist(),
                       "class": classes.tolist()})
    # Save the DataFrame to CSV locally
    df.to_csv(f"../results/marabou_{wandb.config.optimization_function}_{wandb.config.seed}.csv", index=False)
    # Add data points to the table
    for i in range(num_points):
        table.add_data(robs[i], confs[i], scaled_confs[i], classes[i])
    wandb.log({"sampling_table": table})
    # wandb.log({"scatter_plot": wandb.plot.scatter(table, "x", "y", title="Scatter Plot Example")})



if __name__ == "__main__":
    with open("sweep.yaml", 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    #
    # wandb.run(entity="peter-blohm-tu-wien", project="pag_mnist_test-orphans", sweep=sweep_configuration)
    sweep_id = wandb.sweep(entity="peter-blohm-tu-wien", project="pag_mnist_sampling_marabou", sweep=sweep_configuration)
    wandb.agent(sweep_id, function=sampling)
