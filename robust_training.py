import mair
import torch
from mair import Standard
from matplotlib import pyplot as plt

from datasets import get_loaders
from models import FFNetwork
from pag_robustness.robustness_oracles import Quantitative_PGD

train_loader, val_loader, test_loader = get_loaders('mnist', val_split=0.0, batch_size=20048, flatten=True)
input_dim = 784
output_dim = 10

model = FFNetwork(input_dim, output_dim, layer_sizes=[1000, 20])
print(len(train_loader))
# Variables.
EPS = 0.1
ALPHA = 0.02
STEPS = 50
STD = 0.1
n_epochs = 1

# Teacher training.
rmodel = mair.RobModel(model, n_classes=output_dim)  # .cuda
# trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
trainer = Standard(rmodel)
# trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=STD)
trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
              scheduler="Step(milestones=[100, 150], gamma=0.1)",
              scheduler_type="Epoch",
              minimizer=None,  # or "AWP(rho=5e-3)",
              n_epochs=n_epochs
              )
trainer.fit(train_loader=train_loader,
            n_epochs=n_epochs,
            save_path='../rob/',
            save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
            save_type="Epoch",
            save_overwrite=True,
            record_type="Epoch"
            )


def evaluate_and_print(model, label, std, eps):
    print("\n")
    print(f"Model: {label}")
    print(f"Clean accuracy: {model.eval_accuracy(val_loader):.2f}")  # clean accuracy
    print(f"GN robustness: {model.eval_rob_accuracy_gn(val_loader, std=std):.2f}")  # gaussian noise accuracy
    print(f"FGSM robustness: {model.eval_rob_accuracy_fgsm(val_loader, eps=eps):.2f}")  # FGSM accuracy
    print(
        f"PGD robustness: {model.eval_rob_accuracy_pgd(val_loader, eps=0.5, alpha=0.01, steps=100):.2f}")  # FGSM accuracy
    quant_pgd = Quantitative_PGD(model, eps=1, alpha=0.005, steps=200, random_start=False)
    print(f"quantitative robustness:  {quant_pgd(val_loader)}")
    print("second try")
    quant_pgd = Quantitative_PGD(model, eps=1, alpha=0.001, steps=200, random_start=False)
    print(f"quantitative robustness:  {quant_pgd(val_loader)}")
    print("\n")
    robs = []
    Xs = torch.rand((900000, 4))
    # for (X, y) in val_loader:
    #     print(torch.rand((300, 4)))
        # print(X)
    pgd_robs = quant_pgd.forward(Xs, model(Xs).argmax(dim=1)).numpy()
    # robs.append(quantitative_Marabou(model, 1, 0.5, Xs))
    # print(torch.cat(robs, dim=0))
    print(torch.softmax(model(Xs), dim=1).max(dim=1)[0].detach().numpy())
    plt.scatter(
        torch.softmax(model(Xs), dim=1).max(dim=1)[0].detach().numpy(),
        pgd_robs, c=torch.softmax(model(Xs), dim=1).max(dim=1)[1].detach().numpy(), alpha=0.5)
    plt.xlabel("Confidence Score")
    plt.ylabel("Robustness")
    plt.show()

# TODO: ffnetwork class, with something something for temp scaling (OR, in a decorator), output with/without sfmax
# TODO: wandb stuff
# TODO: add cifar to datasets (+ make data dimension thing automatic)


evaluate_and_print(rmodel, "teacher", std=STD, eps=EPS)
