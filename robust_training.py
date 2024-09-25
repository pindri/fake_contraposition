import mair
from mair import Standard
from mair.defenses import AT

from datasets import get_loaders
from models import FFNetwork

train_loader, val_loader, test_loader = get_loaders('iris', val_split=0.0, batch_size=16)
input_dim = 4
output_dim = 3

model = FFNetwork(input_dim, output_dim, layer_sizes=[10, 10])

# Variables.
EPS = 0.1
ALPHA = 0.1
STEPS = 10
STD = 0.1
n_epochs = 10

# Teacher training.
rmodel = mair.RobModel(model, n_classes=output_dim)  # .cuda
trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
# trainer = Standard(model)
trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=STD)
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
    print("\n")


evaluate_and_print(rmodel, "teacher", std=STD, eps=EPS)
