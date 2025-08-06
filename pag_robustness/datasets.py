import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, Subset
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


def kfold_indices(num_samples: int,
                  labels: np.ndarray,
                  k_folds: int,
                  fold_idx: int,
                  random_state: int = 42):
    labels = np.asarray(labels)
    assert len(labels) == num_samples, "num_samples and labels length differ"
    assert 0 <= fold_idx < k_folds, "fold_idx out of range"

    skf = StratifiedKFold(n_splits=k_folds,
                          shuffle=True,
                          random_state=random_state)

    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_samples), labels)):
        if i == fold_idx:
            return train_idx, val_idx


def get_kfold_loaders(dataset_name, batch_size=32, split_num=5, split_index=0, random_state=42, flatten=False):
    if dataset_name.lower() == 'mnist':
        dim_input = 784
        dim_output = 10
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize((0.1307,), (0.3081,))
                                        ])

        if flatten:
            flatten_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.view(-1))  # Flatten.
            ])
            transform = transforms.Compose([transform, flatten_transform])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name.lower() == 'cifar10':
        dim_input = 32 * 32 * 3
        dim_output = 10
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            #                      std=[0.2470, 0.2435, 0.2616]),
        ])
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                                        ]))
    else:
        raise ValueError("Dataset not supported. Please choose 'mnist' or 'cifar10'.")

    train_idx, sampler_idx = kfold_indices(len(train_dataset), train_dataset.targets, split_num, split_index,
                                           random_state)
    sampler_dataset = Subset(train_dataset, sampler_idx)
    train_dataset = Subset(train_dataset, train_idx)
    # Dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sampler_loader = DataLoader(sampler_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return dim_input, dim_output, train_loader, sampler_loader, test_loader
