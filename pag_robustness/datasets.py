import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


def get_loaders(dataset_name, batch_size=32, val_split=0.2, scaler_split=0.2, sampler_split=0.2, test_split=0.2,
                random_state=42, flatten=False):
    if dataset_name.lower() == 'iris':
        dim_input = 4
        dim_output = 3
        iris = sk_datasets.load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_split + val_split,
                                                            random_state=random_state)
        if val_split > 0.0:
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                            test_size=test_split / (test_split + val_split),
                                                            random_state=random_state)
        else:
            X_val = X_temp
            X_test = X_temp
            y_val = y_temp
            y_test = y_temp

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    elif dataset_name.lower() == 'mnist':
        dim_input = 784
        dim_output = 10
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        if flatten:
            flatten_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.view(-1))  # Flatten.
            ])
            transform = transforms.Compose([transform, flatten_transform])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_size = int((1 - scaler_split - sampler_split) * len(train_dataset))
        sampler_split = int(sampler_split * len(train_dataset))
        scaler_split = len(train_dataset) - train_size - sampler_split
        train_dataset, scaler_dataset, sampler_dataset = random_split(train_dataset, [train_size, scaler_split, sampler_split])
        # Dataloaders.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        scaler_loader = DataLoader(scaler_dataset, batch_size=batch_size, shuffle=False)
        sampler_loader = DataLoader(sampler_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return dim_input, dim_output, train_loader, scaler_loader, sampler_loader, test_loader

    elif dataset_name.lower() == 'cifar10':
        dim_input = 32*32*3
        dim_output = 10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Rotate up to ±10 degrees
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
        if flatten:
            flatten_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.view(-1))  # Flatten.
            ])
            transform = transforms.Compose([transform, flatten_transform])
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ]))
        train_size = int((1 - scaler_split - sampler_split) * len(train_dataset))
        sampler_split = int(sampler_split * len(train_dataset))
        scaler_split = len(train_dataset) - train_size - sampler_split
        train_dataset, scaler_dataset, sampler_dataset = random_split(train_dataset, [train_size, scaler_split, sampler_split])
        # Dataloaders.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        scaler_loader = DataLoader(scaler_dataset, batch_size=batch_size, shuffle=False)
        sampler_loader = DataLoader(sampler_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return dim_input, dim_output, train_loader, scaler_loader, sampler_loader, test_loader

    elif dataset_name.lower() == 'susy':  # Ignores the test split, as there is a designated test set.
        dim_input = 8  #?
        dim_output = 2
        full_dataset = torch.tensor(np.load("./datasets/susy.npy"))
        X, y = full_dataset[:4500000, 1:], full_dataset[:4500000, 0].long()
        X_test, y_test = full_dataset[4500000:, 1:], full_dataset[4500000:, 0].long()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split,
                                                          random_state=random_state)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    else:
        raise ValueError("Dataset not supported. Please choose 'iris' or 'mnist'.")

    # Dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dim_input, dim_output, train_loader, val_loader, test_loader
