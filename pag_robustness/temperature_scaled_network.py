import torch


def sample_from_dataloader(loader, num_points, std=0.1):
    # TODO: build in caching
    # TODO: bounds of the data after normalization
    """
    returns a tensor that is sampled from the given dataset with gaussian noise with std
    """
    all_inputs = []
    all_labels = []

    # Iterate through the DataLoader
    for inputs, labels in loader:
        all_inputs.append(inputs)
        all_labels.append(labels)

    # Concatenate all inputs and labels into two big tensors
    dataset = torch.cat(all_inputs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(dataset.shape)
    idx = torch.floor(torch.rand(num_points) * dataset.shape[0]).int()
    dataset_resampled = dataset[idx,]
    print(dataset_resampled.min())
    return (torch.clamp(dataset_resampled + torch.randn_like(dataset_resampled) * std, 0, 1),
            labels_tensor[idx, ])
