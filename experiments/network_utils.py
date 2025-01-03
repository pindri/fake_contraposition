import torch


def get_confidences(logits: torch.tensor) -> torch.tensor:
    return torch.max(torch.softmax(logits, dim=-1), dim=-1)[0]


def get_classes(logits: torch.tensor) -> torch.tensor:
    return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

