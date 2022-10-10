import torch
from torchvision.utils import save_image


def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(
    batch,
    column,
    path,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    device="cpu",
):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct
