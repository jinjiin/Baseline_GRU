import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)


def CrossEntropyLoss(input, target):
    return F.cross_entropy(input, target)
