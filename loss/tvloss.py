import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """
    Total variation loss/regularization.
    """
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, yhat, y):
        bsize, chan, height, width = y.size()
        dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        dyhat = torch.abs(yhat[:, :, 1:, :] - yhat[:, :, :-1, :])
        error = torch.norm(dy - dyhat, 1)
        return error / height