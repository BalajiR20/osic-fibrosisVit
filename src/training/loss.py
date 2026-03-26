import torch
import torch.nn as nn

class LaplaceLoss(nn.Module):
    """
    The metric implemented as a trainable loss.
    Inputs:
        mu    — predicted FVC (ml)
        sigma — predicted uncertainty (ml), clipped to min 70ml
        fvc   — ground truth FVC (ml)
    """
    SIGMA_MIN = 70.0
    DELTA_MAX = 1000.0

    def forward(self, mu: torch.Tensor,
                sigma: torch.Tensor,
                fvc_true: torch.Tensor) -> torch.Tensor:

        sigma = torch.clamp(sigma, min=self.SIGMA_MIN)
        delta = torch.clamp(torch.abs(fvc_true - mu), max=self.DELTA_MAX)

        score = -(delta * (2 ** 0.5) / sigma) - torch.log(sigma * (2 ** 0.5))
        return -score.mean()


def laplace_metric(mu, sigma, fvc_true):
    """
    Numpy version for evaluation — returns the actual competition score
    (higher is better, max 0).
    """
    import numpy as np
    sigma = np.clip(sigma, 70, None)
    delta = np.clip(np.abs(fvc_true - mu), 0, 1000)
    return (-(delta * (2**0.5) / sigma) - np.log(sigma * (2**0.5))).mean()