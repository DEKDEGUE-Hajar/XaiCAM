# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..utils import convert_to_gray


class IntegratedGradients:
    """
    Integrated Gradients for attributing model predictions to input features.

    Computes the average gradient along the straight-line path
    from a baseline input to the original input.

    Reference:
        Sundararajan et al., Axiomatic Attribution for Deep Networks.
        ICML 2017. https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module, n_steps: int = 20):
        """
        Args:
            model: Trained neural network model.
            n_steps: Number of interpolation steps.
        """
        self.model = model.eval()
        self.n_steps = n_steps

    def forward(
        self,
        x: torch.Tensor,
        x_baseline: torch.Tensor | None = None,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients saliency map.

        Args:
            x: Input tensor of shape (1, C, H, W).
            x_baseline: Baseline tensor (same shape as x). Defaults to zero.
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Whether to retain the computation graph.

        Returns:
            Normalized saliency map of shape (1, 1, H, W).
        """
        device = x.device
        if x_baseline is None:
            x_baseline = torch.zeros_like(x)
        else:
            x_baseline = x_baseline.to(device)

        assert x_baseline.size() == x.size()

        saliency_map = torch.zeros_like(x)
        x_diff = x - x_baseline

        for alpha in torch.linspace(0.0, 1.0, self.n_steps, device=device):
            x_step = (x_baseline + alpha * x_diff).detach().requires_grad_(True)

            logits = self.model(x_step)

            if class_idx is None:
                class_idx_ = logits.argmax(dim=1)
                score = logits.gather(1, class_idx_.view(-1, 1)).squeeze()
            else:
                score = logits[:, class_idx].squeeze()

            self.model.zero_grad()
            score.backward(retain_graph=retain_graph)

            saliency_map += x_step.grad

        saliency_map /= self.n_steps
        saliency_map = convert_to_gray(saliency_map.detach())

        saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency_map

    def __call__(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        return self.forward(x, x_baseline, class_idx, retain_graph)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
