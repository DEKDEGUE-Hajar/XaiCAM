# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import convert_to_gray


class GuidedBackProp:
    """
    Guided Backpropagation for visualizing input gradients.

    Modifies ReLU backward behavior to suppress negative gradients,
    highlighting input features that positively influence the prediction.

    Reference:
        Springenberg et al., Striving for Simplicity:
        The All Convolutional Net. ICLR 2015.
        https://arxiv.org/abs/1412.6806
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Trained neural network model.
        """
        self.model = model.eval()

        # Register backward hooks on ReLU layers
        for _, module in self.model.named_modules():
            module.register_backward_hook(self._guided_relu_backward)

    @staticmethod
    def _guided_relu_backward(module, grad_in, grad_out):
        """
        Backward hook for ReLU layers.
        Suppresses negative gradients.
        """
        if isinstance(module, nn.ReLU):
            return (F.relu(grad_in[0]),)

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute Guided Backpropagation saliency map.

        Args:
            x: Input image tensor of shape (1, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Whether to retain the computation graph.

        Returns:
            Normalized saliency map of shape (1, 1, H, W).
        """
        x = x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            score = logits.gather(1, class_idx.view(-1, 1)).squeeze()
        else:
            score = logits[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        saliency_map = x.grad  # (1, C, H, W)
        saliency_map = convert_to_gray(saliency_map.detach().cpu())

        saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
