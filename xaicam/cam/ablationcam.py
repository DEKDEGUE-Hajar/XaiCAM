import torch
import torch.nn.functional as F
from typing import Optional, Callable
from .basecam import BaseCAM


class AblationCAM(BaseCAM):
    """
    Ablation-CAM: Gradient-free Visual Explanations via Channel Ablation.

    Reference:
        Desai et al., WACV 2020
        https://ieeexplore.ieee.org/abstract/document/9093360
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module | None = None,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        use_positive_scores: bool = True,
    ):
        super().__init__(model, target_layer, input_shape)
        self.use_positive_scores = use_positive_scores

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (1, C, H, W). Batch size must be 1.
            class_idx: Target class index (uses predicted class if None).

        Returns:
            Normalized CAM tensor (1, 1, H, W).
        """

        if x.shape[0] != 1:
            raise ValueError("AblationCAM only supports batch size = 1")

        device = x.device
        _, _, h, w = x.shape

        # --------------------------------------------------
        # Original prediction
        # --------------------------------------------------
        with torch.no_grad():
            output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        original_score = output[0, class_idx].item()

        # --------------------------------------------------
        # Get target layer activations
        # --------------------------------------------------
        _ = self.model(x)  # populate hooks
        activations = self.activations["value"].to(device)  # [1, K, Hc, Wc]
        k = activations.size(1)

        importance_scores = torch.zeros(k, device=device)

        # --------------------------------------------------
        # Channel ablation loop (core Ablation-CAM)
        # --------------------------------------------------
        for i in range(k):
            ablated_acts = activations.clone()
            ablated_acts[:, i] = 0  # baseline = zero (as in paper)

            # Upsample ablated activation map
            ablated_mask = F.interpolate(
                ablated_acts[:, i:i+1],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

            # Mask input
            masked_input = x * ablated_mask

            with torch.no_grad():
                ablated_output = self.model(masked_input)
                ablated_score = ablated_output[0, class_idx].item()

            score_drop = original_score - ablated_score

            if self.use_positive_scores:
                score_drop = max(0.0, score_drop)

            importance_scores[i] = score_drop

        # --------------------------------------------------
        # Weighted sum of activations
        # --------------------------------------------------
        cam = torch.sum(
            importance_scores.view(1, -1, 1, 1) * activations,
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu()

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
