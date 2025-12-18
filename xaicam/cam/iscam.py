import torch
import torch.nn.functional as F
from .scorecam import ScoreCAM


class ISCAM(ScoreCAM):
    """
    IS-CAM: Integrated Score-CAM.

    Integrates class scores along a linear path
    from a zero mask to the full activation mask.

    Reference:
        Naidu et al., IS-CAM: Integrated Score-CAM.
        https://arxiv.org/abs/2010.03023
    """

    def __init__(
        self,
        model,
        target_layer,
        num_samples: int = 36,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Trained neural network model.
            target_layer: Target convolutional layer.
            num_samples: Number of integration steps.
            batch_size: Batch size for masked forward passes.
        """
        super().__init__(model, target_layer)
        self.num_samples = num_samples
        self.batch_size = batch_size

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute IS-CAM saliency map.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Kept for API compatibility.

        Returns:
            Normalized saliency map of shape (B, 1, H, W).
        """
        device = x.device
        b, _, h, w = x.size()

        with torch.no_grad():
            logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        activations = self.activations["value"].to(device)
        _, k, _, _ = activations.size()

        cam = torch.zeros((b, 1, h, w), device=device)

        for i in range(k):
            act = activations[:, i:i + 1]
            act = F.interpolate(act, size=(h, w), mode="bilinear", align_corners=False)

            if act.max() == act.min():
                continue

            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            score_sum = torch.zeros(b, device=device)

            for s in range(1, self.num_samples + 1):
                alpha = s / self.num_samples
                masked_inputs = x * (alpha * act)

                for j in range(0, b, self.batch_size):
                    batch_input = masked_inputs[j:j + self.batch_size]
                    out = F.softmax(self.model(batch_input), dim=1)
                    score_sum[j:j + self.batch_size] += out[:, class_idx]

            weight = (score_sum / self.num_samples).view(b, 1, 1, 1)
            cam += weight * act

        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_min != cam_max:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
