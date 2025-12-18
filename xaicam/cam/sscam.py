import torch
import torch.nn.functional as F
from .scorecam import ScoreCAM


class SSCAM(ScoreCAM):
    """
    SS-CAM: Smoothed Score-CAM.

    Applies Gaussian noise to activation maps and
    averages Score-CAM weights over multiple samples.
    """

    def __init__(
        self,
        model,
        target_layer=None,
        num_samples: int = 35,
        std: float = 2.0,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Trained neural network model.
            target_layer: Target convolutional layer.
            num_samples: Number of noise samples per activation.
            std: Standard deviation of Gaussian noise.
            batch_size: Number of activation maps processed at once.
        """
        super().__init__(model, target_layer, batch_size=batch_size)
        self.num_samples = num_samples
        self.std = std

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute SS-CAM saliency map.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Kept for API compatibility.

        Returns:
            Normalized saliency map of shape (B, 1, H, W).
        """
        device = x.device
        b, _, h, w = x.size()

        logits = self.model(x)
        if class_idx is None:
            predicted_class = logits.argmax(dim=1)
        else:
            predicted_class = torch.full((b,), class_idx, device=device, dtype=torch.long)

        activations = self.activations["value"].to(device)
        k = activations.size(1)

        cam = torch.zeros((b, 1, h, w), device=device)

        # Process activation maps in batches
        for i in range(0, k, self.batch_size):
            batch_acts = activations[:, i:i + self.batch_size]
            batch_k = batch_acts.size(1)

            batch_acts = F.interpolate(
                batch_acts, size=(h, w), mode="bilinear", align_corners=False
            )

            # Normalize activation maps
            acts_min = batch_acts.view(b, batch_k, -1).min(dim=2)[0].view(b, batch_k, 1, 1)
            acts_max = batch_acts.view(b, batch_k, -1).max(dim=2)[0].view(b, batch_k, 1, 1)
            valid = acts_max != acts_min
            batch_acts = (batch_acts - acts_min) / (acts_max - acts_min + 1e-8)
            batch_acts = batch_acts * valid

            with torch.no_grad():
                for j in range(batch_k):
                    act = batch_acts[:, j:j + 1]
                    score_sum = torch.zeros(b, device=device)

                    for _ in range(self.num_samples):
                        noise = torch.normal(
                            mean=0.0,
                            std=self.std,
                            size=act.size(),
                            device=device,
                        )
                        noisy_mask = torch.clamp(act + noise, 0.0, 1.0)
                        output = F.softmax(self.model(x * noisy_mask), dim=1)
                        score_sum += output[torch.arange(b), predicted_class]

                    weight = (score_sum / self.num_samples).view(b, 1, 1, 1)
                    cam += weight * act

        cam = F.relu(cam)
        cam_min = cam.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
