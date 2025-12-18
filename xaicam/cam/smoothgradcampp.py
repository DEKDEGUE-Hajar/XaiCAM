import torch
import torch.nn.functional as F
from .basecam import BaseCAM

class SmoothGradCAMpp(BaseCAM):
    
    """
   SmoothGrad-CAM++:

    Reference:
        Omeiza et al., "Score-CAM: Score-Weighted Visual Explanations for CNNs",
        https://arxiv.org/abs/1908.01224


    """


    def __init__(self, model, target_layer, input_shape=(3,224,224), stdev_spread=0.15, n_samples=20, magnitude=True):
        super().__init__(model, target_layer, input_shape)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude  # if True, use squared gradients in alpha

    def forward(self, x, class_idx=None, retain_graph=False):
        b, c, h, w = x.size()
        device = x.device

        if class_idx is None:
            predicted_class = self.model(x).argmax(dim=1)
        else:
            predicted_class = torch.LongTensor([class_idx]).to(device)

        # Initialize CAM
        cam = 0.0

        # Standard deviation for noise
        stdev = self.stdev_spread * (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        for i in range(self.n_samples):
            # Add noise
            x_noisy = torch.normal(mean=x, std=std_tensor).to(device)
            x_noisy.requires_grad_(True)

            self.model.zero_grad()
            output = self.model(x_noisy)
            score = output[0, predicted_class]
            score.backward(retain_graph=True)

            gradients = self.gradients['value']  # (B, K, H, W)
            activations = self.activations['value']  # (B, K, H, W)

            # Compute 2nd and 3rd order gradients for alpha
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)

            # Alpha coefficient per pixel
            alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).flatten(2).sum(-1)[..., None, None] + 1e-8)
            if self.magnitude:
                gradients = gradients.pow(2)

            # Weights
            weights = (alpha * F.relu(gradients)).flatten(2).sum(-1)  # (B, K)
            weights = weights.view(b, -1, 1, 1)

            # Compute CAM
            cam += (weights * activations).sum(1, keepdim=True).data

        # ReLU and resize to input
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
