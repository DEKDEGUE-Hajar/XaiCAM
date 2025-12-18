import torch
import torch.nn.functional as F
from typing import Optional, Union, List
from .basecam import BaseCAM


class LayerCAM(BaseCAM):
    """
    LayerCAM: Exploring Hierarchical Class Activation Maps for Localization

    Reference:
        Jiang et al., LayerCAM: Exploring Hierarchical Class Activation Maps for Localization.
        https://ieeexplore.ieee.org/document/9462463

    """

    def __init__(self, model, target_layer,input_shape=(3, 224, 224), gamma: float = 2.0):
        super().__init__(model, target_layer,input_shape)
        self.gamma = gamma
        self.model.eval()

    def forward(self, x, class_idx: Optional[Union[int, List[int]]] = None, 
                retain_graph: bool = False):

        if x.shape[0] != 1:
            raise ValueError("LayerCAM only supports batch size of 1")

        _, _, h, w = x.size()

        # Forward
        output = self.model(x)

        # Class selection
        if class_idx is None:
            class_idx = [torch.argmax(output).item()]
        elif isinstance(class_idx, int):
            class_idx = [class_idx]

        cams = []

        for idx in class_idx:
            self.model.zero_grad()

            one_hot = torch.zeros_like(output)
            one_hot[0, idx] = 1.0

            output.backward(gradient=one_hot, retain_graph=retain_graph)

            activations = self.activations['value']   # [1, C, H, W]
            gradients = self.gradients['value']       # [1, C, H, W]

            weights = F.relu(gradients)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min)

            cams.append(cam.detach().cpu())

        return cams[0] if len(cams) == 1 else torch.stack(cams)

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)

    @staticmethod
    def fuse_cams(cams_list: List[torch.Tensor], 
                  input_size: Optional[tuple] = None,
                  method: str = 'mean') -> torch.Tensor:

        if not cams_list:
            raise ValueError("cams_list cannot be empty")

        if input_size is None:
            max_h = max(cam.shape[-2] for cam in cams_list)
            max_w = max(cam.shape[-1] for cam in cams_list)
            input_size = (max_h, max_w)

        upsampled = []
        for cam in cams_list:
            if cam.dim() == 2:
                cam = cam.unsqueeze(0).unsqueeze(0)
            elif cam.dim() == 3:
                cam = cam.unsqueeze(1)

            cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
            upsampled.append(cam.squeeze())

        stacked = torch.stack(upsampled, dim=0)

        if method == 'mean':
            fused = stacked.mean(dim=0)
        elif method == 'sum':
            fused = stacked.sum(dim=0)
        elif method == 'max':
            fused = stacked.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        fused -= fused.min()
        if fused.max() > 0:
            fused /= fused.max()

        return fused
