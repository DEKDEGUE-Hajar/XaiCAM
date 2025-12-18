import torch
from ..utils import convert_to_gray


class SmoothIntGrad(object):
    
    def __init__(self, model, stdev_spread=0.15, n_steps=20, magnitude=True):
        super(SmoothIntGrad, self).__init__()
        self.stdev_spread = stdev_spread
        self.n_steps = n_steps
        self.magnitude = magnitude

        self.model = model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        x = x.to(self.device)
        x.requires_grad_(True)

        if x_baseline is None:
            x_baseline = torch.zeros_like(x).to(self.device)
        else:
            x_baseline = x_baseline.to(self.device)

        assert x_baseline.size() == x.size()

        saliency_map = torch.zeros_like(x).to(self.device)

        x_diff = x - x_baseline
        stdev = self.stdev_spread / (x_diff.max() - x_diff.min() + 1e-8)

        for alpha in torch.linspace(0., 1., self.n_steps).to(self.device):
            x_step = x_baseline + alpha * x_diff
            x_step = x_step.clone().detach().requires_grad_(True).to(x.device)  
            
            noise = torch.normal(
                mean=torch.zeros_like(x_step),
                std=stdev
            ).to(self.device)

            x_step_plus_noise = x_step + noise

            logit = self.model(x_step_plus_noise)

            if class_idx is None:
                predicted_class = logit.max(1)[-1]
                score = logit[:, predicted_class].squeeze()
            else:
                predicted_class = torch.LongTensor([class_idx]).to(self.device)
                score = logit[:, class_idx].squeeze()

            self.model.zero_grad()
            if x_step_plus_noise.grad is not None:
                x_step_plus_noise.grad.zero_()

            score.backward(retain_graph=retain_graph)

            saliency_map += x_step_plus_noise.grad

        saliency_map = saliency_map / self.n_steps
        saliency_map = convert_to_gray(saliency_map)  # [1, 1, H, W]

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (
            saliency_map_max - saliency_map_min + 1e-8
        )

        return saliency_map

    def __call__(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        return self.forward(x, x_baseline, class_idx, retain_graph)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass