
**XaiCAM** is a Python library for **explainable AI (XAI) in computer vision** using **Class Activation Maps (CAMs)**.  
It allows you to visualize which regions of an image contribute most to a model's predictions and evaluate CAM quality with metrics like Average Drop/Increase in confidence, Insertion, and Deletion.

---

## Quick Start: Single CAM Workflow

### Extracting the Class Activation Map

After initializing a CAM extractor, inference works as usual. The extractor automatically hooks the necessary layers and collects relevant information.

```python
import torch
from torchvision.models import vgg19
from torchvision.transforms.functional import to_tensor, resize, normalize
from PIL import Image
from XaiVisionCAM import GradCAM

# Configuration
IMAGE_PATH = "path/to/your/image.png"
model = vgg19(pretrained=True).eval()

# Image Preparation
img = Image.open(IMAGE_PATH).convert("RGB")


input_tensor = normalize(to_tensor(resize(img, (224, 224))) ,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)

# target_layer is optional; the package attempts to find the last conv layer if None
with GradCAM(model, target_layer=None) as cam_extractor:
    output = model(input_tensor)
    cam = cam_extractor(input_tensor, class_idx=output.argmax(dim=1).item())
```
*Please note that by default, the CAM is extracted from the last convolutional layer before any spatial reduction. To target a different layer, simply specify it using the target_layer argument when creating the CAM instance*

### Evaluate with Average Drop/Increase in Confidence

```python
from XaiVisionCAM.metrics import AvgDropConf

metric = AvgDropConf(model, cam, input_tensor, class_idx=predicted_class)
avgdrop_summary = metric.summary()

# Example output: {'avg_drop': 0.15, 'avg_increase': 0.05}
print(f"AvgDrop/Increase: {avgdrop_summary}")
```

### Evaluate with Insertion/Deletion
Insertion and Deletion are fidelity metrics that quantify how quickly the model's output changes as pixels are revealed (Insertion) or removed (Deletion) based on the saliency map order.

```python
from XaiVisionCAM.metrics import AvgDropConf

metric = AvgDropConf(model, cam, input_tensor, class_idx=predicted_class)
avgdrop_summary = metric.summary()

# Example output: {'avg_drop': 0.15, 'avg_increase': 0.05}
print(f"AvgDrop/Increase: {avgdrop_summary}")
```


```python
from XaiVisionCAM.metrics import InsertionDeletion

# Insertion
insdel_ins = InsertionDeletion(model=model, mode="Insertion")
# Returns a dictionary of results, including confidence steps and AUC
ins_result = insdel_ins.compute(image=input_tensor, heatmap=cam, return_steps=True)

# Deletion
insdel_del = InsertionDeletion(model=model, mode="Deletion")
del_result = insdel_del.compute(image=input_tensor, heatmap=cam, return_steps=True)
```
### Plot Insertion and Deletion Curves

The results can be visualized to show the confidence vs. percentage of pixels masked/revealed.

```python
import os

# Plot Insertion Curve
InsertionDeletion.plot_curves(
    results=ins_result,
    mode="Insertion",
    image=img_tensor.unsqueeze(0), # Image for overlay on the plot
    cam=cam,                       # CAM for overlay on the plot
    alpha=0.6,
    save_path=os.path.join("outputs", "gradcam_insertion.png")
)

# Plot Deletion Curve
InsertionDeletion.plot_curves(
    results=del_result,
    mode="Deletion",
    image=img_tensor.unsqueeze(0),
    cam=cam,
    alpha=0.6,
    save_path=os.path.join("outputs", "gradcam_deletion.png")
)

```


## Comparative Analysis (Multiple CAMs)


The package supports easy comparison of multiple CAM methods on the same image.

### Visualize Multiple CAMs
Use the visualize_cam utility to display all generated heatmaps side-by-side.

```python
from XaiVisionCAM.utils import visualize_cam

# A dictionary mapping CAM names to their generated Tensor
cams = {
    "GradCAM": cam_gradcam,
    "GradCAMpp": cam_gradcampp,
    # ... and so on
}

visualize_cam(
    img=img_tensor.unsqueeze(0),
    cams=cams,
    nrow=4,
    save_path=os.path.join("outputs", "all_cams.png")
)

```

![All CAMs](outputs\all_cams.png)

### Plot Comparative Metric Curves
To compare the performance of multiple CAMs using Insertion/Deletion, pass a dictionary of results to plot_curves

```python
# Assuming insdel_results is a nested dictionary mapping CAM_NAME -> {'Insertion': results, 'Deletion': results}

# Insertion curves comparison
all_ins_results = {name: res["Insertion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_ins_results,
    mode="Insertion",
    title="Insertion Curves Comparison",
    save_path=os.path.join("outputs", "all_cams_insertion.png")
)

```
![All CAMs](outputs\all_cams_deletion.png)

```python
# Deletion curves comparison
all_del_results = {name: res["Deletion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_del_results,
    mode="Deletion",
    title="Deletion Curves Comparison",
    save_path=os.path.join("outputs", "all_cams_deletion.png")
)

```
![All CAMs](outputs\all_cams_deletion.png)


## CAM Zoo

This project is developed and maintained by the repo owner, but the implementation was based on the following research papers:

- [Grad-CAM](https://arxiv.org/abs/1610.02391): GradCAM paper, generalizing CAM to models without global average pooling. 
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): Extends Grad-CAM with pixel-level weighting to handle multiple occurrences of objects. 
- [XGrad-CAM](https://arxiv.org/abs/2008.02312): Improves sensitivity and conservation of Grad-CAM maps.
- [Guided Baackprop](https://arxiv.org/abs/1412.6806): Modifies ReLU backward behavior to suppress negative gradients, highlighting input features that positively influence the prediction. 
- [Smooth Grad-CAM++](https://arxiv.org/abs/1908.01224): Reduces noise in Grad-CAM maps via input smoothing.
- [Integrated Gradients](https://arxiv.org/abs/1703.01365): Computes pixel importance by integrating gradients along a path from baseline to input.
- [RISE](https://arxiv.org/abs/1806.07421): generates random masks to compute saliency maps. 
- [Score-CAM](https://arxiv.org/abs/1910.01279): Weights activations of class activation for better interpretability. 
- [SS-CAM](https://arxiv.org/abs/2006.14255): SmoothGrad mechanism coupled with Score-CAM. 
- [IS-CAM](https://arxiv.org/abs/2010.03023): integration-based variant of Score-CAM. 
- [Layer-CAM](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf): Grad-CAM alternative leveraging pixel-wise contribution of the gradient to the activation.
- [Ablation-CAM](https://ieeexplore.ieee.org/abstract/document/9093360): Identifies critical neurons by systematically ablating activations.
- [Group-CAM](https://arxiv.org/abs/2103.13859): Aggregates grouped activations for robustness.
- [Union-CAM](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1490198/full): Combines multiple CAM maps to better cover object regions and improve interpretability.
- [FusionCAM](): Fuses gradient and region based cams to enhance localization and highlight complementary features.

## Citation

