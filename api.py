
import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize
from torchvision.models import vgg16
from xaicam.metrics import AvgDropConf, InsertionDeletion
import gc
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize
# --------------------------------------------------
# Add package path
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xaicam.cam import *
from xaicam.utils import visualize_cam

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
IMAGE_PATH = "data\Images\dog_cat.png"
TARGET_LAYER=None

cams_dict = {
    "GradCAM": {"cam": GradCAM, "args": {"target_layer": TARGET_LAYER}},
    "GradCAM++": {"cam": GradCAMpp, "args": {"target_layer": TARGET_LAYER}},
    "Smooth Grad-CAM++": {"cam": SmoothGradCAMpp ,"args": {"target_layer": TARGET_LAYER}}, 
    "XGradCAM": {"cam": XGradCAM, "args": {"target_layer": TARGET_LAYER}},
    "ScoreCAM": {"cam": ScoreCAM, "args": {"target_layer": TARGET_LAYER}},
    "GroupCAM": {"cam": GroupCAM, "args": {"target_layer": TARGET_LAYER}},
    # "Guided BackProp": {"cam": GuidedBackProp},
    # "Integrated Gradients": {"cam": IntegratedGradients}, 
    "LayerCAM": {"cam": LayerCAM, "args": {"target_layer": TARGET_LAYER}}, 
    "RISE": {"cam": RISE, "args": {"input_shape": (3, 224, 224)}}, 
    # "AblationCAM": {"cam": AblationCAM, "args": {"target_layer": TARGET_LAYER}}, 
    "ISCAM": {"cam": ISCAM, "args": {"target_layer": TARGET_LAYER}},
    "SSCAM": {"cam": SSCAM, "args": {"target_layer": TARGET_LAYER}},
    "UnionCAM": {"cam": UnionCAM, "args": {"target_layer": TARGET_LAYER}},
    "FusionCAM": {"cam": FusionCAM, "args": {"target_layer": TARGET_LAYER, "grad_cam":GradCAM, "region_cam":ScoreCAM}},

}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = resize(img, (224, 224))
img_tensor = to_tensor(img)  # (3,H,W)

# Model input (normalized)
input_tensor = normalize(
    img_tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
).unsqueeze(0).to(DEVICE)

# Image for visualization (no normalization)
img_show = img_tensor.unsqueeze(0)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = vgg16(pretrained=True).eval().to(DEVICE)

# Forward pass to get predicted class
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
print(f"Predicted class: {predicted_class}")

# --------------------------------------------------
# GENERATE CAMS AND COMPUTE METRICS
# --------------------------------------------------
cams = {}
avgdrop_results = {}
insdel_results = {}

for cam_name, cam_info in cams_dict.items():
    cam_cls = cam_info["cam"]
    cam_args = cam_info.get("args", {})
    # predicted_class=281
    # Create the CAM instance dynamically
    with cam_cls(model, **cam_args) as cam_extractor:
        cam = cam_extractor(input_tensor, class_idx=predicted_class).cpu().detach()
        cams[cam_name] = cam


        # # AvgDrop / Increase
        # metric = AvgDropConf(model, cam, input_tensor, class_idx=predicted_class)
        # avgdrop_results[cam_name] = metric.summary()
        # print(f"{cam_name} Aerage Drop/Increase: {avgdrop_results[cam_name]}")

        # # Insertion
        # insdel = InsertionDeletion(model=model, mode="Insertion")
        # ins_result = insdel.compute(image=input_tensor, heatmap=cam, return_steps=True)
        # insdel_results.setdefault(cam_name, {})["Insertion"] = ins_result

        # # Deletion
        # insdel_del = InsertionDeletion(model=model, mode="Deletion")
        # del_result = insdel_del.compute(image=input_tensor, heatmap=cam, return_steps=True)
        # insdel_results[cam_name]["Deletion"] = del_result

        # # --------------------------------------------------
        # # Plot single CAM insertion + deletion
        # # --------------------------------------------------
        # print(f"Plotting curves for {cam_name}...")
        # InsertionDeletion.plot_curves(
        #     results=ins_result,
        #     mode="Insertion",
        #     image=img_show,
        #     cam=cam,
        #     alpha=0.6,
        #     title = cam_name,
        #     save_path=os.path.join(SAVE_DIR, f"{cam_name}_insertion.png")
        # )

        # InsertionDeletion.plot_curves(
        #     results=del_result,
        #     mode="Deletion",
        #     image=img_show,
        #     cam=cam,
        #     alpha=0.6,
        #     title = cam_name,
        #     save_path=os.path.join(SAVE_DIR, f"{cam_name}_deletion.png")
        # )

    # -----------------------------
    # MEMORY CLEANUP (CRITICAL)
    # -----------------------------
    del cam
    torch.cuda.empty_cache()
    gc.collect()


# Visualize all CAMs in one figure
visualize_cam(
    img=img_show,
    cams=cams,
    nrow=4,
    save_path=os.path.join(SAVE_DIR, "all_cams.png")
)
# --------------------------------------------------
# Plot all CAMs comparison curves
# --------------------------------------------------
# Insertion curves
# all_ins_results = {name: res["Insertion"] for name, res in insdel_results.items()}
# InsertionDeletion.plot_curves(
#     results=all_ins_results,
#     mode="Insertion",
#     title="Insertion Curves Comparison",
#     save_path=os.path.join(SAVE_DIR, "all_cams_insertion.png")
# )

# # Deletion curves
# all_del_results = {name: res["Deletion"] for name, res in insdel_results.items()}
# InsertionDeletion.plot_curves(
#     results=all_del_results,
#     mode="Deletion",
#     title="Deletion Curves Comparison",
#     save_path=os.path.join(SAVE_DIR, "all_cams_deletion.png")
# )

print("\nAll CAM processing completed. Plots saved to outputs/")
