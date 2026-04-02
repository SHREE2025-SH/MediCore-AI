"""
MediCore AI — GRAD-CAM Visualization
Generates heatmaps showing which regions the CNN focused on
Author: Spandan Das
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class GradCAM:
    """
    GRAD-CAM: Gradient-weighted Class Activation Mapping
    Highlights regions in the image that most influenced the prediction.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Default to layer4 of ResNet50
        if target_layer is None:
            target_layer = model.layer4[-1]

        # Register hooks
        self.hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):
        """
        Generate GRAD-CAM heatmap.
        image_tensor: (1, 3, 224, 224)
        Returns: heatmap as numpy array (224, 224), values 0-1
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute GRAD-CAM
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, class_idx

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def generate_gradcam_image(model, image: Image.Image, class_idx=None) -> Image.Image:
    """
    Full pipeline: PIL Image → GRAD-CAM overlay → PIL Image
    Returns the original image with heatmap overlay.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare tensor
    tensor = img_transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap, predicted_class = gradcam.generate(tensor, class_idx)
    gradcam.remove_hooks()

    # Convert original image to numpy
    orig = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    # Apply colormap to heatmap
    colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel

    # Overlay: 60% original + 40% heatmap
    overlay = 0.6 * orig + 0.4 * colormap
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return Image.fromarray(overlay), heatmap, predicted_class


def create_gradcam_figure(
    original: Image.Image,
    overlay: Image.Image,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    scan_type: str
) -> Image.Image:
    """
    Creates a side-by-side figure: Original | GRAD-CAM Overlay | Heatmap
    Returns as PIL Image ready for Streamlit display.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0f1e')

    titles = ["Original Scan", "GRAD-CAM Overlay", "Attention Heatmap"]
    images_to_show = [
        np.array(original.resize((224, 224))),
        np.array(overlay),
        heatmap
    ]
    cmaps = [None, None, 'jet']

    for ax, title, img, cmap in zip(axes, titles, images_to_show, cmaps):
        ax.set_facecolor('#0a0f1e')
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(title, color='#00d4ff', fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    # Add colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Attention', color='#8899aa', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#8899aa')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8899aa')

    # Main title
    fig.suptitle(
        f"{scan_type.upper()} | Prediction: {prediction} ({confidence*100:.1f}%)",
        color='white',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    # Convert to PIL Image
    buf = plt.get_current_fig_manager()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf_arr = buf_arr.reshape(h, w, 3)
    plt.close(fig)

    return Image.fromarray(buf_arr)


if __name__ == "__main__":
    print("[GRAD-CAM] Module ready!")
    print("Usage: from gradcam import generate_gradcam_image, create_gradcam_figure")