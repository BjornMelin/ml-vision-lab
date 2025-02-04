from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM implementation for CNN visualization."""

    def __init__(self, model: torch.nn.Module, target_layer: str):
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        target = dict([*self.model.named_modules()])[target_layer]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Model input of shape (1, C, H, W)
            target_class: Target class index, if None uses predicted class

        Returns:
            np.ndarray: Heatmap of shape (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Generate weighted activation map
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        cam = torch.sum(weights[:, None, None] * self.activations[0], dim=0)

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)

        return cam.cpu().numpy()

    def overlay_heatmap(
        self,
        image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay heatmap on original image.

        Args:
            image: Original image
            heatmap: Generated heatmap
            alpha: Transparency factor

        Returns:
            PIL.Image: Image with overlaid heatmap
        """
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create heatmap overlay
        heatmap = np.uint8(255 * heatmap)
        heatmap = np.stack((heatmap,) * 3, axis=-1)

        # Blend images
        output = np.uint8(image * (1 - alpha) + heatmap * alpha)

        return Image.fromarray(output)
