"""
Counterfactual image generation utilities
"""
import numpy as np
import cv2
from PIL import Image


def create_counterfactual(fg_image, fg_mask, bg_image, bg_mask, blend_sigma=5):
    """
    Create counterfactual by swapping backgrounds.
    
    CORRECT: Pet 1 foreground + Pet 2 BACKGROUND ONLY
    
    Args:
        fg_image: Pet 1 image (numpy array, uint8)
        fg_mask: Pet 1 mask (1=pet, 0=background)
        bg_image: Pet 2 image (source of background)
        bg_mask: Pet 2 mask (1=pet, 0=background) ← CRITICAL!
        blend_sigma: Gaussian blur sigma for smooth edges
    
    Returns:
        Counterfactual image with Pet 1 foreground + Pet 2 background
    """
    # Ensure images are same size
    h, w = fg_image.shape[:2]
    bg_image_resized = cv2.resize(bg_image, (w, h))
    
    # Ensure masks are 2D
    if fg_mask.ndim == 3:
        fg_mask = fg_mask[:, :, 0]
    if bg_mask.ndim == 3:
        bg_mask = bg_mask[:, :, 0]
    
    # Convert masks to 3 channels for element-wise operations
    fg_mask_3ch = np.stack([fg_mask, fg_mask, fg_mask], axis=-1)
    bg_mask_3ch = np.stack([(bg_mask == 1).astype(np.float32)] * 3, axis=-1)  # Only pet area
    
    # Extract Pet 1 foreground (where fg_mask == 1)
    foreground = fg_image * fg_mask_3ch
    
    # Extract Pet 2 BACKGROUND ONLY (where bg_mask != 1)
    background = bg_image_resized * (1 - bg_mask_3ch)
    
    # Combine: Pet 1 foreground + Pet 2 background
    counterfactual = foreground + background
    
    # Optional: Gaussian blur at edges for smoother transition
    if blend_sigma > 0:
        # Create edge mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask_uint8 = (fg_mask * 255).astype(np.uint8)
        edge_mask = cv2.dilate(fg_mask_uint8, kernel, iterations=1) - cv2.erode(fg_mask_uint8, kernel, iterations=1)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        edge_mask_3ch = np.stack([edge_mask, edge_mask, edge_mask], axis=-1)
        
        # Blur the counterfactual slightly at edges for smooth transition
        blurred = cv2.GaussianBlur(counterfactual.astype(np.float32), (9, 9), blend_sigma)
        counterfactual = counterfactual * (1 - edge_mask_3ch) + blurred * edge_mask_3ch
    
    return counterfactual.astype(np.uint8)

print("✓ Corrected counterfactual generator defined")

