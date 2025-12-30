"""
Oxford-IIIT Pet Dataset with Masks and Counterfactual Support
"""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from PIL import Image
import numpy as np

class OxfordPetDatasetWithMask(Dataset):
    """
    Custom dataset that returns:
    - image (transformed, normalized)
    - mask (binary: 1=foreground, 0=background/border)
    - label (class index)
    - original_image (resized, for counterfactual generation)
    - original_mask (resized, for counterfactual generation)
    """
    def __init__(self, base_dataset, indices, transform=None, target_size=224):
        """
        Args:
            base_dataset: OxfordIIITPet dataset
            indices: List of indices to use from base_dataset
            transform: Transforms to apply to images (for training)
            target_size: Size to resize original images/masks for counterfactuals
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.target_size = target_size
        
        # Transform for original images (no normalization, just resize)
        self.original_transform = T.Compose([
            T.Resize((target_size, target_size)),
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, mask = self.base_dataset[actual_idx]
        label = self.base_dataset._labels[actual_idx]
        
        # Resize original for consistent sizes
        original_image = self.original_transform(image)
        original_mask = T.Resize((self.target_size, self.target_size))(mask)
        
        # Convert to numpy arrays
        original_image_np = np.array(original_image, dtype=np.uint8)
        original_mask_np = np.array(original_mask, dtype=np.uint8)
        
        # Convert mask to binary (1=foreground, 0=background/border)
        binary_mask = (original_mask_np == 1).astype(np.float32)
        
        # Apply transforms to image if provided (for training)
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform, at least convert to tensor
            image = T.ToTensor()(image)
        
        return {
            'image': image,
            'mask': torch.from_numpy(binary_mask),
            'label': label,
            'original_image': original_image_np,
            'original_mask': original_mask_np,
            'index': actual_idx
        }


def get_transforms(img_size=224, augment=True):
    """
    Get standard transforms for Oxford-IIIT Pet dataset
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        transform: torchvision transform
    """
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transforms with augmentation
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation transforms (no augmentation)
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 4:
        # Batch of images
        tensor = tensor.clone()
        for i in range(tensor.size(0)):
            for t, m, s in zip(tensor[i], mean, std):
                t.mul_(s).add_(m)
    else:
        # Single image
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    
    return torch.clamp(tensor, 0, 1)
