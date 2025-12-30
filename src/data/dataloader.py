"""
DataLoader utilities for Oxford-IIIT Pet dataset
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import json
import os

from .dataset import OxfordPetDatasetWithMask, get_transforms


def create_dataloaders(
    data_root,
    split_metadata_path='../data/processed/split_metadata.json',
    batch_size=32,
    num_workers=4,
    img_size=224,
    pin_memory=True
):
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Root directory of dataset
        split_metadata_path: Path to split metadata JSON
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        img_size: Image size
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        dataset: Base OxfordIIITPet dataset
    """
    # Load base dataset
    dataset = OxfordIIITPet(
        root=data_root,
        split='trainval',
        target_types='segmentation',
        download=False
    )
    
    # Load split indices
    if os.path.exists(split_metadata_path):
        with open(split_metadata_path, 'r') as f:
            split_info = json.load(f)
        train_indices = split_info['train_indices']
        val_indices = split_info['val_indices']
        print(f"✓ Loaded split from {split_metadata_path}")
    else:
        # Create split if doesn't exist
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(dataset)))
        print(f"⚠ Split metadata not found, created new split")
    
    # Get transforms
    train_transform = get_transforms(img_size=img_size, augment=True)
    val_transform = get_transforms(img_size=img_size, augment=False)
    
    # Create datasets
    train_dataset = OxfordPetDatasetWithMask(
        dataset,
        train_indices,
        transform=train_transform,
        target_size=img_size
    )
    
    val_dataset = OxfordPetDatasetWithMask(
        dataset,
        val_indices,
        transform=val_transform,
        target_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader, dataset
