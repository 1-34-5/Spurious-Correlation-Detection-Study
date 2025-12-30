import os
import pathlib
import sys
from torchvision.datasets import OxfordIIITPet

def setup_dataset():
    # Setup paths
    script_dir = pathlib.Path(__file__).parent.absolute()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'raw'
    
    try:
        # Ensure directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Target directory: {data_dir}")
        
        # Download dataset directly to data/raw
        dataset = OxfordIIITPet(
            root=str(data_dir),
            split='trainval',
            target_types=['segmentation'],
            download=True
        )
        
        dataset_dir = data_dir / 'oxford-iiit-pet'
        if dataset_dir.exists():
            print(f"\n✓ Successfully downloaded {len(dataset)} images")
            print(f"✓ Dataset location: {dataset_dir}")
        else:
            raise RuntimeError(f"Download failed - {dataset_dir} not found")
            
    except PermissionError:
        print("Error: Permission denied. Check folder permissions")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_dataset()