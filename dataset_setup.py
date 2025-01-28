from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register the dataset
DATASET_NAME = "custom_dataset"
DATASET_METADATA = {
    "thing_classes": ["New", "Old"]  # Only two classes as requested
}

def register_custom_dataset(json_file, image_root):
    """
    Register the custom dataset with two classes: New and Old
    
    Args:
        json_file (str): Path to the COCO format annotation JSON file
        image_root (str): Directory containing the images
    """
    register_coco_instances(
        name=DATASET_NAME,
        metadata=DATASET_METADATA,
        json_file=json_file,
        image_root=image_root
    )
    
    # Verify the registration
    metadata = MetadataCatalog.get(DATASET_NAME)
    print(f"Dataset registered with classes: {metadata.thing_classes}")

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual paths
    JSON_FILE = "path/to/your/annotations.json"
    IMAGE_ROOT = "path/to/your/images"
    register_custom_dataset(JSON_FILE, IMAGE_ROOT) 