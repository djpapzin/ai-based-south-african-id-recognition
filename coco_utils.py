import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'coco_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def validate_coco_format(coco_file: str, image_dir: str) -> bool:
    """
    Validate a COCO format JSON file by checking its structure and contents.
    
    Args:
        coco_file (str): Path to the COCO JSON file
        image_dir (str): Path to the directory containing the images
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        logging.info(f"Starting validation of COCO file: {coco_file}")
        
        # Load JSON file
        with open(coco_file, 'r') as f:
            try:
                coco_data = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON format: {str(e)}")
                return False
        
        # Check required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                logging.error(f"Missing required field: {field}")
                return False
        
        # Validate categories
        if not coco_data['categories']:
            logging.error("No categories found")
            return False
        
        category_ids = {cat['id'] for cat in coco_data['categories']}
        
        # Validate images
        image_ids = set()
        for img in coco_data['images']:
            # Check required image fields
            if not all(key in img for key in ['id', 'width', 'height', 'file_name']):
                logging.error(f"Missing required fields in image entry: {img}")
                return False
            
            # Check image file existence
            img_path = os.path.join(image_dir, img['file_name'])
            if not os.path.exists(img_path):
                logging.error(f"Image file not found: {img_path}")
                return False
            
            # Verify image dimensions
            try:
                actual_img = cv2.imread(img_path)
                if actual_img is None:
                    logging.error(f"Could not read image: {img_path}")
                    return False
                
                actual_height, actual_width = actual_img.shape[:2]
                if actual_width != img['width'] or actual_height != img['height']:
                    logging.error(f"Image dimensions mismatch for {img_path}. "
                                f"JSON: {img['width']}x{img['height']}, "
                                f"Actual: {actual_width}x{actual_height}")
                    return False
            except Exception as e:
                logging.error(f"Error validating image {img_path}: {str(e)}")
                return False
            
            image_ids.add(img['id'])
        
        # Validate annotations
        for ann in coco_data['annotations']:
            # Check required annotation fields
            if not all(key in ann for key in ['id', 'image_id', 'category_id', 'bbox']):
                logging.error(f"Missing required fields in annotation: {ann}")
                return False
            
            # Validate image_id reference
            if ann['image_id'] not in image_ids:
                logging.error(f"Annotation references non-existent image_id: {ann['image_id']}")
                return False
            
            # Validate category_id reference
            if ann['category_id'] not in category_ids:
                logging.error(f"Annotation references non-existent category_id: {ann['category_id']}")
                return False
            
            # Validate bbox format
            if len(ann['bbox']) != 4:
                logging.error(f"Invalid bbox format in annotation: {ann}")
                return False
            
            # Validate bbox values
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                logging.error(f"Invalid bbox dimensions in annotation: {ann}")
                return False
        
        logging.info("COCO validation completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Validation failed with error: {str(e)}")
        return False

def convert_labelstudio_to_coco(
    labelstudio_file: str,
    image_dir: str,
    output_file: str
) -> bool:
    """
    Convert Label Studio JSON export to COCO format.
    
    Args:
        labelstudio_file (str): Path to the Label Studio JSON export file
        image_dir (str): Path to the directory containing the images
        output_file (str): Path where the COCO JSON file will be saved
        
    Returns:
        bool: True if conversion succeeds, False otherwise
    """
    try:
        logging.info(f"Starting conversion of Label Studio file: {labelstudio_file}")
        
        # Load Label Studio JSON
        with open(labelstudio_file, 'r') as f:
            labelstudio_data = json.load(f)
        
        # Initialize COCO format structure
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Create category mapping
        categories = set()
        for item in labelstudio_data:
            if 'annotations' in item:
                for ann in item['annotations']:
                    if 'result' in ann:
                        for result in ann['result']:
                            if 'value' in result and 'rectanglelabels' in result['value']:
                                categories.update(result['value']['rectanglelabels'])
        
        # Add categories to COCO format
        for idx, category in enumerate(sorted(categories), 1):
            coco_data['categories'].append({
                'id': idx,
                'name': category,
                'supercategory': 'none'
            })
        
        category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
        
        # Process images and annotations
        ann_id = 1
        for idx, item in enumerate(labelstudio_data, 1):
            # Get image information
            if 'data' not in item or 'image' not in item['data']:
                logging.warning(f"Skipping item {idx}: No image data found")
                continue
                
            image_filename = os.path.basename(item['data']['image'])
            image_path = os.path.join(image_dir, image_filename)
            
            # Verify image exists and get dimensions
            if not os.path.exists(image_path):
                logging.error(f"Image not found: {image_path}")
                continue
                
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Could not read image: {image_path}")
                continue
                
            height, width = img.shape[:2]
            
            # Add image to COCO format
            coco_data['images'].append({
                'id': idx,
                'width': width,
                'height': height,
                'file_name': image_filename
            })
            
            # Process annotations
            if 'annotations' in item:
                for ann in item['annotations']:
                    if 'result' in ann:
                        for result in ann['result']:
                            if 'value' in result and 'rectanglelabels' in result['value']:
                                # Get bbox coordinates
                                x = result['value']['x'] * width / 100.0
                                y = result['value']['y'] * height / 100.0
                                w = result['value']['width'] * width / 100.0
                                h = result['value']['height'] * height / 100.0
                                
                                # Add annotation to COCO format
                                for label in result['value']['rectanglelabels']:
                                    coco_data['annotations'].append({
                                        'id': ann_id,
                                        'image_id': idx,
                                        'category_id': category_map[label],
                                        'bbox': [x, y, w, h],
                                        'area': w * h,
                                        'iscrowd': 0
                                    })
                                    ann_id += 1
        
        # Save COCO format JSON
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logging.info(f"Successfully converted to COCO format: {output_file}")
        
        # Validate the generated COCO file
        if validate_coco_format(output_file, image_dir):
            logging.info("Generated COCO file passed validation")
            return True
        else:
            logging.error("Generated COCO file failed validation")
            return False
            
    except Exception as e:
        logging.error(f"Conversion failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage with actual paths
    labelstudio_file = "annotated_images/result_fixed.json"
    image_dir = "annotated_images/images"
    output_file = "pipeline_output/annotations.json"
    
    # Create output directory if it doesn't exist
    os.makedirs("pipeline_output", exist_ok=True)
    
    # Convert Label Studio to COCO
    if convert_labelstudio_to_coco(labelstudio_file, image_dir, output_file):
        print("Successfully converted Label Studio annotations to COCO format")
        
        # Validate the converted file
        if validate_coco_format(output_file, image_dir):
            print("COCO format validation passed")
        else:
            print("COCO format validation failed")
    else:
        print("Conversion failed") 