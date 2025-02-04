import json
import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import colorsys
import os
from PIL import Image

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    """Draw keypoints and their connections on the image."""
    # Draw points
    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] > 0:  # Only draw visible keypoints
        x, y = int(keypoints[i]), int(keypoints[i + 1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Draw point with inner and outer circle for better visibility
                cv2.circle(image, (x, y), 4, color, -1)  # Filled inner circle
                cv2.circle(image, (x, y), 6, color, 2)   # Outer circle border
                
                # Add point label with better visibility
                point_idx = i // 3
                point_labels = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
                label = point_labels[point_idx]
                
                # Add white background to text for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw white background
                cv2.rectangle(image, 
                            (x + 5, y - text_h - 5),
                            (x + text_w + 10, y + 5),
                            (255, 255, 255), -1)
                
                # Draw text
                cv2.putText(image, label,
                           (x + 7, y),
                           font, font_scale, color, thickness)
    
    # Draw connections between points
    valid_points = [(int(keypoints[i]), int(keypoints[i + 1])) 
                   for i in range(0, len(keypoints), 3)
                   if keypoints[i + 2] > 0 and 
                   0 <= int(keypoints[i]) < image.shape[1] and 
                   0 <= int(keypoints[i + 1]) < image.shape[0]]
    
    if len(valid_points) >= 2:
        for i in range(len(valid_points)):
            pt1 = valid_points[i]
            pt2 = valid_points[(i + 1) % len(valid_points)]
            cv2.line(image, pt1, pt2, color, 2)

def draw_bbox(image, bbox, category_name, color):
    """Draw a bounding box with label on the image."""
    x, y, w, h = [int(v) for v in bbox]
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Draw bounding box with thicker lines
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Add label with improved visibility
    label = category_name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Position label at top of bounding box
    label_x = x
    label_y = max(y - 5, label_h + 5)
    
    # Draw label background
    cv2.rectangle(image,
                 (label_x, label_y - label_h - baseline - 5),
                 (label_x + label_w + 10, label_y + baseline - 5),
                 color, -1)
    
    # Draw label text
    cv2.putText(image, label,
                (label_x + 5, label_y - 5),
                font, font_scale, (255, 255, 255),
                thickness)

def save_visualized_image(image, output_dir, filename):
    """Save the visualized image to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    return output_path

def validate_annotations(dataset_dir, split="train", num_samples=2):
    """Validate annotations and save visualized samples."""
    dataset_dir = Path(dataset_dir)
    ann_file = dataset_dir / split / "annotations.json"
    
    print(f"Loading annotations from {ann_file}")
    with open(ann_file, 'r') as f:
        dataset = json.load(f)
    
    # Create output directory for visualizations
    output_dir = dataset_dir / split / "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get categories
    categories = {cat["id"]: cat["name"] for cat in dataset["categories"]}
    print(f"\nCategories:")
    for cat_id, cat_name in categories.items():
        print(f"  - {cat_name}")
    
    # Generate distinct colors for each category
    num_categories = len(categories)
    colors = {}
    for i, cat_id in enumerate(categories.keys()):
        hue = i / num_categories
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors[cat_id] = tuple(int(255 * x) for x in rgb)
    
    print(f"\nTotal images: {len(dataset['images'])}")
    
    # Sample random images
    image_ids = np.random.choice(len(dataset["images"]), 
                               size=min(num_samples, len(dataset["images"])), 
                               replace=False)
    
    # Process each sampled image
    for img_idx in image_ids:
        img_info = dataset["images"][img_idx]
        img_id = img_info["id"]
        
        # Load image
        img_path = dataset_dir / split / "images" / Path(img_info["file_name"]).name
        print(f"\nProcessing image: {img_path}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        print(f"Image dimensions: {img.shape}")
        
        # Get annotations for this image
        img_anns = [ann for ann in dataset["annotations"] if ann["image_id"] == img_id]
        print(f"Found {len(img_anns)} annotations")
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Draw annotations
        for ann in img_anns:
            cat_id = ann["category_id"]
            cat_name = categories[cat_id]
            color = colors[cat_id]
            
            print(f"\nAnnotation: {cat_name}")
            print(f"Category ID: {cat_id}")
            
            # Draw keypoints for id_document
            if "keypoints" in ann and len(ann["keypoints"]) > 0:
                keypoints = ann["keypoints"]
                print("Drawing keypoints:")
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    point_idx = i // 3
                    point_labels = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
                    print(f"  - {point_labels[point_idx]}: x={x:.1f}, y={y:.1f}, visible={v}")
                draw_keypoints(vis_img, keypoints, color)
            
            # Draw bounding box
            if "bbox" in ann and len(ann["bbox"]) == 4:
                bbox = ann["bbox"]
                print("Drawing bbox:", bbox)
                draw_bbox(vis_img, bbox, cat_name, color)
        
        # Save the visualized image
        output_filename = f"vis_{Path(img_info['file_name']).stem}.jpg"
        saved_path = save_visualized_image(vis_img, output_dir, output_filename)
        print(f"\nSaved visualization to: {saved_path}")

def convert_rgba_to_rgb(img_path):
    """Convert RGBA image to RGB format."""
    try:
        with Image.open(img_path) as img:
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                else:
                    background.paste(img, mask=img.split()[1])  # Use alpha channel as mask
                return background
            elif img.mode != 'RGB':
                return img.convert('RGB')
            return img
    except Exception as e:
        print(f"Error converting {img_path}: {str(e)}")
        return None

def verify_and_fix_image(img_path):
    """Verify and fix a single image, returning the path to the fixed image."""
    try:
        # Convert JPEG to JPG if needed
        if img_path.lower().endswith('.jpeg'):
            new_path = str(img_path).replace('.jpeg', '.jpg')
            
            # Convert image
            img = convert_rgba_to_rgb(img_path)
            if img is None:
                return None
                
            # Save as JPG
            img.save(new_path, 'JPEG', quality=95, optimize=True)
            
            # Verify the saved image
            try:
                with Image.open(new_path) as check_img:
                    check_img.verify()
                # Remove old file if new one is valid
                if os.path.exists(img_path):
                    os.remove(img_path)
                print(f"Successfully converted {os.path.basename(img_path)} to JPG")
                return new_path
            except:
                if os.path.exists(new_path):
                    os.remove(new_path)
                print(f"Warning: Failed to verify converted image {os.path.basename(img_path)}")
                return None
        
        # Verify existing image
        with Image.open(img_path) as img:
            img.verify()
            # If it's RGBA, convert to RGB
            if img.mode in ('RGBA', 'LA'):
                img = convert_rgba_to_rgb(img_path)
                if img is not None:
                    img.save(img_path, 'JPEG', quality=95, optimize=True)
            return img_path
    except Exception as e:
        print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
        return None

def verify_and_fix_images(img_dir):
    """Verify and fix all images in directory."""
    print(f"\nVerifying images in {img_dir}")
    fixed_count = 0
    error_count = 0
    
    # Get list of all image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpeg', '.jpg'))]
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        fixed_path = verify_and_fix_image(img_path)
        
        if fixed_path is not None:
            fixed_count += 1
        else:
            error_count += 1
    
    print(f"Fixed {fixed_count} images, {error_count} errors encountered")
    return fixed_count, error_count

if __name__ == "__main__":
    # Validate training set
    validate_annotations("merged_dataset", "train", num_samples=2)
    
    # Validate validation set
    validate_annotations("merged_dataset", "val", num_samples=1)
