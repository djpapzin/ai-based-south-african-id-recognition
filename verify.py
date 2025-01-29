import os
import json
from PIL import Image

def verify_and_fix_dataset(json_path, images_dir):
    print(f"Verifying and fixing dataset at {json_path}")
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Fix paths and dimensions
    fixed_images = []
    for img in coco_data['images']:
        # Get filename only
        filename = os.path.basename(img['file_name'].replace('\\', '/'))
        image_path = os.path.join(images_dir, filename)
        
        if os.path.exists(image_path):
            try:
                # Get actual image dimensions
                with Image.open(image_path) as im:
                    width, height = im.size
                    
                # Update image entry
                img['file_name'] = os.path.join('images', filename)
                img['width'] = width
                img['height'] = height
                fixed_images.append(img)
                print(f"Fixed {filename}: {width}x{height}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Warning: File not found: {filename}")
    
    # Update images in JSON
    coco_data['images'] = fixed_images
    
    # Save fixed JSON
    output_path = json_path.replace('.json', '_verified.json')
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"\nProcessed {len(fixed_images)} images")
    return output_path

# Verify both datasets
abenathi_json = verify_and_fix_dataset(
    "abenathi_data/result.json",
    "abenathi_data/images"
)