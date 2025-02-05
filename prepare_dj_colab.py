import json
import os
from sklearn.model_selection import train_test_split

local_dataset_dir = r'C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset'
output_dir = os.path.join(local_dataset_dir, 'merged_dataset')

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(local_dataset_dir, 'result.json')) as f:
    data = json.load(f)

for img in data['images']:
    img['file_name'] = f"images/{os.path.basename(img['file_name']).replace('\\', '/')}"

train_ids, val_ids = train_test_split(
    [img['id'] for img in data['images']],
    test_size=0.2,
    random_state=42,
    stratify=[ann['category_id'] for ann in data['annotations']]
)

for split_name, split_ids in [('train', train_ids), ('val', val_ids)]:
    split_data = {
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in split_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in split_ids]
    }
    
    with open(os.path.join(output_dir, f'{split_name}.json'), 'w') as f:
        json.dump(split_data, f)

print(f"Colab-ready dataset prepared at: {output_dir}")
