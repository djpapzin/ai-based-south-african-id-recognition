import requests
import json
from pprint import pprint
from pathlib import Path

# Configuration
BASE_URL = 'http://localhost:8080'
TOKEN = 'c4965d5c5515d76644b18343d198685d738ce34a'
HEADERS = {
    'Authorization': f'Token {TOKEN}',
    'Content-Type': 'application/json'
}

def check_labelstudio():
    try:
        # Get projects
        print("Connecting to Label Studio...")
        r = requests.get(f'{BASE_URL}/api/projects', headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        
        if not data.get('results'):
            print("No projects found in Label Studio!")
            return
            
        # Get the object detection project
        project = data['results'][0]  # We know there's only one project
        print(f"\nProject Details:")
        print(f"Title: {project['title']}")
        print(f"ID: {project['id']}")
        
        # Print annotation statistics
        print("\nAnnotation Status:")
        print(f"Total Tasks: {project['task_number']}")
        print(f"Tasks with Annotations: {project['num_tasks_with_annotations']}")
        print(f"Total Annotations: {project['total_annotations_number']}")
        print(f"Skipped Tasks: {project['skipped_annotations_number']}")
        
        # Calculate remaining tasks
        remaining = project['task_number'] - project['num_tasks_with_annotations'] - project['skipped_annotations_number']
        print(f"Remaining Tasks: {remaining}")
        
        # Print completion percentage
        completion_pct = (project['num_tasks_with_annotations'] / project['task_number']) * 100
        print(f"\nCompletion Progress: {completion_pct:.1f}%")
        
        # Summary of findings
        print("\nSummary:")
        print(f" {project['num_tasks_with_annotations']} tasks have been annotated")
        print(f" {project['skipped_annotations_number']} tasks have been skipped")
        if remaining > 0:
            print(f" {remaining} tasks remaining")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Label Studio. Is it running at http://localhost:8080?")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

def check_label_studio_export():
    # Paths
    dataset_dir = Path(r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset")
    min_json_file = dataset_dir / "result_min.json"
    
    # Load Label Studio export
    with open(min_json_file) as f:
        data = json.load(f)
    
    # Analyze annotations
    bbox_labels = {}
    keypoint_count = 0
    images_with_keypoints = 0
    images_with_bboxes = 0
    
    for item in data:
        has_keypoints = False
        has_bboxes = False
        
        # Check for keypoints
        if 'corners' in item:
            keypoint_count += len(item['corners'])
            has_keypoints = True
        
        # Check for bounding boxes
        if 'bbox' in item:
            has_bboxes = True
            for bbox in item['bbox']:
                if 'labels' in bbox and len(bbox['labels']) > 0:
                    label = bbox['labels'][0]
                    bbox_labels[label] = bbox_labels.get(label, 0) + 1
        
        if has_keypoints:
            images_with_keypoints += 1
        if has_bboxes:
            images_with_bboxes += 1
    
    print("\nLabel Studio Export Analysis:")
    print(f"Total images: {len(data)}")
    print(f"Images with keypoints: {images_with_keypoints}")
    print(f"Images with bounding boxes: {images_with_bboxes}")
    print(f"Total keypoints: {keypoint_count}")
    print("\nBounding box labels distribution:")
    for label, count in sorted(bbox_labels.items()):
        print(f"  - {label}: {count} boxes")

if __name__ == "__main__":
    check_labelstudio()
    check_label_studio_export()
