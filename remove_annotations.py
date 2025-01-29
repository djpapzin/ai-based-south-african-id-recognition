import requests
import json
from collections import Counter

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
PROJECT_ID = "1"

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def get_annotations():
    """Get all annotations from the project"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting annotations: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

def delete_annotation(annotation_id):
    """Delete a specific annotation by ID"""
    url = f"{LABEL_STUDIO_URL}/api/annotations/{annotation_id}"
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        print(f"Successfully deleted annotation {annotation_id}")
        return True
    else:
        print(f"Error deleting annotation {annotation_id}: {response.status_code}")
        print(f"Response content: {response.text}")
        return False

def get_label_statistics():
    """Get statistics about all labels in use"""
    annotations = get_annotations()
    if not annotations:
        return None
    
    label_counter = Counter()
    for task in annotations:
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                labels = result.get('value', {}).get('rectanglelabels', [])
                label_counter.update(labels)
    
    return label_counter

def remove_annotations_by_label(label_to_remove):
    """Remove all annotations that use a specific label"""
    annotations = get_annotations()
    if not annotations:
        return
    
    deleted_count = 0
    for task in annotations:
        for annotation in task.get('annotations', []):
            # Check if any result in this annotation uses the specified label
            has_label = any(
                label_to_remove in result.get('value', {}).get('rectanglelabels', [])
                for result in annotation.get('result', [])
            )
            
            if has_label:
                annotation_id = annotation['id']
                if delete_annotation(annotation_id):
                    deleted_count += 1
    
    print(f"\nSummary:")
    print(f"Total annotations with label '{label_to_remove}' deleted: {deleted_count}")

if __name__ == "__main__":
    # First, show all labels and their counts
    print("Current labels in use:")
    print("-" * 40)
    
    label_stats = get_label_statistics()
    if label_stats:
        for label, count in label_stats.most_common():
            print(f"{label:30s}: {count:5d} annotations")
    
    # Ask which label to remove
    print("\nWhich label would you like to remove?")
    label_to_remove = input("Enter label name (exactly as shown above): ").strip()
    
    # Confirm before deletion
    confirm = input(f"\nAre you sure you want to remove all annotations with label '{label_to_remove}'? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        remove_annotations_by_label(label_to_remove)
    else:
        print("Operation cancelled.") 