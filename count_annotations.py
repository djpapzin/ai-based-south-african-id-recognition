import requests
import json
from collections import defaultdict

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

def analyze_annotations(annotations):
    """Analyze annotations and count by label type"""
    # Counters for different statistics
    stats = {
        'total_images': len(annotations),
        'labels_per_image': defaultdict(lambda: defaultdict(int)),
        'total_by_label': defaultdict(int)
    }
    
    # Process each annotated image
    for task in annotations:
        image_filename = task.get('data', {}).get('image', 'Unknown')
        
        # Process each annotation for this image
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                if 'value' in result and 'rectanglelabels' in result['value']:
                    label = result['value']['rectanglelabels'][0]
                    stats['labels_per_image'][image_filename][label] += 1
                    stats['total_by_label'][label] += 1
    
    return stats

def print_statistics(stats):
    """Print the analysis results in a readable format"""
    print("\n=== Annotation Statistics ===")
    print(f"\nTotal Images: {stats['total_images']}")
    
    print("\nTotal Annotations by Label Type:")
    print("-" * 40)
    for label, count in sorted(stats['total_by_label'].items()):
        print(f"{label:20s}: {count:5d}")
    
    print("\nDetailed Breakdown by Image:")
    print("-" * 60)
    for image, labels in stats['labels_per_image'].items():
        print(f"\nImage: {image}")
        for label, count in sorted(labels.items()):
            print(f"  {label:20s}: {count:5d}")

if __name__ == "__main__":
    # Get all annotations
    annotations = get_annotations()
    
    if annotations:
        # Analyze the annotations
        stats = analyze_annotations(annotations)
        
        # Print the results
        print_statistics(stats)
    else:
        print("Failed to retrieve annotations.") 