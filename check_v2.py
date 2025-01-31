import requests
from collections import defaultdict

# Configuration
url = "http://localhost:8080/api/projects/7/tasks"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}

try:
    print("Fetching tasks...")
    response = requests.get(url, headers=headers)
    tasks = response.json()
    
    print(f"Processing {len(tasks)} tasks...")
    duplicates = defaultdict(list)
    
    for task in tasks:
        if not task.get('annotations'):
            continue
            
        image = task['data'].get('image', '')
        annotation = task['annotations'][-1]
        
        for result in annotation.get('result', []):
            if 'value' in result:
                for label_type in ['labels', 'rectanglelabels']:
                    if label_type in result['value']:
                        labels = result['value'][label_type]
                        for label in labels:
                            duplicates[image].append(label)
    
    print("\nChecking for duplicates...")
    for image, labels in duplicates.items():
        dups = {label: labels.count(label) for label in set(labels) if labels.count(label) > 1}
        if dups:
            print(f"\nImage: {image}")
            for label, count in dups.items():
                print(f"  {label}: {count} times")
                
except Exception as e:
    print(f"Error: {str(e)}") 