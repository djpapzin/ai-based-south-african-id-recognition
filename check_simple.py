import requests

url = "http://localhost:8080/api/projects/7/tasks"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}

try:
    print("Fetching tasks...")
    response = requests.get(url, headers=headers)
    tasks = response.json()
    
    print(f"Found {len(tasks)} tasks")
    duplicates = {}
    
    for task in tasks:
        if task.get('annotations'):
            image = task['data'].get('image', '')
            latest = task['annotations'][-1]
            
            for result in latest.get('result', []):
                if 'value' in result and 'rectanglelabels' in result['value']:
                    labels = result['value']['rectanglelabels']
                    if image not in duplicates:
                        duplicates[image] = {}
                    for label in labels:
                        duplicates[image][label] = duplicates[image].get(label, 0) + 1
    
    print("\nChecking for duplicates...")
    found = False
    for image, labels in duplicates.items():
        dups = {l: c for l, c in labels.items() if c > 1}
        if dups:
            found = True
            print(f"\nImage: {image}")
            for label, count in dups.items():
                print(f"  {label}: {count} times")
    
    if not found:
        print("No duplicates found!")
        
except Exception as e:
    print(f"Error: {str(e)}") 