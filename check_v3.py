import requests
from collections import defaultdict

url = "http://localhost:8080/api/projects/7/tasks"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}

try:
    # Fetch all tasks
    all_tasks = []
    next_url = url
    
    while next_url:
        print(f"Fetching from: {next_url}")
        response = requests.get(next_url, headers=headers)
        data = response.json()
        
        if isinstance(data, dict) and 'results' in data:
            all_tasks.extend(data['results'])
            next_url = data.get('next')
        else:
            all_tasks.extend(data)
            break
    
    print(f"\nFound {len(all_tasks)} total tasks")
    duplicates = defaultdict(lambda: defaultdict(int))
    
    # Process tasks
    for task in all_tasks:
        if not task.get('annotations'):
            continue
        
        image = task['data'].get('image', '')
        annotation = task['annotations'][-1]
        
        for result in annotation.get('result', []):
            if 'value' in result:
                if 'rectanglelabels' in result['value']:
                    for label in result['value']['rectanglelabels']:
                        duplicates[image][label] += 1
    
    # Report duplicates
    print("\nChecking for duplicates...")
    found = False
    
    for image, labels in duplicates.items():
        dups = {label: count for label, count in labels.items() if count > 1}
        if dups:
            found = True
            print(f"\nImage: {image}")
            for label, count in dups.items():
                print(f"  {label}: {count} times")
    
    if not found:
        print("No duplicates found!")
    
except Exception as e:
    print(f"Error: {str(e)}") 