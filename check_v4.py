import requests
from collections import defaultdict

# Configuration
base_url = "http://localhost:8080/api/tasks"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}
params = {
    "project": 7,
    "page_size": 100
}

try:
    # Get total count first
    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()
    total_tasks = data.get('count', 0)
    print(f"Total tasks in project: {total_tasks}")
    
    # Process all pages
    all_tasks = []
    page = 1
    
    while len(all_tasks) < total_tasks:
        params['page'] = page
        print(f"Fetching page {page}...")
        
        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            break
            
        all_tasks.extend(results)
        print(f"Got {len(results)} tasks (Total: {len(all_tasks)})")
        page += 1
    
    # Process tasks for duplicates
    print(f"\nAnalyzing {len(all_tasks)} tasks for duplicates...")
    duplicates = defaultdict(lambda: defaultdict(int))
    task_ids = {}
    
    for task in all_tasks:
        if not task.get('annotations'):
            continue
        
        image = task['data'].get('image', '')
        task_ids[image] = task.get('id', 'unknown')
        
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                if 'value' in result and 'rectanglelabels' in result['value']:
                    for label in result['value']['rectanglelabels']:
                        duplicates[image][label] += 1
    
    # Report findings
    print("\nResults:")
    found = False
    
    for image, labels in duplicates.items():
        dups = {label: count for label, count in labels.items() if count > 1}
        if dups:
            found = True
            task_id = task_ids.get(image, 'unknown')
            print(f"\nImage: {image}")
            print(f"View at: http://localhost:8080/tasks/{task_id}")
            for label, count in dups.items():
                print(f"  {label}: {count} times")
    
    if not found:
        print("No duplicate labels found!")
    
    print(f"\nSummary:")
    print(f"Total tasks processed: {len(all_tasks)}")
    print(f"Total images with annotations: {len(duplicates)}")
    
except Exception as e:
    print(f"Error: {str(e)}") 