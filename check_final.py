from collections import defaultdict

# Configuration
PROJECT_ID = 7
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
BASE_URL = "http://localhost:8080"

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

try:
    # First verify project exists
    print("Verifying project access...")
    project_url = f"{BASE_URL}/api/projects/{PROJECT_ID}"
    project_response = requests.get(project_url, headers=headers)
    project_response.raise_for_status()
    project = project_response.json()
    print(f"Found project: {project.get('title', 'Unknown')}")

    # Get all tasks
    print("\nFetching tasks...")
    tasks_url = f"{BASE_URL}/api/projects/{PROJECT_ID}/tasks"
    tasks_response = requests.get(tasks_url, headers=headers)
    tasks_response.raise_for_status()
    tasks = tasks_response.json()

    if not tasks:
        print("No tasks found in project")
        exit()

    print(f"Processing {len(tasks)} tasks...")
    duplicates = defaultdict(lambda: defaultdict(list))

    # Process each task
    for task in tasks:
        image = task['data'].get('image', '')
        task_id = task.get('id', 'unknown')
        
        if not task.get('annotations'):
            continue

        # Get latest annotation
        annotation = task['annotations'][-1]
        
        # Process each result
        for result in annotation.get('result', []):
            if 'value' in result and 'rectanglelabels' in result['value']:
                for label in result['value']['rectanglelabels']:
                    # Store label with its coordinates
                    coords = {
                        'x': result['value'].get('x', 0),
                        'y': result['value'].get('y', 0),
                        'width': result['value'].get('width', 0),
                        'height': result['value'].get('height', 0)
                    }
                    duplicates[image][label].append(coords)

    # Report findings
    print("\nResults:")
    found_duplicates = False

    for image, labels in duplicates.items():
        image_has_duplicates = False
        duplicate_info = []

        for label, instances in labels.items():
            if len(instances) > 1:
                image_has_duplicates = True
                duplicate_info.append(f"\n  Label '{label}' appears {len(instances)} times:")
                for i, coords in enumerate(instances, 1):
                    duplicate_info.append(
                        f"    Instance {i}: x={coords['x']:.1f}, y={coords['y']:.1f}, "
                        f"w={coords['width']:.1f}, h={coords['height']:.1f}"
                    )

        if image_has_duplicates:
            found_duplicates = True
            print(f"\nImage: {image}")
            print(f"View at: {BASE_URL}/tasks/{task_id}")
            print("Duplicate labels:" + "\n".join(duplicate_info))

    if not found_duplicates:
        print("No duplicate labels found!")

    print(f"\nSummary:")
    print(f"Total tasks processed: {len(tasks)}")
    print(f"Total images with annotations: {len(duplicates)}")

except requests.exceptions.RequestException as e:
    print(f"API Error: {str(e)}")
except Exception as e:
 