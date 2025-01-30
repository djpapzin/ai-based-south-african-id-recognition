import requests
import json
from pprint import pprint

API_TOKEN = 'c4965d5c5515d76644b18343d198685d738ce34a'
BASE_URL = 'http://localhost:8090/api'
HEADERS = {
    'Authorization': f'Token {API_TOKEN}',
    'Content-Type': 'application/json'
}

def get_projects():
    """Get all projects from Label Studio."""
    try:
        response = requests.get(f'{BASE_URL}/projects/', headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        print("\nRaw API Response:")
        pprint(data)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Label Studio API: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response: {response.text}")
        return None

def get_project_details(project_id):
    """Get detailed information about a specific project."""
    try:
        response = requests.get(f'{BASE_URL}/projects/{project_id}/', headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting project details: {e}")
        return None

def get_project_tasks(project_id):
    """Get all tasks for a specific project."""
    try:
        response = requests.get(f'{BASE_URL}/projects/{project_id}/tasks/', headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting project tasks: {e}")
        return None

def analyze_project(project_id):
    """Analyze a single project's data."""
    project = get_project_details(project_id)
    if not project:
        return
    
    print(f"\n=== Project Analysis ===")
    print(f"Title: {project.get('title', 'Unknown')}")
    print(f"Description: {project.get('description', 'No description')}")
    
    # Get label configuration
    label_config = project.get('label_config')
    if label_config:
        print("\nLabel Configuration:")
        print(label_config)
    
    # Get tasks and annotations
    tasks = get_project_tasks(project_id)
    if not tasks:
        return
    
    print(f"\nNumber of Tasks: {len(tasks)}")
    
    # Analyze annotations
    annotation_stats = {
        'total_annotations': 0,
        'label_counts': {},
        'type_counts': {}
    }
    
    for task in tasks:
        annotations = task.get('annotations', [])
        annotation_stats['total_annotations'] += len(annotations)
        
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                # Count by type
                result_type = result.get('type')
                if result_type:
                    annotation_stats['type_counts'][result_type] = \
                        annotation_stats['type_counts'].get(result_type, 0) + 1
                
                # Count by label
                if 'value' in result and 'labels' in result['value']:
                    for label in result['value']['labels']:
                        annotation_stats['label_counts'][label] = \
                            annotation_stats['label_counts'].get(label, 0) + 1
    
    # Print statistics
    print(f"\nTotal Annotations: {annotation_stats['total_annotations']}")
    
    if annotation_stats['type_counts']:
        print("\nAnnotation Types:")
        for type_name, count in annotation_stats['type_counts'].items():
            print(f"  - {type_name}: {count}")
    
    if annotation_stats['label_counts']:
        print("\nLabel Distribution:")
        for label, count in annotation_stats['label_counts'].items():
            print(f"  - {label}: {count}")

def main():
    print("Connecting to Label Studio API...")
    projects = get_projects()
    
    if not projects:
        print("No projects found or error connecting to Label Studio")
        return
    
    if isinstance(projects, list):
        print("\nFound Projects:")
        for project in projects:
            if isinstance(project, dict):
                print(f"ID: {project.get('id', 'Unknown')}, "
                      f"Title: {project.get('title', 'Unknown')}")
                analyze_project(project.get('id'))
            else:
                print(f"Unexpected project format: {project}")
    else:
        print(f"Unexpected response format: {type(projects)}")
        pprint(projects)

if __name__ == "__main__":
    main()
