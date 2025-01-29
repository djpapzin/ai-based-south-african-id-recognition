import requests
import json
import time
import argparse
import sys

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def get_project_id():
    print("Getting project ID...")
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=HEADERS)
    if response.status_code != 200:
        print(f"Error getting projects: {response.status_code}")
        sys.exit(1)
    
    data = response.json()
    if isinstance(data, dict) and 'results' in data:
        projects = data['results']
    elif isinstance(data, list):
        projects = data
    else:
        projects = []
    
    if not projects:
        print("No projects found")
        sys.exit(1)
    
    project_id = projects[0].get('id')
    if not project_id:
        print("Could not find project ID")
        sys.exit(1)
    
    print(f"✓ Found project with ID: {project_id}")
    return project_id

def export_annotations(project_id):
    print("Exporting annotations...")
    response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export?exportType=JSON",
        headers=HEADERS
    )
    if response.status_code != 200:
        print(f"Error exporting annotations: {response.status_code}")
        return False
    
    with open("annotations_backup.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    print("✓ Annotations exported and saved to annotations_backup.json")
    return True

def delete_project(project_id):
    print(f"Deleting project {project_id}...")
    response = requests.delete(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}",
        headers=HEADERS
    )
    if response.status_code not in [204, 404]:
        print(f"Error deleting project: {response.status_code}")
        return False
    print("✓ Project deleted successfully")
    return True

def create_new_project():
    print("Creating new project...")
    project_data = {
        "title": "Object Detection Project",
        "label_config": """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="box_labels" toName="image">
    <Label value="Object" background="green"/>
  </RectangleLabels>
</View>
"""
    }
    
    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/projects",
        json=project_data,
        headers=HEADERS
    )
    
    if response.status_code != 201:
        print(f"Error creating project: {response.status_code}")
        return None
    
    new_project = response.json()
    print(f"✓ New project created with ID: {new_project['id']}")
    return new_project['id']

def modify_annotations():
    print("Modifying annotations...")
    try:
        with open("annotations_backup.json", "r") as f:
            annotations = json.load(f)
        
        for annotation in annotations:
            if "label" in annotation.get("annotations", [{}])[0]:
                annotation["annotations"][0]["box_labels"] = annotation["annotations"][0].pop("label")
        
        with open("modified_annotations.json", "w") as f:
            json.dump(annotations, f, indent=2)
        print("✓ Annotations modified successfully")
        return True
    except Exception as e:
        print(f"Error modifying annotations: {e}")
        return False

def import_annotations(project_id):
    print("Importing modified annotations...")
    try:
        with open("modified_annotations.json", "r") as f:
            annotations = json.load(f)
        
        for task in annotations:
            response = requests.post(
                f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import",
                json=task,
                headers=HEADERS
            )
            if response.status_code != 201:
                print(f"Error importing task: {response.status_code}")
                return False
            time.sleep(0.5)  # Add delay to prevent overwhelming the server
        
        print("✓ Annotations imported successfully")
        return True
    except Exception as e:
        print(f"Error importing annotations: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', action='store_true', dest='continue_migration',
                      help='Continue from the last successful step')
    args = parser.parse_args()

    # Create a file to track progress
    try:
        with open('migration_progress.json', 'r') as f:
            progress = json.load(f)
    except FileNotFoundError:
        progress = {'step': 0}

    current_step = progress['step']
    
    try:
        # Step 1: Get project ID and export annotations
        if current_step < 1:
            project_id = get_project_id()
            if export_annotations(project_id):
                progress['step'] = 1
                progress['project_id'] = project_id
                with open('migration_progress.json', 'w') as f:
                    json.dump(progress, f)
            else:
                print("Failed to export annotations")
                return
        
        # Step 2: Delete old project and create new one
        if current_step < 2:
            if not delete_project(progress['project_id']):
                print("Failed to delete project")
                return
            
            new_project_id = create_new_project()
            if new_project_id:
                progress['step'] = 2
                progress['new_project_id'] = new_project_id
                with open('migration_progress.json', 'w') as f:
                    json.dump(progress, f)
            else:
                print("Failed to create new project")
                return
        
        # Step 3: Modify and import annotations
        if current_step < 3:
            if not modify_annotations():
                print("Failed to modify annotations")
                return
            
            if import_annotations(progress['new_project_id']):
                progress['step'] = 3
                with open('migration_progress.json', 'w') as f:
                    json.dump(progress, f)
                print("\n✓ Migration completed successfully!")
            else:
                print("Failed to import annotations")
                return

    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main() 