import requests
import json
import time
from pathlib import Path

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"  # Changed back to 8090
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
PROJECT_ID = "1"

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def export_annotations():
    """Export current annotations"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Save backup
        with open('annotations_backup.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("Annotations exported and backed up")
        return data
    else:
        print(f"Error exporting: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def modify_annotations(annotations):
    """Modify annotations to use new label name"""
    modified = False
    for task in annotations:
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('from_name') == 'label':
                    result['from_name'] = 'box_labels'
                    modified = True
    
    if modified:
        with open('modified_annotations.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        print("Annotations modified and saved")
    return annotations if modified else None

def create_new_project():
    """Create a new project with correct configuration"""
    url = f"{LABEL_STUDIO_URL}/api/projects"
    
    new_config = """<View>
  <Image name="image" value="$image" rotateControl="true"/>
  <RectangleLabels name="box_labels" toName="image">
    <Label value="New ID Card" background="#FFA39E"/>
    <Label value="Old ID Book" background="#0dd325"/>
    <Label value="Surname" background="#FFA39E"/>
    <Label value="Names" background="#D4380D"/>
    <Label value="Sex" background="#FFC069"/>
    <Label value="Nationality" background="#AD8B00"/>
    <Label value="ID Number" background="#D3F261"/>
    <Label value="Date of Birth" background="#389E0D"/>
    <Label value="Country of Birth" background="#5CDBD3"/>
    <Label value="Signature" background="#ADC6FF"/>
    <Label value="Face Photo" background="#9254DE"/>
    <Label value="Citizenship Status" background="#FFA39E"/>
  </RectangleLabels>
</View>"""
    
    payload = {
        "title": "ID Card Detection - Fixed",
        "label_config": new_config
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        new_project = response.json()
        print(f"New project created with ID: {new_project['id']}")
        return new_project['id']
    else:
        print(f"Error creating project: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def import_annotations(project_id, annotations):
    """Import annotations into new project"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import"
    
    files = {
        'file': ('annotations.json', json.dumps(annotations), 'application/json')
    }
    
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 201:
        print("Annotations imported successfully")
        return True
    else:
        print(f"Error importing: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    # 1. Export current annotations
    print("Exporting current annotations...")
    annotations = export_annotations()
    if not annotations:
        return
    
    # 2. Modify annotations
    print("\nModifying annotations...")
    modified_data = modify_annotations(annotations)
    if not modified_data:
        print("No modifications needed")
        return
    
    # 3. Create new project
    print("\nCreating new project...")
    new_project_id = create_new_project()
    if not new_project_id:
        return
    
    # 4. Wait a bit for project creation to complete
    print("Waiting for project setup...")
    time.sleep(5)
    
    # 5. Import modified annotations
    print("\nImporting modified annotations...")
    if import_annotations(new_project_id, modified_data):
        print("\nProcess completed successfully!")
        print(f"New project ID: {new_project_id}")
        print("You can now delete the old project if everything looks correct.")
    else:
        print("\nFailed to complete the process.")

if __name__ == "__main__":
    main() 