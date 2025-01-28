import requests
import json
import os
from datetime import datetime

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
PROJECT_ID = "1"

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def backup_annotations():
    """Export and save current annotations"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"annotations_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(response.json(), f, indent=2)
        print(f"Backup saved to {backup_file}")
        return response.json()
    else:
        print(f"Error exporting annotations: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

def update_annotation(annotation_id, updated_result):
    """Update a single annotation"""
    url = f"{LABEL_STUDIO_URL}/api/annotations/{annotation_id}"
    payload = {"result": updated_result}
    response = requests.patch(url, headers=headers, json=payload)
    return response.status_code == 200

def update_project_config(new_config):
    """Update project configuration"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    response = requests.patch(url, headers=headers, json={"label_config": new_config})
    if response.status_code == 200:
        print("Label configuration updated successfully!")
        return True
    else:
        print(f"Error updating configuration: {response.status_code}")
        print(f"Response content: {response.text}")
        return False

def migrate_labels():
    """Migrate labels from 'label' to 'box_labels'"""
    # 1. Backup current annotations
    print("Backing up current annotations...")
    annotations = backup_annotations()
    if not annotations:
        return

    # 2. Update annotations
    print("\nUpdating annotations...")
    updated_count = 0
    for task in annotations:
        for annotation in task.get('annotations', []):
            updated_results = []
            needs_update = False
            
            for result in annotation.get('result', []):
                if result.get('from_name') == 'label':
                    result['from_name'] = 'box_labels'
                    needs_update = True
                updated_results.append(result)
            
            if needs_update:
                if update_annotation(annotation['id'], updated_results):
                    updated_count += 1
                    print(f"Updated annotation {annotation['id']}")
                else:
                    print(f"Failed to update annotation {annotation['id']}")

    print(f"\nUpdated {updated_count} annotations")

    # 3. Update configuration
    print("\nUpdating label configuration...")
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
    
    if update_project_config(new_config):
        print("\nMigration completed successfully!")
    else:
        print("\nFailed to update configuration. Please check the error messages above.")

if __name__ == "__main__":
    print("Starting label migration process...")
    migrate_labels() 