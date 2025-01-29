import requests
import json

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
PROJECT_ID = "1"

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def get_project_config():
    """Get current project configuration"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting project config: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

def update_project_config(new_config):
    """Update project configuration"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    response = requests.patch(url, headers=headers, json={"label_config": new_config})
    if response.status_code == 200:
        print("Label configuration updated successfully!")
        return response.json()
    else:
        print(f"Error updating configuration: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

def update_label_name():
    """Update the label name in the configuration"""
    project = get_project_config()
    if not project:
        return
    
    current_config = project["label_config"]
    print("\nCurrent configuration:")
    print(current_config)
    
    # Update the name attribute from "label" to "box_labels"
    new_config = current_config.replace('name="label"', 'name="box_labels"')
    
    # Update configuration
    if new_config != current_config:
        print("\nUpdating configuration...")
        update_project_config(new_config)
    else:
        print("\nNo changes needed in configuration.")

if __name__ == "__main__":
    print("Updating label configuration...")
    update_label_name() 