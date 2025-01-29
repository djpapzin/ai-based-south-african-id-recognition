import requests
import json

# Configuration
LABEL_STUDIO_URL = "http://localhost:8090"
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
PROJECT_ID = "1"  # Using the first project since you mentioned you have 1 project

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

# Get current project configuration
def get_project_config():
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting project config: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

# Update project labels
def update_project_labels(new_config):
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    response = requests.patch(url, headers=headers, json={"label_config": new_config})
    if response.status_code == 200:
        print("Labels updated successfully!")
        return response.json()
    else:
        print(f"Error updating labels: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

# Main execution
if __name__ == "__main__":
    # Get current configuration
    project = get_project_config()
    if project:
        current_config = project["label_config"]
        print("Current configuration:")
        print(current_config)
        
        # Update both labels
        new_config = current_config.replace('"New"', '"New ID Card"').replace('"Old"', '"Old ID Book"')
        
        # Update with new configuration
        update_project_labels(new_config) 