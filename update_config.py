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

def update_project_config():
    """Update project configuration"""
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    
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
    
    response = requests.patch(url, headers=headers, json={"label_config": new_config})
    if response.status_code == 200:
        print("Project configuration updated successfully!")
        return True
    else:
        print(f"Error updating configuration: {response.status_code}")
        print(f"Response: {response.text}")
        return False

if __name__ == "__main__":
    print("Updating project configuration...")
    update_project_config() 