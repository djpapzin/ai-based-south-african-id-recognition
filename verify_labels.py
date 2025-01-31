import requests
import json
from pathlib import Path

# Label Studio configuration
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
BASE_URL = "http://localhost:8080/api"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def get_project_export(project_id):
    url = f"{BASE_URL}/projects/{project_id}/export"
    try:
        print(f"Attempting to connect to: {url}")
        
        response = requests.get(
            url,
            headers=HEADERS,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully retrieved project export")
            return data
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Label Studio: {str(e)}")
        return None

def analyze_annotations(annotations):
    if not annotations:
        print("No annotations data available to analyze")
        return

    print(f"\nProcessing {len(annotations)} annotations...")
    
    for item in annotations:
        image_path = item.get("data", {}).get("image", "")
        print(f"\nImage: {Path(image_path).name}")
        
        annotations = item.get("annotations", [])
        if not annotations:
            print("No annotations found for this image")
            continue
            
        for annotation in annotations:
            results = annotation.get("result", [])
            if not results:
                print("No results found in this annotation")
                continue
                
            print(f"Processing {len(results)} annotations")
            
            # Separate rectangle and keypoint labels
            rectangle_labels = {}
            corner_labels = set()
            
            for result in results:
                try:
                    result_type = result.get("type")
                    
                    if result_type == "rectanglelabels":
                        # Handle rectangle labels
                        label = result.get("value", {}).get("rectanglelabels", [])[0]
                        if label:
                            if label in rectangle_labels:
                                rectangle_labels[label].append(result)
                            else:
                                rectangle_labels[label] = [result]
                    elif result_type == "keypointlabels":
                        # Handle corner keypoints
                        label = result.get("value", {}).get("keypointlabels", [])[0]
                        if label:
                            corner_labels.add(label)
                            
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
            
            if rectangle_labels:
                print("\nRectangle labels found:")
                for label, instances in sorted(rectangle_labels.items()):
                    print(f"- {label}: {len(instances)} instance(s)")
                    
            if corner_labels:
                print("\nCorner labels found:")
                for corner in sorted(corner_labels):
                    print(f"- {corner}")
                    
            # Check specifically for corner labels
            expected_corners = {"top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner"}
            missing_corners = expected_corners - corner_labels
            if missing_corners:
                print("\nMissing corner labels:")
                for corner in sorted(missing_corners):
                    print(f"- {corner}")
            else:
                print("\nAll corner labels are present!")

def main():
    print("Fetching project export from Label Studio...")
    data = get_project_export(7)  # Using project ID 7
    
    if data:
        analyze_annotations(data)
    else:
        print("No data found in the project export")

if __name__ == "__main__":
    main()
