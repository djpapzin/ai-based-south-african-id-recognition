import requests
import json
from collections import defaultdict
import os
from datetime import datetime

def check_duplicates():
    # Your Label Studio configuration
    label_studio_url = "http://localhost:8080"
    project_id = "1"
    auth_token = "c4965d5c5515d76644b18343d198685d738ce34a"
    
    # API endpoint for tasks
    url = f"{label_studio_url}/api/projects/{project_id}/tasks?page_size=1000"
    
    # Headers for authentication
    headers = {
        "Authorization": f"Token {auth_token}"
    }
    
    try:
        # Get all tasks
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tasks = response.json()
        
        # Initialize counters for labeled files
        labeled_files = {
            'New': [],
            'Old': [],
            'Unlabeled': []
        }
        
        # Process tasks
        for task in tasks:
            image_path = task.get('data', {}).get('image')
            if not image_path:
                print(f"Warning: No image path found for task {task.get('id')}")
                continue
            
            # Check if task has any annotations
            annotations = task.get('annotations', [])
            if annotations:
                # Get the most recent non-cancelled annotation
                valid_annotations = [a for a in annotations if not a.get('was_cancelled', False)]
                if valid_annotations:
                    latest_annotation = max(valid_annotations, key=lambda x: x.get('created_at', ''))
                    results = latest_annotation.get('result', [])
                    
                    # Look for rectanglelabels in the results
                    label_found = False
                    for result in results:
                        if result.get('type') == 'rectanglelabels':
                            labels = result.get('value', {}).get('rectanglelabels', [])
                            if labels:
                                label = labels[0]  # Take the first label
                                if label in ['New', 'Old']:
                                    labeled_files[label].append(image_path)
                                    label_found = True
                                    break
                    
                    if not label_found:
                        labeled_files['Unlabeled'].append(image_path)
                else:
                    labeled_files['Unlabeled'].append(image_path)
            else:
                labeled_files['Unlabeled'].append(image_path)
        
        print("\nChecking for duplicates...\n")
        all_files = labeled_files['New'] + labeled_files['Old'] + labeled_files['Unlabeled']
        duplicates = [item for item in all_files if all_files.count(item) > 1]
        
        if duplicates:
            print("Duplicates found:")
            for duplicate in set(duplicates):
                print(f"File {duplicate} appears {all_files.count(duplicate)} times")
        else:
            print("No duplicates found!")
        
        print(f"\nSummary:")
        print(f"Total unique files: {len(set(all_files))}")
        print(f"Total tasks: {len(all_files)}")
        
        print(f"\nPage type distribution:")
        a_pages = len([f for f in all_files if '_A_' in f or '_A.' in f])
        b_pages = len([f for f in all_files if '_B_' in f or '_B.' in f])
        c_pages = len([f for f in all_files if '_C_' in f or '_C.' in f])
        print(f"A pages: {a_pages}")
        print(f"B pages: {b_pages}")
        print(f"C pages: {c_pages}")
        
        print(f"\nLabeling distribution:")
        print(f"New: {len(labeled_files['New'])} files")
        print(f"Old: {len(labeled_files['Old'])} files")
        print(f"Unlabeled: {len(labeled_files['Unlabeled'])} files")
        
        print("\nDetailed Label Information:\n")
        print("New files ({}):".format(len(labeled_files['New'])))
        for f in sorted(labeled_files['New']):
            print(f)
        
        print("\nOld files ({}):".format(len(labeled_files['Old'])))
        for f in sorted(labeled_files['Old']):
            print(f)
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    check_duplicates() 