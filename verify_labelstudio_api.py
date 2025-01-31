import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import tempfile
from urllib.parse import urljoin
import argparse

class LabelStudioVerifier:
    def __init__(self, api_token, host="http://localhost:8080"):
        """Initialize the Label Studio API verifier."""
        self.host = host
        self.headers = {
            "Authorization": f"Token {api_token}"
        }
        self.current_index = 0
        self.tasks = []
        
        # Define fixed colors for each label
        self.class_colors = {
            # Rectangle labels
            "id_document": (0, 0, 1),      # blue
            "surname": (1, 0, 0),          # red
            "names": (0, 1, 0),            # green
            "sex": (1, 1, 0),              # yellow
            "nationality": (1, 0.5, 0),    # orange
            "id_number": (0.5, 0, 0.5),    # purple
            "date_of_birth": (0, 1, 1),    # cyan
            "country_of_birth": (1, 0, 1), # magenta
            "citizenship_status": (0.5, 0.5, 0.5), # gray
            "face": (0.6, 0.3, 0),         # brown
            "signature": (1, 0.7, 0.7),    # pink
            
            # Keypoint labels
            "top_left_corner": (1, 0, 0),      # red
            "top_right_corner": (0, 1, 0),     # green
            "bottom_left_corner": (0, 0, 1),   # blue
            "bottom_right_corner": (1, 1, 0)   # yellow
        }
        
        self.fig = None
        self.ax = None
        plt.ion()  # Enable interactive mode
        
    def get_projects(self):
        """Get list of all projects."""
        url = urljoin(self.host, "/api/projects")
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            print("Projects response:", data)  # Debug print
            return data
        else:
            print(f"Failed to get projects: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    def get_project_tasks(self, project_id):
        """Get all tasks for a project."""
        url = urljoin(self.host, f"/api/tasks?project={project_id}")
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            print("Tasks response:", data)  # Debug print
            if isinstance(data, dict):
                return data.get('tasks', [])
            return data
        else:
            print(f"Failed to get tasks: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    
    def download_image(self, url):
        """Download image from URL."""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp.write(response.content)
                temp.close()
                return temp.name
            print(f"Failed to download image: {response.status_code}")
            return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def load_project(self, project_id):
        """Load project tasks and annotations."""
        print(f"Loading project {project_id}...")
        self.tasks = self.get_project_tasks(project_id)
        print(f"Loaded {len(self.tasks)} tasks")
        
        print(f"\nProject Statistics:")
        print(f"Number of tasks: {len(self.tasks)}")
        print(f"Available labels: {list(self.class_colors.keys())}")
    
    def draw_annotations(self, image, annotations):
        """Draw bounding boxes and keypoints on the image."""
        img = image.copy()
        
        if not annotations:
            return img
            
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                value = result.get('value', {})
                if not isinstance(value, dict):
                    continue
                    
                try:
                    # Handle rectangle labels
                    if result.get('type') == 'rectanglelabels':
                        x = value.get('x', 0) / 100.0 * img.shape[1]
                        y = value.get('y', 0) / 100.0 * img.shape[0]
                        width = value.get('width', 0) / 100.0 * img.shape[1]
                        height = value.get('height', 0) / 100.0 * img.shape[0]
                        
                        labels = value.get('labels', [])
                        if not labels:
                            continue
                            
                        label = labels[0]
                        if label not in self.class_colors:
                            continue
                            
                        color = self.class_colors[label]
                        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        
                        cv2.rectangle(img, 
                                    (int(x), int(y)), 
                                    (int(x + width), int(y + height)), 
                                    color, 2)
                        
                        # Add label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(img,
                                    (int(x), int(y - 25)),
                                    (int(x + label_size[0] + 10), int(y)),
                                    color, -1)
                        cv2.putText(img, label,
                                  (int(x + 5), int(y - 8)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)
                    
                    # Handle keypoint labels
                    elif result.get('type') == 'keypointlabels':
                        x = value.get('x', 0) / 100.0 * img.shape[1]
                        y = value.get('y', 0) / 100.0 * img.shape[0]
                        
                        labels = value.get('labels', [])
                        if not labels:
                            continue
                            
                        label = labels[0]
                        if label not in self.class_colors:
                            continue
                            
                        color = self.class_colors[label]
                        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        
                        # Draw keypoint
                        cv2.circle(img, (int(x), int(y)), 5, color, -1)
                        cv2.circle(img, (int(x), int(y)), 8, color, 2)
                        
                        # Add label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(img,
                                    (int(x), int(y - 25)),
                                    (int(x + label_size[0] + 10), int(y)),
                                    color, -1)
                        cv2.putText(img, label,
                                  (int(x + 5), int(y - 8)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)
                        
                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    continue
        
        return img
    
    def show_task(self, index):
        """Display a task with its annotations."""
        if not 0 <= index < len(self.tasks):
            print(f"Invalid task index: {index}")
            return
        
        task = self.tasks[index]
        print(f"\nShowing task {index + 1}/{len(self.tasks)}")
        
        # Get image URL
        image_url = task.get('data', {}).get('image')
        if not image_url:
            print("No image URL found in task")
            return
            
        if not image_url.startswith('http'):
            image_url = urljoin(self.host, image_url)
            
        print(f"Downloading image from: {image_url}")
        temp_image_path = self.download_image(image_url)
        if temp_image_path is None:
            print(f"Error downloading image from {image_url}")
            return
            
        img = cv2.imread(temp_image_path)
        os.unlink(temp_image_path)  # Clean up temporary file
        
        if img is None:
            print(f"Error loading image from {temp_image_path}")
            return
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw annotations
        annotations = task.get('annotations', [])
        print(f"Found {len(annotations)} annotations")
        img_with_anns = self.draw_annotations(img, annotations)
        
        # Clear previous plot
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create new plot
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.imshow(img_with_anns)
        self.ax.axis('off')
        self.fig.suptitle(f"Task {index + 1}/{len(self.tasks)}")
        
        # Add annotation details
        details = []
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                if isinstance(result, dict):
                    value = result.get('value', {})
                    if isinstance(value, dict):
                        labels = value.get('labels', [])
                        if labels:
                            label = labels[0]
                            if result.get('type') == 'rectanglelabels':
                                coords = f"x={value.get('x', 0):.1f}%, y={value.get('y', 0):.1f}%, w={value.get('width', 0):.1f}%, h={value.get('height', 0):.1f}%"
                            else:  # keypointlabels
                                coords = f"x={value.get('x', 0):.1f}%, y={value.get('y', 0):.1f}%"
                            details.append(f"{label}: {coords}")
        
        plt.figtext(0.1, 0.02, '\n'.join(details), fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot
    
    def on_key_press(self, event):
        """Handle keyboard events for navigation."""
        if event.key == 'right':
            self.next_task()
        elif event.key == 'left':
            self.previous_task()
        elif event.key == 'q':
            plt.close('all')
    
    def next_task(self):
        """Show next task."""
        if self.current_index < len(self.tasks) - 1:
            self.current_index += 1
            self.show_task(self.current_index)
    
    def previous_task(self):
        """Show previous task."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_task(self.current_index)
    
    def start_verification(self):
        """Start the verification process."""
        if not self.tasks:
            print("No tasks loaded. Please load a project first.")
            return
            
        print("\nStarting verification...")
        print("Controls:")
        print("- Right arrow: Next task")
        print("- Left arrow: Previous task")
        print("- Q: Quit")
        
        self.show_task(0)
        plt.show(block=True)  # Block until all windows are closed

def main():
    parser = argparse.ArgumentParser(description='Verify Label Studio annotations using API')
    parser.add_argument('--token', required=True, help='Label Studio API token')
    parser.add_argument('--host', default='http://localhost:8080', help='Label Studio host URL')
    parser.add_argument('--project', required=True, type=int, help='Project ID to verify')
    
    args = parser.parse_args()
    
    verifier = LabelStudioVerifier(args.token, args.host)
    
    # List available projects
    projects = verifier.get_projects()
    print("\nAvailable projects:")
    if isinstance(projects, dict) and 'results' in projects:
        projects = projects['results']
    
    if isinstance(projects, list):
        for project in projects:
            if isinstance(project, dict):
                print(f"ID: {project.get('id')}, Title: {project.get('title', 'No title')}")
    else:
        print("No projects found or unexpected response format")
        print(f"Response type: {type(projects)}")
        print(f"Response content: {projects}")
        return
    
    # Load specified project
    verifier.load_project(args.project)
    verifier.start_verification()

if __name__ == "__main__":
    main() 