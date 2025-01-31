import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class DatasetVerifier:
    def __init__(self, json_path, images_dir):
        """Initialize the dataset verifier with paths to annotations and images."""
        self.json_path = json_path
        self.images_dir = images_dir
        self.current_index = 0
        self.annotations = None
        self.images = None
        self.class_colors = {}
        self.fig = None
        self.ax = None
        
    def load_dataset(self):
        """Load the Label Studio JSON annotations."""
        print(f"Loading annotations from {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        # Extract images and annotations
        self.annotations = data
        print(f"\nDataset Statistics:")
        print(f"Number of images: {len(data['images'])}")
        print(f"Number of annotations: {len(data['annotations'])}")
        print(f"Categories: {[cat['name'] for cat in data['categories']]}")
        
        # Create color map for classes
        for category in data['categories']:
            self.class_colors[category['id']] = np.random.rand(3,)
    
    def draw_annotations(self, image, annotations):
        """Draw bounding boxes and labels on the image."""
        img = image.copy()
        
        for ann in annotations:
            # Get bbox coordinates
            bbox = ann['bbox']
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Get category
            category_id = ann['category_id']
            category_name = next(cat['name'] for cat in self.annotations['categories'] 
                               if cat['id'] == category_id)
            
            # Get color for this class
            color = self.class_colors[category_id]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{category_name}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
        return img
    
    def show_image(self, index):
        """Display an image with its annotations."""
        if not 0 <= index < len(self.annotations['images']):
            print(f"Invalid image index: {index}")
            return
        
        # Get image info
        img_info = self.annotations['images'][index]
        img_path = os.path.join(self.images_dir, os.path.basename(img_info['file_name']))
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            return
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        img_anns = [ann for ann in self.annotations['annotations'] 
                   if ann['image_id'] == img_info['id']]
        
        # Draw annotations
        img_with_anns = self.draw_annotations(img, img_anns)
        
        # Clear previous plot
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create new plot
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.imshow(img_with_anns)
        self.ax.axis('off')
        self.fig.suptitle(f"Image {index + 1}/{len(self.annotations['images'])}: {os.path.basename(img_path)}")
        
        # Add annotation details
        details = []
        for ann in img_anns:
            category = next(cat['name'] for cat in self.annotations['categories'] 
                          if cat['id'] == ann['category_id'])
            details.append(f"{category}: {ann['bbox']}")
        
        plt.figtext(0.1, 0.02, '\n'.join(details), fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()
    
    def on_key_press(self, event):
        """Handle keyboard events for navigation."""
        if event.key == 'right':
            self.next_image()
        elif event.key == 'left':
            self.previous_image()
        elif event.key == 'q':
            plt.close('all')
    
    def next_image(self):
        """Show next image."""
        self.current_index = min(self.current_index + 1, 
                               len(self.annotations['images']) - 1)
        self.show_image(self.current_index)
    
    def previous_image(self):
        """Show previous image."""
        self.current_index = max(self.current_index - 1, 0)
        self.show_image(self.current_index)
    
    def start_verification(self):
        """Start the verification process."""
        print("\nStarting verification...")
        print("Controls:")
        print("- Right arrow: Next image")
        print("- Left arrow: Previous image")
        print("- Q: Quit")
        
        self.show_image(0)
        
        # Connect keyboard events
        if self.fig is not None:
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

def main():
    parser = argparse.ArgumentParser(description='Verify Label Studio annotations')
    parser.add_argument('json_path', help='Path to the Label Studio JSON annotations file')
    parser.add_argument('images_dir', help='Path to the directory containing images')
    
    args = parser.parse_args()
    
    verifier = DatasetVerifier(args.json_path, args.images_dir)
    verifier.load_dataset()
    verifier.start_verification()

if __name__ == "__main__":
    main() 