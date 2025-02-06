import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

class DocumentClassifier:
    def __init__(self, model_path, device=None):
        """Initialize the document classifier.
        
        Args:
            model_path (str): Path to the classification model weights
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model (assuming ResNet50 architecture)
        self.model = models.resnet50(pretrained=False)
        # Modify the final layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        # Load model weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded classification model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Class labels
        self.classes = ['old_id', 'new_id']
    
    def preprocess_image(self, image_path):
        """Preprocess an image for classification.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image {image_path}: {str(e)}")
    
    def classify(self, image_path):
        """Classify an ID document as old or new.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.classes[predicted.item()]
                confidence_score = confidence.item()
                
                return predicted_class, confidence_score
        except Exception as e:
            raise RuntimeError(f"Classification failed for {image_path}: {str(e)}")
    
    def batch_classify(self, image_paths, batch_size=8):
        """Classify multiple images in batches.
        
        Args:
            image_paths (list): List of image file paths
            batch_size (int): Number of images to process at once
            
        Returns:
            list: List of (predicted_class, confidence_score) tuples
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Warning: Failed to process {path}: {str(e)}")
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack tensors and run inference
            batch_tensor = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Store results
                for j, (pred, conf) in enumerate(zip(predicted, confidence)):
                    predicted_class = self.classes[pred.item()]
                    confidence_score = conf.item()
                    results.append((predicted_class, confidence_score))
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    model_path = "classification_model_final.pth"
    classifier = DocumentClassifier(model_path)
    
    # Single image classification
    image_path = "test_images/sample_id.jpg"
    if os.path.exists(image_path):
        predicted_class, confidence = classifier.classify(image_path)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
