import os
import cv2
import json
import time
import re
import numpy as np
import easyocr
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import traceback
from datetime import datetime
import threading
from queue import Queue
from functools import partial
from pdf2image import convert_from_path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting OCR evaluator script...")

# Initialize OCR engines and global flags
PADDLE_OCR_AVAILABLE = False

logger.info("Attempting to import and initialize OCR engines...")

try:
    import pytesseract
    # Set Tesseract path explicitly
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Try to get Tesseract version
    version = pytesseract.get_tesseract_version()
    logger.info(f"Tesseract version: {version}")
except Exception as e:
    logger.error(f"Tesseract not available: {str(e)}")
    logger.error("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    raise SystemExit("Cannot proceed without Tesseract OCR engine.")

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
    logger.info("PaddleOCR imported successfully")
except Exception as e:
    logger.warning(f"PaddleOCR not available: {str(e)}")

logger.info(f"OCR engine availability: Tesseract=True, PaddleOCR={PADDLE_OCR_AVAILABLE}")

# Add this at the top of the script
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

# Add at the top with other constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), "ocr_models")
EASYOCR_MODEL_PATH = os.path.join(MODELS_DIR, "easyocr")
PADDLEOCR_MODEL_PATH = os.path.join(MODELS_DIR, "paddleocr")

# Models can be downloaded using download_ocr_models.py
MODELS_DIR = os.path.join(os.path.dirname(__file__), "ocr_models")
EASYOCR_MODEL_PATH = os.path.join(MODELS_DIR, "easyocr")
PADDLEOCR_MODEL_PATH = os.path.join(MODELS_DIR, "paddleocr")

class OCREvaluator:
    def __init__(self):
        """Initialize OCR engines and other components."""
        logging.info("Initializing OCR engines with local models...")
        
        # Initialize EasyOCR with local models
        try:
            self.reader = easyocr.Reader(['en'], 
                                       gpu=False,
                                       model_storage_directory=EASYOCR_MODEL_PATH,
                                       download_enabled=False)
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {str(e)}")
            raise

        # Initialize PaddleOCR with local models
        try:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True,
                                       lang='en',
                                       show_log=False,
                                       model_dir=PADDLEOCR_MODEL_PATH)
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise

    def is_likely_id_card(self, image, debug=True):
        """
        Determine if an image is likely to contain an ID card based on various features.
        Returns: (bool, dict) - Whether it's an ID card and the metrics used
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape
        aspect_ratio = width / height

        # Calculate text density using adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        text_pixels = np.sum(thresh == 255)
        total_pixels = gray.size
        text_density = (text_pixels / total_pixels) * 100

        # Calculate pattern density using edge detection
        edges = cv2.Canny(gray, 50, 150)
        pattern_density = np.sum(edges == 255) / total_pixels * 100

        # Define more lenient thresholds
        MIN_ASPECT_RATIO = 1.0  # More permissive
        MAX_ASPECT_RATIO = 2.2  # More permissive
        MIN_TEXT_DENSITY = 1.0  # More permissive
        MAX_TEXT_DENSITY = 50   # More permissive
        MIN_PATTERN_DENSITY = 5 # More permissive

        # Check conditions
        aspect_ratio_ok = MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
        text_density_ok = MIN_TEXT_DENSITY <= text_density <= MAX_TEXT_DENSITY
        pattern_density_ok = pattern_density >= MIN_PATTERN_DENSITY

        metrics = {
            'aspect_ratio': aspect_ratio,
            'text_density': text_density,
            'pattern_density': pattern_density,
            'conditions_met': [aspect_ratio_ok, text_density_ok, pattern_density_ok]
        }

        if debug:
            logging.debug(f"ID Card Detection Metrics:")
            logging.debug(f"- Aspect Ratio: {aspect_ratio:.2f} (valid: {aspect_ratio_ok})")
            logging.debug(f"- Text Density: {text_density:.2f}% (valid: {text_density_ok})")
            logging.debug(f"- Pattern Density: {pattern_density:.2f} (valid: {pattern_density_ok})")

        # Consider it an ID card if at least 2 conditions are met
        conditions_met = sum([aspect_ratio_ok, text_density_ok, pattern_density_ok])
        return True, metrics  # Always return True to process all images

    def preprocess_image(self, image, save_path=None):
        """Enhanced preprocessing pipeline for ID card images with rotation correction"""
        # Check if image needs rotation based on aspect ratio
        height, width = image.shape[:2]
        current_ratio = width / height
        
        # South African ID cards are typically wider than tall
        if current_ratio < 1:  # Image is taller than wide
            # Rotate 90 degrees clockwise
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            logging.info("Rotated image 90 degrees clockwise (portrait to landscape)")
        
        # Continue with existing preprocessing steps...
        # Resize image while maintaining aspect ratio
        target_height = 1200
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)
        image = cv2.resize(image, (target_width, target_height))
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Denoise using bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding with reduced block size
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 9, 2)

        # Morphological operations
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Add white border
        border_size = 20
        binary = cv2.copyMakeBorder(binary, border_size, border_size, border_size, border_size,
                                   cv2.BORDER_CONSTANT, value=255)

        # Save preprocessed image if path is provided
        if save_path:
            cv2.imwrite(save_path, binary)

        return binary

    def perform_tesseract_ocr(self, image, timeout=30):
        """Perform OCR using Tesseract with timeout"""
        try:
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            
            # Create a queue for the result
            result_queue = Queue()
            
            def ocr_task():
                try:
                    text = pytesseract.image_to_string(image, config=custom_config)
                    result_queue.put(''.join(filter(str.isdigit, text)))
                except Exception as e:
                    result_queue.put(None)
            
            # Start OCR in a thread
            thread = threading.Thread(target=ocr_task)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout
            thread.join(timeout)
            if thread.is_alive():
                logging.warning("Tesseract OCR timed out")
                return ""
            
            # Get result
            result = result_queue.get_nowait() if not result_queue.empty() else ""
            return result if result is not None else ""
            
        except Exception as e:
            logging.error(f"Tesseract OCR error: {str(e)}")
            return ""

    def perform_easyocr(self, image):
        """Perform OCR using EasyOCR with optimized settings"""
        try:
            # Get results
            results = self.reader.readtext(image)
            
            # Extract and concatenate all numeric text
            text = ''
            for detection in results:
                # detection is a tuple of (bbox, text, prob)
                potential_id = ''.join(filter(str.isdigit, detection[1]))
                if len(potential_id) > 0:
                    text += potential_id
            
            logging.debug(f"EasyOCR raw output: {text}")
            return text
        except Exception as e:
            logging.error(f"EasyOCR error: {str(e)}")
            return ""

    def perform_paddleocr(self, image):
        """Perform OCR using PaddleOCR with optimized settings"""
        try:
            # Get results
            results = self.paddle_ocr.ocr(image, det=True, rec=True)
            
            # Extract and concatenate all numeric text
            text = ''
            if results is not None and len(results) > 0:  # Check if results exist and not empty
                for line in results:
                    if line is not None:  # Additional check for None
                        for detection in line:
                            if detection is not None and len(detection) >= 2:  # Ensure detection has required elements
                                # detection is a tuple of (bbox, (text, prob))
                                potential_id = ''.join(filter(str.isdigit, detection[1][0]))
                                if len(potential_id) > 0:
                                    text += potential_id
            
            logging.debug(f"PaddleOCR raw output: {text}")
            return text
        except Exception as e:
            logging.error(f"PaddleOCR error: {str(e)}")
            return ""  # Return empty string instead of failing

    def process_image(self, image_path, ground_truth, save_dir=None):
        """Process a single image and return OCR results"""
        try:
            # Load and preprocess full image
            image = cv2.imread(image_path)
            processed = self.preprocess_image(image)

            # Extract ROI coordinates
            height, width = processed.shape
            roi_y = height // 3
            roi_height = height // 3
            margin = int(width * 0.2)
            
            # Draw ROI rectangle on preprocessed image
            processed_with_roi = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(processed_with_roi, 
                         (margin, roi_y), 
                         (width - margin, roi_y + roi_height), 
                         (0, 255, 0), 2)

            # Save preprocessed image with ROI marked
            if save_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                preprocessed_path = os.path.join(save_dir, f"{base_name}_preprocessed.png")
                cv2.imwrite(preprocessed_path, processed_with_roi)

            # Extract and save ROI
            roi = processed[roi_y:roi_y+roi_height, margin:width-margin]
            if save_dir:
                roi_path = os.path.join(save_dir, f"{base_name}_roi.png")
                cv2.imwrite(roi_path, roi)

            # Perform OCR with all engines
            results = {}
            
            # Tesseract OCR
            try:
                results['tesseract'] = self.perform_tesseract_ocr(roi)
            except Exception as e:
                logging.error(f"Tesseract OCR failed: {str(e)}")
                results['tesseract'] = ""

            # EasyOCR
            try:
                results['easyocr'] = self.perform_easyocr(roi)
            except Exception as e:
                logging.error(f"EasyOCR failed: {str(e)}")
                results['easyocr'] = ""

            # PaddleOCR
            try:
                results['paddleocr'] = self.perform_paddleocr(roi)
            except Exception as e:
                logging.error(f"PaddleOCR failed: {str(e)}")
                results['paddleocr'] = ""

            return results
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None

def validate_id_number(id_number: str) -> bool:
    """
    Validate South African ID number using the Luhn algorithm and checking date validity.
    Returns True if the ID number is valid, False otherwise.
    """
    try:
        if not id_number or not id_number.isdigit() or len(id_number) != 13:
            return False

        # Extract components
        birth_date = id_number[:6]
        gender = int(id_number[6:10])
        citizenship = int(id_number[10])
        race = int(id_number[11])
        checksum = int(id_number[12])

        # Validate birth date
        try:
            year = int(birth_date[:2])
            month = int(birth_date[2:4])
            day = int(birth_date[4:])
            
            # Assume 19xx for birth years before current year
            current_year = datetime.now().year % 100
            if year > current_year:
                year += 1900
            else:
                year += 2000
                
            datetime(year, month, day)
        except ValueError:
            return False

        # Validate gender (0000-4999 for female, 5000-9999 for male)
        if not (0 <= gender <= 9999):
            return False

        # Validate citizenship (0 for SA citizen, 1 for permanent resident)
        if citizenship not in [0, 1]:
            return False

        # Validate race (8 or 9 for Cape Coloured)
        if race not in range(10):
            return False

        # Calculate checksum using Luhn algorithm
        total = 0
        for i, digit in enumerate(id_number[:-1]):
            num = int(digit)
            if i % 2 == 0:
                total += num
            else:
                doubled = num * 2
                total += doubled if doubled < 10 else doubled - 9

        check = (10 - (total % 10)) % 10
        return check == checksum

    except Exception as e:
        logging.error(f"Error validating ID number: {str(e)}")
        return False

def process_directory(evaluator, image_dir, results_dir="results", max_workers=4, test_mode=True):
    """Process all images in a directory and generate a report."""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    preprocessed_dir = os.path.join(results_dir, "preprocessed_images")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Get all ID folders
    id_folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    
    if not id_folders:
        logging.error(f"No ID folders found in directory: {image_dir}")
        return
    
    # In test mode, only process first 10 folders
    if test_mode:
        id_folders = id_folders[:10]
        logging.info(f"Test mode: Processing first 10 folders out of {len(id_folders)} total folders")
    else:
        logging.info(f"Found {len(id_folders)} ID folders")
    
    results = []
    results_lock = threading.Lock()
    
    def process_folder(id_folder):
        folder_results = []
        try:
            # Create corresponding folder in preprocessed_dir
            folder_preprocessed_path = os.path.join(preprocessed_dir, id_folder)
            os.makedirs(folder_preprocessed_path, exist_ok=True)
            
            folder_path = os.path.join(image_dir, id_folder)
            
            # Get all files (including PDFs now)
            all_files = [f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.jpeg', '.jpg', '.png', '.pdf'))]
            
            for file in all_files:
                file_path = os.path.join(folder_path, file)
                
                # Handle PDFs
                if file.lower().endswith('.pdf'):
                    try:
                        # Convert PDF pages to images with explicit poppler path
                        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                        
                        for i, image in enumerate(images):
                            # Create a unique temporary file name
                            temp_dir = os.path.join(folder_preprocessed_path, 'temp')
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_path = os.path.join(temp_dir, f"{os.path.splitext(file)[0]}_page{i+1}.png")
                            
                            try:
                                # Save the image
                                image.save(temp_path, 'PNG')
                                logging.info(f"Processing PDF page {i+1} from {file_path}")
                                
                                # Process the temporary image
                                result = evaluator.process_image(
                                    temp_path,
                                    id_folder,
                                    folder_preprocessed_path
                                )
                                
                                if result:
                                    folder_results.append({
                                        "image_name": f"{os.path.splitext(file)[0]}_page{i+1}",
                                        "folder_name": id_folder,
                                        "ground_truth_id": id_folder,
                                        "source_type": "pdf",
                                        "page_number": i+1,
                                        "tesseract_output": result.get('tesseract', ''),
                                        "tesseract_extracted_id": result.get('tesseract', ''),
                                        "easyocr_output": result.get('easyocr', ''),
                                        "easyocr_extracted_id": result.get('easyocr', ''),
                                        "paddleocr_output": result.get('paddleocr', ''),
                                        "paddleocr_extracted_id": result.get('paddleocr', '')
                                    })
                            finally:
                                # Clean up temporary file
                                try:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                except Exception as e:
                                    logging.warning(f"Could not remove temporary file {temp_path}: {str(e)}")
                                
                    except Exception as e:
                        logging.error(f"Error processing PDF {file_path}: {str(e)}")
                        continue
                    finally:
                        # Clean up temp directory if empty
                        try:
                            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                                os.rmdir(temp_dir)
                        except Exception:
                            pass
                
                # Handle regular images
                else:
                    logging.info(f"Processing image {file_path}")
                    try:
                        result = evaluator.process_image(
                            file_path,
                            id_folder,
                            folder_preprocessed_path
                        )
                        
                        if result:
                            folder_results.append({
                                "image_name": file,
                                "folder_name": id_folder,
                                "ground_truth_id": id_folder,
                                "source_type": "image",
                                "tesseract_output": result.get('tesseract', ''),
                                "tesseract_extracted_id": result.get('tesseract', ''),
                                "easyocr_output": result.get('easyocr', ''),
                                "easyocr_extracted_id": result.get('easyocr', ''),
                                "paddleocr_output": result.get('paddleocr', ''),
                                "paddleocr_extracted_id": result.get('paddleocr', '')
                            })
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {str(e)}")
                        continue
                
            return folder_results
        except Exception as e:
            logging.error(f"Error processing folder {id_folder}: {str(e)}")
            return []

    # Process folders in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(process_folder, folder): folder for folder in id_folders}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_folder), total=len(id_folders), desc="Processing folders"):
            folder = future_to_folder[future]
            try:
                folder_results = future.result()
                with results_lock:
                    results.extend(folder_results)
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {str(e)}")
    
    # Generate report
    generate_report(results, results_dir, preprocessed_dir)

def generate_report(results, results_dir, preprocessed_dir):
    """Generate a comprehensive report of the OCR evaluation."""
    if not results:
        print("No results to generate report from. Please check if the image directory contains valid images.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save detailed results to JSON
    json_path = os.path.join(results_dir, 'ocr_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate markdown report
    report = "# OCR Evaluation Report\n\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add preprocessing information
    report += "## Image Preprocessing Steps\n\n"
    report += "The following preprocessing steps were applied to each image:\n\n"
    report += "1. **Resize**: Images are resized to a height of 1200px while maintaining aspect ratio\n"
    report += "2. **Region of Interest (ROI) Extraction**:\n"
    report += "   - Middle third of the image vertically (height/3 to 2*height/3)\n"
    report += "   - Middle 60% of the image horizontally (20% margin from each side)\n"
    report += "   - This targeting is based on the standard South African ID layout where the ID number\n"
    report += "     is typically located in the middle section of the document\n"
    report += "3. **Grayscale Conversion**: Converted to grayscale for better text recognition\n"
    report += "4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:\n"
    report += "   - Improves local contrast\n"
    report += "   - Helps with varying lighting conditions\n"
    report += "5. **Bilateral Filtering**: Reduces noise while preserving edges\n"
    report += "6. **Adaptive Thresholding**:\n"
    report += "   - Creates binary image\n"
    report += "   - Uses local area (9x9 pixels) to determine threshold\n"
    report += "7. **Morphological Operations**:\n"
    report += "   - Opening: Removes small noise\n"
    report += "   - Closing: Connects nearby text components\n"
    report += "8. **Border Addition**: 20px white border added to prevent text touching edges\n\n"
    
    report += f"Preprocessed images are saved in: `{preprocessed_dir}`\n\n"
        
    # Performance table
    report += "## Performance Summary\n\n"
    report += "| Metric | Tesseract | EasyOCR | PaddleOCR |\n"
    report += "|--------|-----------|----------|------------|\n"
    
    for metric in ['accuracy', 'partial_matches', 'no_matches']:
        report += f"| {metric.replace('_', ' ').title()} | {metrics['tesseract'][metric]:.2f}% | {metrics['easyocr'][metric]:.2f}% | {metrics['paddleocr'][metric]:.2f}% |\n"
    
    # Results comparison table
    report += "\n## Results Comparison\n\n"
    report += "| Image | Ground Truth ID | Tesseract | EasyOCR | PaddleOCR |\n"
    report += "|-------|----------------|-----------|----------|------------|\n"
    
    for result in results:
        report += f"| {result['image_name']} | {result['ground_truth_id']} | {result['tesseract_output']} | {result['easyocr_output']} | {result['paddleocr_output']} |\n"
    
    # Add source type statistics
    report += "\n## Source Type Statistics\n\n"
    pdf_count = sum(1 for r in results if r.get('source_type') == 'pdf')
    image_count = sum(1 for r in results if r.get('source_type') == 'image')
    total_count = len(results)
    
    report += f"- PDF Documents: {pdf_count} ({(pdf_count/total_count)*100:.1f}%)\n"
    report += f"- Image Files: {image_count} ({(image_count/total_count)*100:.1f}%)\n\n"
    
    # Save report
    report_path = os.path.join(results_dir, 'ocr_evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report generated: {report_path}")

def calculate_metrics(results):
    """Calculate evaluation metrics for all OCR engines."""
    metrics = {}
    total = len(results)
    
    for engine in ['tesseract', 'easyocr', 'paddleocr']:
        exact_matches = sum(1 for r in results 
                          if r[f"{engine}_extracted_id"] == r["ground_truth_id"])
        partial_matches = sum(1 for r in results 
                            if r[f"{engine}_extracted_id"] and 
                            r[f"{engine}_extracted_id"] != r["ground_truth_id"])
        no_matches = total - exact_matches - partial_matches
        
        metrics[engine] = {
            "accuracy": (exact_matches / total) * 100,
            "partial_matches": (partial_matches / total) * 100,
            "no_matches": (no_matches / total) * 100
        }
    
    return metrics

def main():
    """Main function to run the OCR evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate OCR performance on ID card images.')
    parser.add_argument('image_dir', help='Directory containing ID card images')
    parser.add_argument('--results-dir', default='results', help='Directory to store results')
    parser.add_argument('--test-mode', action='store_true', help='Process only first 10 folders')
    args = parser.parse_args()

    try:
        # Initialize evaluator
        logging.info("Initializing OCR evaluator...")
        evaluator = OCREvaluator()
        
        # Process images and generate report
        logging.info("Starting image evaluation...")
        process_directory(evaluator, args.image_dir, args.results_dir, test_mode=args.test_mode)
        
        logging.info("Evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 