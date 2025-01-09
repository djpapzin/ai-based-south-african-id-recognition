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

class OCREvaluator:
    def __init__(self, image_dir: str, results_dir: str = "results"):
        """Initialize the OCR evaluator with the directory containing images."""
        global PADDLE_OCR_AVAILABLE
        
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.results = []
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize OCR engines
        logger.info("Initializing OCR engines...")
        self.reader = easyocr.Reader(['en'])
        
        # Initialize optional OCR engines
        self.paddle_ocr = None
        
        if PADDLE_OCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
                PADDLE_OCR_AVAILABLE = False
        
        # Regex pattern for South African ID numbers
        self.id_pattern = r'\b\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4}\d{1}\b'

    def get_id_region(self, image: np.ndarray) -> np.ndarray:
        """Extract the region of interest where ID number is likely to be found."""
        height, width = image.shape[:2]
        
        # ID number is usually in the middle third of the image
        # Adjust these values based on your ID card layout
        roi_y = height // 3
        roi_height = height // 3
        
        # Take the middle 60% of the width
        margin = int(width * 0.2)  # 20% margin from each side
        roi_width = width - 2 * margin
        
        # Extract ROI
        roi = image[roi_y:roi_y+roi_height, margin:margin+roi_width]
        
        return roi

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for OCR."""
        # Calculate new dimensions maintaining aspect ratio
        height = 1200
        aspect_ratio = image.shape[1] / image.shape[0]
        width = int(height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (width, height))
        
        # Get region of interest
        roi = self.get_id_region(resized)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle different lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise to remove speckles
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised

    def extract_id_number(self, text: str) -> Optional[str]:
        """Extract ID number from OCR text using regex."""
        # Clean the text
        cleaned_text = text.replace(" ", "").replace("\n", "").strip()
        
        # First try: Look for exact 13-digit number
        exact_match = re.search(r'\b\d{13}\b', cleaned_text)
        if exact_match:
            return exact_match.group(0)
        
        # Second try: Look for any 13 consecutive digits
        all_numbers = ''.join(re.findall(r'\d+', cleaned_text))
        for i in range(len(all_numbers) - 12):
            potential_id = all_numbers[i:i+13]
            # Validate first 6 digits as date (YYMMDD)
            try:
                year = int(potential_id[:2])
                month = int(potential_id[2:4])
                day = int(potential_id[4:6])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return potential_id
            except ValueError:
                continue
        
        return None

    def process_tesseract(self, image: np.ndarray) -> Tuple[str, Optional[str], float]:
        """Process image with Tesseract OCR."""
        start_time = time.time()
        try:
            # Configure Tesseract to look for digits
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            raw_output = pytesseract.image_to_string(image, config=custom_config)
            extracted_id = self.extract_id_number(raw_output)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            return raw_output, extracted_id, processing_time
        except Exception as e:
            logger.error(f"Tesseract OCR error: {str(e)}")
            return "", None, 0.0

    def process_easyocr(self, image: np.ndarray) -> Tuple[str, Optional[str], float]:
        """Process image with EasyOCR."""
        start_time = time.time()
        try:
            # Get OCR results
            results = self.reader.readtext(image, allowlist='0123456789')
            # Combine all detected text
            raw_output = " ".join([text[1] for text in results])
            extracted_id = self.extract_id_number(raw_output)
            processing_time = (time.time() - start_time) * 1000
            return raw_output, extracted_id, processing_time
        except Exception as e:
            logger.error(f"EasyOCR error: {str(e)}")
            return "", None, 0.0

    def process_paddleocr(self, image: np.ndarray) -> Tuple[str, Optional[str], float]:
        """Process image with PaddleOCR."""
        if not PADDLE_OCR_AVAILABLE:
            return "", None, 0.0
            
        start_time = time.time()
        try:
            # Get OCR results
            results = self.paddle_ocr.ocr(image)
            # Combine all detected text
            if results[0]:
                raw_output = " ".join([line[1][0] for line in results[0]])
            else:
                raw_output = ""
            extracted_id = self.extract_id_number(raw_output)
            processing_time = (time.time() - start_time) * 1000
            return raw_output, extracted_id, processing_time
        except Exception as e:
            logger.error(f"PaddleOCR error: {str(e)}")
            return "", None, 0.0

    def process_single_image(self, image_path: str, ground_truth_id: str) -> Optional[Dict]:
        """Process a single image with all OCR engines."""
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            preprocessed_image = self.preprocess_image(image)
            
            # Process with all OCR engines
            tesseract_output, tesseract_id, tesseract_time = self.process_tesseract(preprocessed_image)
            easyocr_output, easyocr_id, easyocr_time = self.process_easyocr(preprocessed_image)
            paddleocr_output, paddleocr_id, paddleocr_time = self.process_paddleocr(preprocessed_image)
            
            return {
                "image_name": os.path.basename(image_path),
                "folder_name": os.path.basename(os.path.dirname(image_path)),
                "ground_truth_id": ground_truth_id,
                "tesseract_output": tesseract_output,
                "tesseract_extracted_id": tesseract_id,
                "tesseract_time": tesseract_time,
                "easyocr_output": easyocr_output,
                "easyocr_extracted_id": easyocr_id,
                "easyocr_time": easyocr_time,
                "paddleocr_output": paddleocr_output,
                "paddleocr_extracted_id": paddleocr_id,
                "paddleocr_time": paddleocr_time
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def calculate_metrics(self) -> Dict:
        """Calculate evaluation metrics for all OCR engines."""
        if not self.results:
            logger.error("No results to calculate metrics from")
            return {}
            
        metrics = {}
        total = len(self.results)
        
        for engine in ['tesseract', 'easyocr', 'paddleocr']:
            exact_matches = sum(1 for r in self.results 
                              if r[f"{engine}_extracted_id"] == r["ground_truth_id"])
            partial_matches = sum(1 for r in self.results 
                                if r[f"{engine}_extracted_id"] and r[f"{engine}_extracted_id"] != r["ground_truth_id"])
            no_matches = total - exact_matches - partial_matches
            avg_time = sum(r[f"{engine}_time"] for r in self.results) / total
            
            metrics[engine] = {
                "accuracy": (exact_matches / total) * 100,
                "partial_matches": (partial_matches / total) * 100,
                "no_matches": (no_matches / total) * 100,
                "avg_processing_time_ms": avg_time
            }
        
        return metrics

    def generate_report(self):
        """Generate a comprehensive report of the OCR evaluation."""
        if not self.results:
            return "No results to generate report from. Please check if the image directory contains valid images in the correct structure."
            
        metrics = self.calculate_metrics()
        if not metrics:
            return "Could not calculate metrics. No valid results found."
            
        # Save detailed results to JSON in results directory
        json_path = os.path.join(self.results_dir, 'ocr_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Generate summary report in markdown
        report = "# OCR Evaluation Report\n\n"
        report += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Performance table
        report += "## Performance Summary\n\n"
        report += "| Metric | Tesseract | EasyOCR | PaddleOCR |\n"
        report += "|--------|-----------|----------|------------|\n"
        
        metrics_to_show = ['accuracy', 'partial_matches', 'no_matches', 'avg_processing_time_ms']
        metric_names = ['Accuracy', 'Partial Matches', 'No Matches', 'Avg Processing Time (ms)']
        
        for metric, name in zip(metrics_to_show, metric_names):
            report += f"| {name} | {metrics['tesseract'][metric]:.2f}{'%' if 'time' not in metric else ''} | {metrics['easyocr'][metric]:.2f}{'%' if 'time' not in metric else ''} | {metrics['paddleocr'][metric]:.2f}{'%' if 'time' not in metric else ''} |\n"
        
        # Results comparison table
        report += "\n## Results Comparison\n\n"
        report += "| Image | Ground Truth ID | Tesseract | EasyOCR | PaddleOCR |\n"
        report += "|-------|----------------|-----------|----------|------------|\n"
        for result in self.results:
            # Clean up raw outputs for table display
            tesseract_output = result["tesseract_output"].replace("\n", " ").strip()
            easyocr_output = result["easyocr_output"].replace("\n", " ").strip()
            paddleocr_output = result["paddleocr_output"].replace("\n", " ").strip()
            
            # Limit length for readability
            if len(tesseract_output) > 20:
                tesseract_output = tesseract_output[:17] + "..."
            if len(easyocr_output) > 20:
                easyocr_output = easyocr_output[:17] + "..."
            if len(paddleocr_output) > 20:
                paddleocr_output = paddleocr_output[:17] + "..."
                
            report += f"| {result['image_name']} | {result['ground_truth_id']} | {tesseract_output} | {easyocr_output} | {paddleocr_output} |\n"
        
        # Save report as markdown in results directory
        report_path = os.path.join(self.results_dir, 'ocr_evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report

    def evaluate_images(self):
        """Process all images in the directory and evaluate OCR performance."""
        # Check if directory exists
        if not os.path.exists(self.image_dir):
            logger.error(f"Directory not found: {self.image_dir}")
            return

        # Get all ID folders (limit to first 10)
        id_folders = [f for f in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, f))][:10]
        
        if not id_folders:
            logger.error(f"No ID folders found in directory: {self.image_dir}")
            return
            
        logger.info(f"Processing first {len(id_folders)} ID folders")
        
        # Create a list of all images to process
        image_tasks = []
        for id_folder in id_folders:
            folder_path = os.path.join(self.image_dir, id_folder)
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image_tasks.append((image_path, id_folder))
        
        logger.info(f"Found {len(image_tasks)} images to process in the first 10 folders")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_image = {
                executor.submit(self.process_single_image, image_path, ground_truth_id): image_path
                for image_path, ground_truth_id in image_tasks
            }
            
            for future in tqdm(as_completed(future_to_image), total=len(image_tasks), desc="Processing images"):
                result = future.result()
                if result:
                    self.results.append(result)

def main():
    """Main function to run the OCR evaluation."""
    try:
        logger.info("Starting OCR evaluation...")
        
        # Use the specified image directory path
        image_dir = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\IDcopies"
        results_dir = "results"
        
        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Results directory: {results_dir}")
        
        logger.info("Initializing OCR evaluator...")
        evaluator = OCREvaluator(image_dir, results_dir)
        
        logger.info("Starting image evaluation...")
        evaluator.evaluate_images()
        
        logger.info("Generating report...")
        report = evaluator.generate_report()
        
        logger.info("Evaluation complete. Printing report...")
        print(report)
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 