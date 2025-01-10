import os
import easyocr
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.INFO)

def download_models():
    """Download OCR models for offline use"""
    models_dir = "ocr_models"
    easyocr_dir = os.path.join(models_dir, "easyocr")
    paddleocr_dir = os.path.join(models_dir, "paddleocr")
    
    # Create directories if they don't exist
    os.makedirs(easyocr_dir, exist_ok=True)
    os.makedirs(paddleocr_dir, exist_ok=True)
    
    logging.info("Downloading EasyOCR models...")
    try:
        # Initialize EasyOCR to trigger download
        reader = easyocr.Reader(['en'], 
                              gpu=False, 
                              model_storage_directory=easyocr_dir,
                              download_enabled=True)
        logging.info("EasyOCR models downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading EasyOCR models: {str(e)}")
    
    logging.info("Downloading PaddleOCR models...")
    try:
        # Initialize PaddleOCR to trigger download
        ocr = PaddleOCR(use_angle_cls=True, 
                       lang='en',
                       show_log=False,
                       model_dir=paddleocr_dir)
        logging.info("PaddleOCR models downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading PaddleOCR models: {str(e)}")

if __name__ == "__main__":
    download_models() 