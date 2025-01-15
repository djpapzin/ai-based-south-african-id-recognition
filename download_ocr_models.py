import os
import easyocr
from paddleocr import PaddleOCR
import logging

logging.basicConfig(level=logging.INFO)

def check_models_exist(easyocr_dir, paddleocr_dir):
    """Check if models already exist in the directories"""
    # Check EasyOCR models (common model files)
    easyocr_files = [
        "craft_mlt_25k.pth",
        "english_g2.pth"
    ]
    easyocr_exists = all(os.path.exists(os.path.join(easyocr_dir, f)) for f in easyocr_files)
    
    # Check PaddleOCR models (common model files)
    paddleocr_files = [
        "en_PP-OCRv3_det_infer",
        "en_PP-OCRv3_rec_infer"
    ]
    paddleocr_exists = any(
        os.path.exists(os.path.join(paddleocr_dir, f)) for f in paddleocr_files
    )
    
    return easyocr_exists, paddleocr_exists

def download_models():
    """Download OCR models for offline use if they don't exist"""
    models_dir = "ocr_models"
    easyocr_dir = os.path.join(models_dir, "easyocr")
    paddleocr_dir = os.path.join(models_dir, "paddleocr")
    
    # Create directories if they don't exist
    os.makedirs(easyocr_dir, exist_ok=True)
    os.makedirs(paddleocr_dir, exist_ok=True)
    
    # Check existing models
    easyocr_exists, paddleocr_exists = check_models_exist(easyocr_dir, paddleocr_dir)
    
    if easyocr_exists:
        logging.info("EasyOCR models already exist, skipping download")
    else:
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
    
    if paddleocr_exists:
        logging.info("PaddleOCR models already exist, skipping download")
    else:
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