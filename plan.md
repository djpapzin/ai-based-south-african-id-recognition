# Plan for End-to-End Processing with Segmentation and OCR

## Project Updates

### OCR Engine Selection
- **Removed easyOCR**: After evaluation, easyOCR was removed from the project due to:
  1. Significantly slower processing speed compared to alternatives
  2. Higher resource consumption
  3. Not necessary for our specific use case as Tesseract provides better performance for ID card text extraction

### Current OCR Implementation
- Using Tesseract OCR for text extraction due to:
  1. Faster processing speed
  2. Better integration with Windows environment
  3. Sufficient accuracy for ID card text fields
  4. Lighter resource footprint

## 1. Inference Setup

### 1.1. Load Trained Model
    - Create a function to load the Detectron2 model weights from the local path.
    - Ensure the model is set to inference mode (eval mode).
    - The script must load the model onto the CPU as a GPU is not available.

### 1.2. Load Test Images and Predictions
    - Write code to load the test images and their corresponding object detection bounding boxes generated from the previous script.
    -  Include methods to parse the JSON output and obtain the necessary information.
    -  The JSON output includes the image file name, the bounding box coordinates, and also the class label of the detected objects.

## 2. Segmentation and OCR Processing

### 2.1. Segmentation of Image
   - Loop over each image and its bounding boxes
   - Based on the bounding box information, crop each region.

### 2.2. OCR on Segmented Regions
    -  Implement OCR on the cropped image regions using either `Tesseract v5` or `PaddleOCR` libraries.
    - Ensure that your method for calling `Tesseract` or `PaddleOCR` can be used in a Windows environment, using the CPU.
    -  Implement a way to handle errors in the OCR.
    -  Also implement a way to clean up the output after the OCR step.

## 3. Result Reporting

### 3.1. Save OCR Results
    - Create an output file for each image, to store the output of the OCR step, in JSON format.
    - The JSON output must include the filename, the bounding box, and the text that was detected by the OCR.

### 3.2. Create a Summary Report
    - Create an additional summary report, that gives an overview of how the system performed on the test dataset.
    -  This report should also be in JSON format, and will consist of the filenames, bounding box locations, the labels and the text extracted from the OCR models.

## 4. Local Verification
   * Load the JSON file to view the output
   * Verify the text that is being extracted for each region.
   * Visually verify that the bounding boxes are correctly assigned to each object.

## 5. Next Steps
    - Based on your testing, determine the next steps, and highlight all the challenges, or limitations that were encountered in this process.