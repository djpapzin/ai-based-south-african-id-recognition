# Checklist for End-to-End Processing with Segmentation and OCR

## 1. Inference Setup

- [ ] Create function to load the trained Detectron2 model, onto the CPU.
- [ ] Verify the model is in inference mode (eval mode).
- [ ] Implement error handling in case the model fails to load, or if there are invalid weights.
- [ ] Write a function to load test images, and the output from the object detection model.
- [ ] Verify that the JSON data is being parsed correctly.

## 2. Segmentation and OCR Processing

- [ ] Implement a loop to iterate over each image, and each of the detected bounding boxes.
- [ ] Crop each region based on the bounding box information.
- [x] Using Tesseract v5 for optimal performance
- [ ] Implement error handling in case OCR fails.
- [ ] Implement a method for cleaning the output of OCR.
- [ ] Ensure the code for OCR can be run on a Windows environment, using the CPU.

## 3. Result Reporting

- [ ] Generate a JSON output file for each image.
- [ ] Verify that these files contain all the required data (image filename, bounding box info, labels, text output from the OCR).
- [ ] Create an additional summary report, that gives an overview of how the system performed on the test dataset.
- [ ] Ensure this report is in JSON format, and include all the bounding boxes, the labels, and the text that was extracted from the OCR model.

## 4. Local Verification
  - [ ] Verify the outputs in a visual way, to check for all the different components of the project.
  - [ ] Verify that all the bounding boxes are correct.
  - [ ] Verify the labels, and text from the OCR.

## 5. Performance Improvements
  - [x] Remove easyOCR due to slow processing speed
  - [x] Optimize OCR pipeline for faster processing
  - [ ] Implement batch processing for multiple images
  - [ ] Add progress tracking for long-running processes

## 6. Next Steps
  - [ ] Summarize the results and determine the next steps to improve the project.
  - [ ] Note down any limitations, and challenges that you faced.