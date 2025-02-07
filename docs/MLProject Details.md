# Machine Learning

# Introduction

**1\. Project Overview**

The project aims to develop an artificial intelligence system capable of accurately recognizing and extracting metadata from both the traditional South African ID book and the new smart ID card formats. This system will support legitimate identity verification processes while maintaining high security and privacy standards.

**2\. Scope of Work**

**2.1 System Development**

The contractor shall:

* Develop an AI model capable of recognizing and distinguishing between:  
  * Traditional green bar-coded ID book  
  * New smart ID card  
* Implement robust image processing capabilities to handle various lighting conditions, angles, and image qualities  
* Extract name, identity number, and date of birth for each document into JSON format  
* Ability to extract and save the photograph of the participant for further processing in facial recognition

**2.2 Technical Requirements**

* Minimum 99% accuracy in document type classification  
* Minimum 1% error in metadata extraction rate  
* Maximum 10-second processing time per document  
* Ability to handle common image formats (JPEG, PNG, TIFF)  
* Error handling for poor quality images with appropriate user feedback

**2.3 Data Requirements**

The contractor must:

* Keep sample documents provided for training confidential  
* Implement data augmentation techniques to expand training dataset  
* Ensure compliance with POPIA regarding data handling  
* Maintain detailed documentation of data sources and usage

**3\. Deliverables**

**3.1 Development Phase**

* Detailed project plan and timeline  
* Regular progress reports (weekly)  
* Training dataset documentation  
* Model architecture documentation  
* Test plans and procedures

**3.2 Final Deliverables**

* Trained AI model meeting accuracy requirements  
* API documentation for system integration  
* User manual and technical documentation  
* Security audit report  
* Performance benchmark results

**4\. Timeline**

* Project Duration: 1 month  
* Key Milestones:  
  * Week 1: Project setup and data collection  
  * Week 2-3: Model development and initial training  
  * Week 4: Testing and optimization  
  * Week 4: Final testing and documentation

**8\. Intellectual Property**

All intellectual property developed during the project, including:

* Source code  
* Training methodologies  
* Model architecture  
* Documentation shall become the property of the client upon project completion.

# Work Plan

**Phase 1: Baseline Assessment and Data Preparation**

1. **Initial Data Acquisition and Audit:**  
     
   * Acquire the full dataset of South African ID images (both green bar-coded books and smart ID cards).  
   * Conduct an initial data audit to document the characteristics of the dataset, including:  
     * Distribution of ID types (number of samples per class)  
     * Variety in image quality, lighting conditions, and angles.  
     * Availability and quality of existing ground truth information (if any).

   

2. **OCR Baseline Performance Evaluation:** (*Using Tesseract v5 and PaddleOCR*)  
     
   * **Image Preprocessing (Subset):** Preprocess a subset of 100 images to assess and determine suitable preprocessing methods and parameters.  
   * **OCR Testing (Subset):** Apply both Tesseract v5 ([https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)) and EasyOCR ([https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)) to the preprocessed subset of 100 images.  
   * **Field Extraction (Subset):** Extract specific fields (name, identity number, date of birth) from the OCR output.  
   * **Accuracy Assessment (Subset):** Compare the extracted information with ground truth to evaluate the baseline accuracy of OCR. This evaluation will be limited to assessing if the OCR engine can correctly read text on the documents and if the outputs can be correctly processed.  
   * **Document Findings:** Create a report containing all the information collected from this initial analysis.  
   * **OCR Limitations:** Based on the tests, it is clear that Tesseract v5 and EasyOCR alone cannot achieve the required accuracy for the project. The main reason for this was not due to issues with the OCR itself, but with the inability to reliably extract a clean Region Of Interest (ROI), as the quality of the output of the OCR is completely dependant on the quality of the preprocessing.

   

3. **Decision Point: OCR Efficacy**  
     
   * **OCR Performance:** Analyze results and make an evaluation based on metrics obtained from the previous step.  
   * **Decision:** Due to the poor performance of OCR alone, the project will now move into Phase 2, which involves training machine learning models for both document type classification, and region of interest extraction.

   

4. **Document Classification Preliminary Investigation:**  
     
   * Manually examine a subset of images to see if there are features that can allow manual classification of different document types. This step also revealed that the images are too complex for manual classification, and therefore a model is needed.

**Phase 2: Model Training**

1. **Data Annotation:** Annotate the dataset for document classification, text extraction, and object detection using Label Studio or Roboflow:  
     
   * Prepare the data for model training.  
     * Using a combination of SAM and manual labelling, annotating the data by creating bounding boxes on the different data points.  
   * Annotate bounding boxes around main ID region (labelled as `id_document`), for the purpose of defining a Region of Interest (ROI) and for performing document classification.  
   * Annotate ID type for each image, which can be either 'Old ID Book' or 'New ID Card'.  
   * Annotate the four corner points of each ID document, to define the area to extract from the ROI.  
   * Annotate bounding boxes around text fields for OCR and ground-truth matching.  
   * Annotate bounding boxes around faces for face detection (if required).

   

2. **Model Training:** Train models for each of the necessary tasks using Detectron2:  
     
   * Train a document classification model using the annotated data and bounding boxes around the `id_document` region. The purpose of this model is to classify between "Old ID Book" and "New ID Card". Possible models include EfficientNetV2 ([https://github.com/google/automl/tree/master/efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2)), ResNet50 ([https://pytorch.org/vision/stable/models/resnet.html](https://pytorch.org/vision/stable/models/resnet.html)), MobileNetV3 ([https://pytorch.org/vision/stable/models/mobilenetv3.html](https://pytorch.org/vision/stable/models/mobilenetv3.html)).  
   * Train a Detectron2 model to extract the metadata bounding boxes, and the location of the four corner points. This includes text fields and face bounding boxes, using the labelled bounding boxes. The purpose of this step is to extract the coordinates for performing cropping, and also to create labelled bounding boxes for text extraction.  
   * If the OCR is not accurate enough, train an OCR fine-tuned model using techniques identified in the prior model research. Possible model include PaddleOCR ([https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)).  
   * Train a face detection model using the annotated bounding boxes. Possible models include MTCNN ([https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)), RetinaFace ([https://github.com/biubug6/Pytorch\_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)), BlazeFace ([https://github.com/tensorflow/tfjs-models/tree/master/blazeface](https://github.com/tensorflow/tfjs-models/tree/master/blazeface)).

   

3. **Evaluation:** Evaluate performance for each model and iterate based on results.

\*Note: The following sections will be developed further after phase 2:

* Image Preprocessing and Augmentation (will be based on findings from phase 1\)  
* Structured Field Extraction (Will be implemented if OCR is not enough)  
* Face/Photo Extraction (Will be implemented if OCR is not enough)

# Tech Stack Summary

| Component | Technology |
| :---- | :---- |
| Document Classification | [EfficientNetV2](https://github.com/google/automl/tree/master/efficientnetv2), [ResNet50](https://pytorch.org/vision/stable/models/resnet.html), [MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html) (or similar) |
| Data Annotation | [Label Studio](https://github.com/HumanSignal/label-studio), [Label Studio ML  Backend](https://github.com/HumanSignal/label-studio-ml-backend), [Roboflow Annotation](https://roboflow.com/annotate) |
| Object Detection and Keypoint Detection | [Detectron2](https://github.com/facebookresearch/detectron2) |
| OCR / Text Extraction | [Tesseract v5](https://github.com/tesseract-ocr/tesseract) or [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| Face/Photo Extraction | [MTCNN](https://github.com/timesler/facenet-pytorch), [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface), [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface) (or similar) |
| Image Preprocessing | [OpenCV](https://opencv.org/) |
| Data Augmentation | [imgaug](https://imgaug.readthedocs.io/en/latest/) |

# Project Progress and Notes

* **Core Problem:** The core problem identified is not the performance of the OCR engines themselves, but the inability to perform reliable preprocessing, which is a critical step that enables the OCR engine to function correctly. Therefore, the emphasis is now on building a model to perform accurate and robust ROI extraction.  
* **Key Points:** As part of the ROI extraction, the model should also detect the four key corner points on the image, to allow for more accurate cropping, and image normalization.  
* **Data Requirements:** The model should be trained on both new and old ID cards, and it should also be able to perform a document type classification task, which can differentiate between the two.  
* **Initial OCR Testing (Phase 1):**  
  * Tesseract, EasyOCR, and PaddleOCR were tested for the task of ID Number extraction.  
  * Tesseract was fast but completely inaccurate.  
  * EasyOCR was slow, with low accuracy.  
  * PaddleOCR was moderately fast, with somewhat better, but not accurate enough results.  
  * Due to the poor performance of the tested OCR models on raw images, it is clear that a machine learning based system will be needed to achieve the required performance and accuracy.  
* **Shift to Model Training (Phase 2):** The project is now moving to training models, as a machine learning approach is needed to perform region of interest detection and document type classification.  
* **Data Acquisition**: A limited number of ID images are available. More data is needed, particularly with older ID documents.  
* **Next Steps:**  
  * Start annotating images using Label Studio or Roboflow.  
  * Begin testing different preprocessing and data augmentation methods.  
  * The immediate next step is to start writing the script for the Detectron2 model, and to start training it on the small dataset that I have annotated.
