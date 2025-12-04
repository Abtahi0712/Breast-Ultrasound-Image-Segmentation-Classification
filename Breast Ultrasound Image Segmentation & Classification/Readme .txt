Breast Ultrasound Image Segmentation and Classification
A Multi-Task U-Net Approach
Overview
This project implements a multi-task deep learning architecture for breast ultrasound analysis, combining lesion segmentation and image-level classification (normal, benign, malignant) into a single end-to-end model.
The model is based on a shared U-Net encoder with two task-specific heads:
* Segmentation decoder for pixel-wise lesion masks

* Classification branch using global pooling and dense layers

This repository includes the full workflow: dataset review, preprocessing, model development, joint training, evaluation, and visualization.
________________


Dataset
We use the Breast Ultrasound Images Dataset, containing:
   * Three classes: Normal, Benign, Malignant

   * Corresponding lesion masks for supervised segmentation

   * Grayscale ultrasound scans

Preprocessing Steps
      * Loaded and validated all image–mask pairs

      * Resized images to a consistent resolution

      * Normalized pixel intensities

      * Applied basic noise-reduction techniques

      * Encoded classification labels

________________


Model Architecture
Multi-Task U-Net
The architecture includes:
         * Shared Encoder: Contracting path of U-Net

         * Segmentation Decoder: Expanding path with skip connections

         * Classification Head:

            * Global Average Pooling at the bottleneck

            * Fully Connected layers

            * Softmax output for three classes

Loss Functions
A weighted loss is used to jointly optimize both tasks:
Total Loss = α * SegmentationLoss + β * ClassificationLoss
Segmentation: Dice loss or BCE
Classification: Categorical cross-entropy
________________


Training Strategy
               * Joint training (shared encoder, two task heads)

               * Adam optimizer with LR scheduling

               * Early stopping based on validation metrics

               * Batch-wise training with augmentation if required

________________


Evaluation Metrics
Segmentation
                  * Dice Coefficient

                  * Mean Intersection over Union (mIoU)

                  * Pixel Accuracy

Classification
                     * Accuracy

                     * Precision

                     * Recall

                     * F1-Score

Both sets of metrics are reported for training and validation splits.
Running the Project
Requirements
Python 3.x  
TensorFlow / Keras  
NumPy  
Matplotlib  
scikit-learn  
opencv-python  


How to Run
                        1. Open the notebook:
Breast Ultrasound Image Segmentation & Classification.ipynb

                        2. Execute cells sequentially: preprocessing → model definition → training → evaluation.

                        3. For testing on a new ultrasound image, preprocess it and call:
model.predict(image)
 Then overlay the segmentation mask.

________________


Project Deliverables Included
                           * Complete Jupyter Notebook

                           * Model architecture diagram 

                           * Segmentation visualizations

                           * Trained model weights