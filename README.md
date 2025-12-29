# Automatic Number Plate Recognition (ANPR)

An end-to-end Automatic Number Plate Recognition system that detects vehicle number plates and extracts alphanumeric text using deep learning and OCR.

## Overview
This project uses **YOLOv8** to detect number plates from vehicle images and **EasyOCR** to read the characters from the detected plates.  
The pipeline is designed to work on real-world images and can handle moderately blurred plates.

## Pipeline
- Vehicle image input  
- Number plate detection (YOLOv8)  
- Plate cropping  
- Text recognition (EasyOCR)  

## Dataset
Custom dataset annotated in YOLO format with separate train and validation splits.  
The dataset is not included in this repository due to size constraints.

## Results
- 606 number plates successfully cropped
- OCR supports English letters and digits  
- Sample outputs are available in `RESULTS/sample_outputs`

## Tech Stack
Python · YOLOv8 · OpenCV · EasyOCR · PyTorch
