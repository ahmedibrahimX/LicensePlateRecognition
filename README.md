# License Plate Recognition

## Introduction

> This is the project of the Image Processing academic elective course `CMPN446` in Cairo University - Faculty of Engineering - Credit Hours System - Communication and Computer program
>
> This project is about applying image processing techniques to localize the license plate in an image and apply OCR to get the license plate number.

***

## Used Technologies

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scipy&logoColor=white">

***

## Live Demo :man_technologist:

- *Click [**<u>here</u>**](https://share.streamlit.io/ahmedibrahimabdellatif/licenseplaterecognition/main/main.py) to see a live demo of the project*

***

## Pipeline

### Preprocessing:

1. Convert image to grayscale
2. Remove noise by applying bilateral filter
3. Contrast enhancement using Contrast Limited Adaptive Histogram Equalization

### Plate Detection:

1. Vertical edge detection using sobel
2. Image Binarization
3. ROI mask to divide the image into regions of interests according to variance
4. Filter regions according to their sizes
5. Harris corner detection and dilation on remaining regions
6. Weighting to remaining regions according to closeness of corners
7. Choosing region with highest weight
8. Getting contours of the best region to detect the bounding rectangle of the plate

### Character Recognition:

1. Adjusting the phase of the plate
2. Character segmentation
3. Binarization and morphological operations to prepare characters for OCR
4. OCR using `pytesseract`

***

## Team Members

1. [Ahmed Ibrahim](https://github.com/AhmedIbrahimAbdellatif)
2. [Abdelrahman Shahda](https://github.com/Abdelrahman-Shahda)
3. [Mahmoud Maghraby](https://github.com/memaghraby)
4. [Ahmed El-Khatib](https://github.com/ahmedelkhatib99)

***

