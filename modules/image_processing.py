import cv2
import numpy as np

def preprocess_receipt_image(image):
    """
    Applies noise reduction, grayscale conversion, and thresholding to improve OCR accuracy.
    Expects an OpenCV image (numpy array).
    """
    if image is None:
        raise Exception("Image not found")

    # Universal Processor: Keeps it simple to avoid destroying data
    
    # 1. Convert to grayscale (Standard for OCR)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Resize if too small (Upscaling helps huge with small text)
    height, width = gray.shape
    if height < 1000:
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
    # 3. No Thresholding! EasyOCR handles grayscale internally better than we can threshold it manually in generic cases.
    # We return the gray image directly.
    return gray
