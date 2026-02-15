import easyocr
import numpy as np
import streamlit as st

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'], gpu=False)

def extract_text(processed_image):
    """
    Uses EasyOCR to extract raw text from the processed image.
    """
    try:
        reader = get_reader()
        # EasyOCR expects a file path or numpy array
        # detail=0 returns just the text list
        result = reader.readtext(processed_image, detail=0)
        
        # Join the text into a single string with newlines to preserve structure
        return "\n".join(result)
    except Exception as e:
        return f"Error in OCR: {str(e)}"
