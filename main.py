import re
import json
import cv2
import numpy as np

# import pytesseract # Uncomment for real OCR

class MedicalOCRPipeline:
    def __init__(self):
        # Regex patterns for common medical PII
        self.pii_patterns = {
            "UHID": r"UHID\s*(?:No|:)?\s*[:\-\.]?\s*(\d{10,14})",
            
            "IPD": r"IPD\s*(?:No|:)?\s*[:\-\.]?\s*(\d+)",
            
            "Date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            
            "Age_Sex": r"Age\s*[:\-\.]?\s*(\d+\s*[Yy]?)\s*(?:[/,]|\s+)\s*Sex\s*[:\-\.]?\s*([MF])",
            
            "Patient_Name": r"Patient\s*Name\s*[:\-\.]?\s*([A-Za-z\s]+)(?=\s+Age|\s+Sex|\n)"
        }

    def preprocess_image(self, image_path):
        """
        Step 1: Image Preprocessing
        Converts to grayscale and applies thresholding to clean noise.
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load {image_path}")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        denoised = cv2.fastNlMeansDenoising(gray)
        
        
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def ocr_engine(self, processed_image, mock_text=None):
        """
        Step 2: Optical Character Recognition
        """
        if mock_text:
            return mock_text
            
       
        return ""

    def extract_pii(self, text):
        """
        Step 3: PII Extraction using Regex
        """
        extracted_data = {}
        
        for key, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Clean and store unique matches
                cleaned_matches = list(set([m.strip() if isinstance(m, str) else m for m in matches]))
                extracted_data[key] = cleaned_matches
                
        
        if "Patient_Name" in extracted_data:
            # Take the first match and clean extra whitespace
            extracted_data["Patient_Name"] = extracted_data["Patient_Name"][0].strip()

        return extracted_data

    def run(self, image_path, mock_text_input=None):
        # 1. Preprocess
        processed_img = self.preprocess_image(image_path)
        
        # 2. OCR
        raw_text = self.ocr_engine(processed_img, mock_text=mock_text_input)
        
        # 3. Extract PII
        pii_data = self.extract_pii(raw_text)
        
        return {
            "file": image_path,
            "pii_detected": pii_data,
            "raw_text_snippet": raw_text.strip()[:100] + "..." # First 100 chars
        }


# Simulating the text that OCR would produce for 'page_35.jpg'
page_35_ocr_output = """
INSTITUTE OF MEDICAL SCIENCES & SUM HOSPITAL
Patient Name: Santosh Pradhan   Age: 36Y   Sex: M
IPD No: 2236927833    UHID No: 202504110195   Bed No: 10
DATE: 16/04/25
Diagnosis: Mental and Behavioral disorder due to use of alcohol.
"""

pipeline = MedicalOCRPipeline()

# Running the pipeline
# Note: We pass 'mock_text_input' here because we don't have Tesseract installed.
# In a real scenario, remove that argument to use the image directly.
result = pipeline.run("page_35.jpg", mock_text_input=page_35_ocr_output)

print(json.dumps(result, indent=4))