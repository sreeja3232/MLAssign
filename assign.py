
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import pandas as pd
import re

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        for img in doc.get_page_images(i):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    doc.close()
    return images

def enhance_image(image):
    # Convert PIL Image to OpenCV format and apply enhancements
    open_cv_image = np.array(image.convert('L'))  # Convert to grayscale
    
    # Apply a slight blur to reduce noise
    blurred_image = cv2.GaussianBlur(open_cv_image, (3, 3), 0)
    
    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_image = cv2.dilate(thresh, kernel, iterations=1)

    final_image = Image.fromarray(dilated_image)
    return final_image

def perform_ocr_on_image(image):
    custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR engine and assume a single uniform block of text
    return pytesseract.image_to_string(image, config=custom_config)

def process_extracted_text(text):
    lines = text.split('\n')
    data = []
    
    current_entry = {
        "Voter Full Name": "",
        "Relative's Name": "",
        "Relation Type": "",
        "Age": None,
        "Gender": "",
        "House No": "",
        "EPIC No": ""
    }

    for line in lines:
        line = line.strip()
        
        if "Name" in line:
            if current_entry["Voter Full Name"]:
                data.append(current_entry)
                current_entry = {
                    "Voter Full Name": "",
                    "Relative's Name": "",
                    "Relation Type": "",
                    "Age": None,
                    "Gender": "",
                    "House No": "",
                    "EPIC No": ""
                }
            
            current_entry["Voter Full Name"] = re.sub(r"Name\s*[:!\?]*\s*", "", line).strip()

        elif "Father's Name" in line or "Husband's Name" in line or "Mother's Name" in line:
            relation_type = "FTHR" if "Father's Name" in line else "HSBN" if "Husband's Name" in line else "MTHR"
            relative_name = re.sub(r"(Father's|Husband's|Mother's)\s*Name\s*[:!\?]*\s*", "", line).strip()
            current_entry["Relative's Name"] = relative_name
            current_entry["Relation Type"] = relation_type

        elif "Age" in line and "Gender" in line:
            age_gender_match = re.search(r'Age\s*[:!\?]*\s*(\d+)\s*Gender\s*[:!\?]*\s*(Male|Female)', line)
            if age_gender_match:
                current_entry["Age"] = int(age_gender_match.group(1).strip())
                current_entry["Gender"] = age_gender_match.group(2).strip()

        elif "House Number" in line:
            house_no = re.sub(r"House\s*Number\s*[:!\?]*\s*", "", line).strip()
            current_entry["House No"] = house_no

        elif "EPIC No" in line:
            epic_no = re.sub(r"EPIC\s*No\s*[:!\?]*\s*", "", line).strip()
            current_entry["EPIC No"] = epic_no
    
    if current_entry["Voter Full Name"]:
        data.append(current_entry)
    
    return data

def main(pdf_path):
    images = extract_images_from_pdf(pdf_path)
    all_text = []
    for image in images:
        enhanced_image = enhance_image(image)
        text = perform_ocr_on_image(enhanced_image)
        all_text.append(text)
    
    # Process all extracted text
    all_data = []
    for text in all_text:
        processed_data = process_extracted_text(text)
        all_data.extend(processed_data)
    
    # Create a DataFrame with the correct columns
    df = pd.DataFrame(all_data)
    
    # Add Part S.No column for indexing
    df.insert(0, 'Part S.No', range(1, 1 + len(df)))
    
    # Save to Excel
    df.to_excel(r"C:/Users/sreeja reddy/Downloads/Output File.xlsx", index=False)
    print(df.head())

pdf_path = "C:/Users/sreeja reddy/Downloads/enhance_pdf.pdf"
main(pdf_path)
