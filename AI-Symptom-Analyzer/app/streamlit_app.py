import streamlit as st
import pickle
import pandas as pd
import sys
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import pytesseract
import PyPDF2
from docx import Document


# allow streamlit to access src folder
sys.path.append("../src")

from disease_model import DiseasePredictionSystem

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# ----------------------------
# LOAD MODEL FILES
# ----------------------------

@st.cache_resource
def load_system():

    model = pickle.load(open("../model/disease_model.pkl", "rb"))
    le = pickle.load(open("../model/label_encoder.pkl", "rb"))
    symptom_list = pickle.load(open("../model/symptom_list.pkl", "rb"))
    symptom_index = pickle.load(open("../model/symptom_index.pkl", "rb"))
    severity = pickle.load(open("../model/severity.pkl", "rb"))

    predictor = DiseasePredictionSystem(
        model=model,
        label_encoder=le,
        symptom_list=symptom_list,
        symptom_index=symptom_index,
        severity_dict=severity
    )

    return predictor

predictor = load_system()

# ----------------------------
# TEXT EXTRACTION FUNCTIONS
# ----------------------------

def extract_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# ----------------------------
# UI
# ----------------------------

st.title("AI Disease Prediction System")

st.write("Provide symptoms using text, PDF, image, or document.")

mode = st.radio(
    "Select Input Type",
    ["Text", "PDF", "Image", "DOCX"]
)

user_text = ""

# TEXT
if mode == "Text":
    user_text = st.text_area("Enter symptoms separated by space or comma")

# PDF
elif mode == "PDF":
    file = st.file_uploader("Upload medical report PDF", type=["pdf"])
    if file:
        user_text = extract_from_pdf(file)
        st.success("PDF text extracted")
        st.text(user_text[:1000])

# IMAGE
elif mode == "Image":
    file = st.file_uploader("Upload prescription or report image", type=["png","jpg","jpeg"])
    if file:
        st.image(file, width=300)
        user_text = extract_from_image(file)
        st.success("Image text extracted")
        st.text(user_text[:1000])

# DOCX
elif mode == "DOCX":
    file = st.file_uploader("Upload Word document", type=["docx"])
    if file:
        user_text = extract_from_docx(file)
        st.success("Document text extracted")
        st.text(user_text[:1000])

# ----------------------------
# PREDICTION
# ----------------------------

if st.button("Predict Disease"):

    if not user_text.strip():
        st.error("Please provide symptoms or upload a file.")
    else:
        result = predictor.predict(user_text)

        if "error" in result:
            st.error(result["error"])

        else:
            st.subheader("Detected Symptoms")
            st.write(result["symptoms"])

            st.subheader("Predicted Diseases")

            for pred in result["predictions"]:
                st.markdown("---")
                st.write("Disease:", pred["disease"])
                st.write("Confidence:", round(pred["confidence"]*100, 2), "%")
