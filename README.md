# AI-Based-Disease-Prediction-System-using-Machine-Learning
Internship Major Project 
## Overview

This project is a Machine Learning–based web application that predicts possible diseases from user symptoms and medical documents.
The system accepts multiple input formats, including typed symptoms, PDF medical reports, prescription images, and Word documents.
It extracts symptoms from the provided input and uses a trained Random Forest classifier to predict the most probable diseases along with confidence scores.

The project is designed as a preliminary health awareness tool and is not a replacement for professional medical diagnosis.

---

## Features

* Predict diseases from user-entered symptoms
* Upload and analyze PDF medical reports
* Upload prescription images (OCR text extraction)
* Upload Word (DOCX) medical documents
* Displays detected symptoms
* Shows top predicted diseases with confidence percentage
* Simple and interactive web interface using Streamlit

---

## Technology Stack

* Python
* Machine Learning (Random Forest – Scikit-learn)
* Streamlit (Web Interface)
* Pandas & NumPy (Data processing)
* PyPDF2 (PDF text extraction)
* Python-docx (DOCX file reading)
* Pytesseract & Pillow (OCR for images)

---

## Project Structure

```
disease-prediction-system/
│
├── app/
│   ├── streamlit_app.py
│   └── text_extractor.py
│
├── src/
│   ├── train_model.py
│   └── disease_model.py
│
├── data/
│   ├── dataset.csv
│   ├── Symptom-severity.csv
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
│
├── requirements.txt
├── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/disease-prediction-system.git
cd disease-prediction-system
```

### 2. Create a virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate     (Windows)
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Install OCR Engine (Important for Image Upload)

Download and install Tesseract OCR:

https://github.com/UB-Mannheim/tesseract/wiki

Install the default path:

```
C:\Program Files\Tesseract-OCR\
```

---

## How to Run the Project

### Step 1: Train the model

```
cd src
python train_model.py
```

This will generate trained model files inside the `model/` directory.

### Step 2: Start the web application

```
cd ../app
streamlit run streamlit_app.py
```

Open browser:

```
http://localhost:8501
```

---

## Example Inputs

**Fungal Infection**

```
itching skin rash nodal skin eruptions
```

**Diabetes**

```
frequent urination, excessive thirst, fatigue, weight loss
```

**Migraine**

```
headache, nausea, sensitivity to light, blurred vision
```

---

## Disclaimer

This system provides only preliminary disease predictions for awareness purposes.
It should **not be used as a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare professional.

---

## Future Improvements

* Larger medical dataset
* Improved natural language processing
* Better handwritten prescription recognition
* Mobile application deployment
* Multi-language support

---

## Author
Parkote Abutalha Rahimoddin
