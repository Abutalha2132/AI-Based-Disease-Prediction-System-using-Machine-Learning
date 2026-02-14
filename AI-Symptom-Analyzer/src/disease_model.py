import numpy as np
import pandas as pd
import re

class DiseasePredictionSystem:

    def __init__(self, model, label_encoder, symptom_list, symptom_index, severity_dict, desc_df=None, prec_df=None):
        self.model = model
        self.label_encoder = label_encoder
        self.symptom_list = symptom_list
        self.symptom_index = symptom_index
        self.severity_dict = severity_dict
        self.desc_df = desc_df
        self.prec_df = prec_df

    # ---------------------------
    # CLEAN TEXT
    # ---------------------------
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    # ---------------------------
    # EXTRACT SYMPTOMS FROM TEXT
    # ---------------------------
    def extract_symptoms(self, text):
        text = self.clean_text(text)
        found = []

        for symptom in self.symptom_list:
            readable = symptom.replace("_", " ")
            if readable in text or symptom in text:
                found.append(symptom)

        return found

    # ---------------------------
    # CREATE FEATURE VECTOR
    # ---------------------------
    def create_vector(self, symptoms):
        vector = np.zeros(len(self.symptom_list))

        for s in symptoms:
            if s in self.symptom_index:
                idx = self.symptom_index[s]
                weight = self.severity_dict.get(s, 1)
                vector[idx] = weight

        return vector.reshape(1, -1)

    # ---------------------------
    # MAIN PREDICTION FUNCTION
    # (THIS IS WHAT STREAMLIT CALLS)
    # ---------------------------
    def predict(self, text):

        # extract symptoms
        symptoms = self.extract_symptoms(text)

        if len(symptoms) == 0:
            return {"error": "No matching symptoms detected"}

        # create model input
        X = self.create_vector(symptoms)

        # probability prediction
        probabilities = self.model.predict_proba(X)[0]

        # top 3 diseases
        top_indices = probabilities.argsort()[-3:][::-1]

        results = []
        for idx in top_indices:
            disease = self.label_encoder.classes_[idx]
            confidence = float(probabilities[idx])

            results.append({
                "disease": disease,
                "confidence": confidence
            })

        return {
            "symptoms": symptoms,
            "predictions": results
        }
