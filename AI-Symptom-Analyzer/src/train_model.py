import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from disease_model import DiseasePredictionSystem

print("Loading dataset...")

df = pd.read_csv(r"D:\disease-prediction-system\data\dataset.csv")
severity_df = pd.read_csv(r"D:\disease-prediction-system\data\Symptom-severity.csv")
desc_df = pd.read_csv(r"D:\disease-prediction-system\data\symptom_Description.csv")
prec_df = pd.read_csv(r"D:\disease-prediction-system\data\symptom_precaution.csv")

# -------------------------
# PREPROCESS
# -------------------------
symptom_cols = [c for c in df.columns if "Symptom" in c]

for col in symptom_cols:
    df[col] = df[col].fillna('').str.strip().str.replace(' ', '_')

# unique symptoms
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].unique())

all_symptoms.discard('')
all_symptoms = sorted(list(all_symptoms))

symptom_index = {s:i for i,s in enumerate(all_symptoms)}

# severity
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

# feature matrix
X = np.zeros((len(df), len(all_symptoms)))

for i,row in df.iterrows():
    for col in symptom_cols:
        s = row[col]
        if s in symptom_index:
            X[i][symptom_index[s]] = severity_dict.get(s,1)

# labels
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# -------------------------
# TRAIN MODEL
# -------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X,y)

print("Training done")

# -------------------------
# SAVE
# -------------------------
import os
os.makedirs("../model", exist_ok=True)

pickle.dump(model, open("../model/disease_model.pkl","wb"))
pickle.dump(le, open("../model/label_encoder.pkl","wb"))
pickle.dump(all_symptoms, open("../model/symptom_list.pkl","wb"))
pickle.dump(symptom_index, open("../model/symptom_index.pkl","wb"))
pickle.dump(severity_dict, open("../model/severity.pkl","wb"))

print("All model files saved.")
