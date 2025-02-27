import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy


# App Title
st.title('Mesothelioma Prediction App')

# About Section
st.markdown('''
# Mesothelioma Detector
This app detects if you have Mesothelioma based on Machine Learning!
- App built by Abdul Haq of Team Skillocity.
- Dataset Creators: Abdullah Cetin Tanrikulu from Dicle University, Faculty of Medicine, Department of Chest Diseases, 21100 Diyarbakir, Turkey
- Orhan Er from Bozok University, Faculty of Engineering, Department of Electrical and Electronics Eng., 66200 Yozgat, Turkey
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.
''')
st.write('---')

# Load Data
try:
    df = pd.read_csv('Mesothelioma-data-set.csv')
except FileNotFoundError:
    st.error("Error: Mesothelioma-data-set.csv not found.  Make sure it's in the same directory.")
    st.stop()

# Clean Data (handle missing values)
df = df.dropna() # crucial to remove rows with missing data

# Data Summary
st.sidebar.header('Patient Data Input')
st.subheader('Dataset Overview')
st.write(df.describe())

# Prepare Data for Model
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Encode target variable
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# User Input Form
def user_input_features():
    age = st.sidebar.slider('Age', int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].mean()))
    platelet_count = st.sidebar.slider('Platelet Count', float(X['Platelet_Count'].min()), float(X['Platelet_Count'].max()), float(X['Platelet_Count'].mean()))
    blood_lactic_dehydrogenise = st.sidebar.slider('Blood Lactic Dehydrogenise', float(X['Blood_Lactic_Dehydrogenise'].min()), float(X['Blood_Lactic_Dehydrogenise'].max()), float(X['Blood_Lactic_Dehydrogenise'].mean()))
    alkaline_phosphatise = st.sidebar.slider('Alkaline Phosphatise', float(X['Alkaline_Phosphatise'].min()), float(X['Alkaline_Phosphatise'].max()), float(X['Alkaline_Phosphatise'].mean()))
    total_protein = st.sidebar.slider('Total Protein', float(X['Total_Protein'].min()), float(X['Total_Protein'].max()), float(X['Total_Protein'].mean()))
    albumin = st.sidebar.slider('Albumin', float(X['Albumin'].min()), float(X['Albumin'].max()), float(X['Albumin'].mean()))
    glucose = st.sidebar.slider('Glucose', float(X['Glucose'].min()), float(X['Glucose'].max()), float(X['Glucose'].mean()))
    pleural_lactic_dehydrogenise = st.sidebar.slider('Pleural Lactic Dehydrogenise', float(X['Pleural_Lactic_Dehydrogenise'].min()), float(X['Pleural_Lactic_Dehydrogenise'].max()), float(X['Pleural_Lactic_Dehydrogenise'].mean()))
    pleural_protein = st.sidebar.slider('Pleural Protein', float(X['Pleural_Protein'].min()), float(X['Pleural_Protein'].max()), float(X['Pleural_Protein'].mean()))
    pleural_albumin = st.sidebar.slider('Pleural Albumin', float(X['Pleural_Albumin'].min()), float(X['Pleural_Albumin'].max()), float(X['Pleural_Albumin'].mean()))
    pleural_glucose = st.sidebar.slider('Pleural Glucose', float(X['Pleural_Glucose'].min()), float(X['Pleural_Glucose'].max()), float(X['Pleural_Glucose'].mean()))
    creactive_protein = st.sidebar.slider('C-reactive Protein', float(X['Creactive_Protein'].min()), float(X['Creactive_Protein'].max()), float(X['Creactive_Protein'].mean()))


    data = {
        'Age': age,
        'Platelet_Count': platelet_count,
        'Blood_Lactic_Dehydrogenise': blood_lactic_dehydrogenise,
        'Alkaline_Phosphatise': alkaline_phosphatise,
        'Total_Protein': total_protein,
        'Albumin': albumin,
        'Glucose': glucose,
        'Pleural_Lactic_Dehydrogenise': pleural_lactic_dehydrogenise,
        'Pleural_Protein': pleural_protein,
        'Pleural_Albumin': pleural_albumin,
        'Pleural_Glucose': pleural_glucose,
        'Creactive_Protein': creactive_protein
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
predicted_class = label_encoder.inverse_transform(prediction)[0] # Decode prediction
st.subheader('Prediction:')

if predicted_class == 0:
    st.write('Healthy')
else:
    st.write('Mesothelioma')

# Model Performance
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'{accuracy * 100:.2f}%')

# Visualization (Example: Feature Importance)
st.subheader('Feature Importance:')
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt) # display plot in streamlit

# Footer
st.write('App built by Abdul Haq.')
st.write('Disclaimer: This is for educational purposes only. Consult a doctor for medical advice.')
