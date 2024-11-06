
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("Student Data1.csv")

# Replace any 'testing' value in the 'workshops' column with 'Testing'
df['workshops'] = df['workshops'].replace(['testing'], 'Testing')

# Load your trained machine learning model
with open('DecisionTreeModel1.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('RandomForestClassifier1.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('SupportVectorMachine1.pkl', 'rb') as f:
    svm_model = pickle.load(f)

def main():
    st.title("STUDENT FUTURE PREDICTION USING MACHINE LEARNING")

    # Custom CSS styling with animated background
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content .stText > p {
            color: #4b4f56;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #5248ff;
        }
        .animated-background {
            background: linear-gradient(270deg, #f00, #00f, #0f0);
            background-size: 600% 600%;
            animation: gradient-animation 10s ease infinite;
            border-radius: 10px;
            color: white;
        }
        @keyframes gradient-animation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display select lists for each column in the dataset
    selected_values = {}  # Store selected values for later use in prediction
    for column in df.columns[:-1]:  # Exclude the last column as it's the output for prediction
        st.subheader(column)
        values = df[column].unique()
        selected_value = st.selectbox(f"Select a value for {column}", values)
        st.write(f"You selected: {selected_value}")
        st.write("")
        selected_values[column] = selected_value

    if st.button('Predict'):  # Add a button for prediction
        # Prepare input features for prediction
        input_features = []
        for column in df.columns[:-1]:  # Exclude the last column
            input_features.append(np.where(df[column].unique() == selected_values[column])[0][0])
        # Make prediction using the loaded model
        dtree_prediction = decision_tree_model.predict([input_features])[0]
        rf_prediction = random_forest_model.predict([input_features])[0]
        svm_prediction = svm_model.predict([input_features])[0]

        st.write(f"REDICTIONS ARE :")
        st.write(f"{dtree_prediction}")
        st.write(f"{rf_prediction}")
        st.write(f"{svm_prediction}")

if __name__ == "__main__":
    main()
