!pip install --upgrade streamlit
streamlit --version
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Cache data-related functions
@st.cache_data
def load_data(r"D:\CodeBasics (ML)\Healthcare Premium Prediction_Project_1\Updated Proejct\premiums_with_life_style", type = ["xlsx"]):
    """
    Load the uploaded Excel file into a pandas DataFrame.
    """
    return pd.read_excel(uploaded_file)

# Cache resource-related functions
@st.cache_resource
def load_model(file_path):
    """
    Load the pre-trained model from a pickle file.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Streamlit app title
st.title("Insurance Premium Prediction App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
if uploaded_file:
    try:
        # Load and display the uploaded dataset
        data = load_data(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        # Display dataset statistics
        if st.checkbox("Show Dataset Statistics"):
            st.write(data.describe())

        # Sidebar inputs for prediction
        st.sidebar.header("Custom Inputs for Prediction")
        age = st.sidebar.slider("Age", 18, 100, 30)
        income_lakhs = st.sidebar.slider("Income (in Lakhs)", 0.5, 50.0, 5.0)
        lifestyle_risk_score = st.sidebar.slider("Lifestyle Risk Score", 0, 100, 50)

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'income_lakhs': [income_lakhs],
            'lifestyle_risk_score': [lifestyle_risk_score]
        })

        # Load the model and make predictions
        model_path = "best_xgboost_model.pkl"  # Ensure this file exists in your working directory
        model = load_model(model_path)

        if st.button("Predict Premium"):
            prediction = model.predict(input_data)
            st.write(f"Predicted Annual Premium: ₹{prediction[0]:,.2f}")

        # Display feature importance
        if st.checkbox("Show Feature Importance"):
            feature_importance = model.feature_importances_
            features = input_data.columns
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": feature_importance
            }).sort_values(by="Importance", ascending=False)

            st.write("Feature Importance:")
            st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a dataset to get started.")
