import pandas as pd
import streamlit as st
from PIL import Image
from model import load_model_and_predict


def process_main_page():
    st.title('Diabetes Prediction App')
    st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage. This is a sample application and cannot be used as a substitute for real medical advice.')
    image = Image.open('data/diabetes.jpg')
    st.image(image)
    personal_info = process_side_bar_inputs()
    if st.button("Predict"):
        prediction, prediction_probas = load_model_and_predict(personal_info)
        write_prediction(prediction, prediction_probas)


def write_prediction(prediction, prediction_probas):
    st.write("## Prediction")
    st.write(prediction)

    st.write("## Prediction Probability")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('User Input Features')
    user_input_df = sidebar_input_features()
    st.write("## Your Data")
    st.write(user_input_df)

    return user_input_df

def sidebar_input_features():
    age = st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
    pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, 0, 1)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 25, 1)
    skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 20, 1)
    bloodpressure = st.sidebar.slider('Blood Pressure', 0, 122, 69, 1)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79, 1)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 31.4, 0.1)
    dpf = st.sidebar.slider("Diabetics Pedigree Function", 0.000, 2.420, 0.471, 0.001)
    
    user_input_data = {
        "Age": [age],
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "SkinThickness": [skinthickness],
        "BloodPressure": [bloodpressure],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf]
    }
    
    user_input_df = pd.DataFrame(user_input_data)
    
    return user_input_df
    
    df = pd.DataFrame(data, index=[0])
    
    return df


if __name__ == "__main__":
    process_main_page()