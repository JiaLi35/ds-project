import tensorflow as tf
import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title='Skin Cancer ML App', layout='wide')

patient_df = pd.read_csv('./patient_data_cleaned.csv')

class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions ', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']              

# load the saved model
loaded_model = keras.models.load_model('good_model_skin_cancer.keras')

# sidebar 
with st.sidebar: 
    st.sidebar.title('Skin Cancer App')
    page = st.sidebar.radio('Navigation', ['Prediction Model', 'Analytics'])

if page == 'Prediction Model':
    col1, col2, col3 = st.columns([1,4,1])
    
    with col2: 
        st.header('Skin Cancer Prediction Model')
    
        uploaded_file = st.file_uploader(label='Upload Image', type=["jpg", "jpeg", "png"])
        if (uploaded_file): 
            col4, col5, col6 = st.columns([1,4,1])
            with col5: 
                img = Image.open(uploaded_file)
                st.image(img)
                submit_btn = st.button(label='Predict')
            
                if submit_btn: 
                    with st.spinner('Analyzing Image...'):
                        img = img.resize((224,224))
                        img_array = image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array/255.0
                        pred = loaded_model.predict(img_array, verbose=0)[0]
                        pred_class_idx = np.argmax(pred)
                        
                    confidence = pred[pred_class_idx] * 100
                    pred_label = class_names[pred_class_idx]

                    if pred_label == 'Benign keratosis-like lesions' or pred_label == 'Dermatofibroma' or pred_label == 'Melanocytic nevi':
                        st.success(f"Skin Lesion Type: {pred_label}, {confidence:.2f}% confident")
                    else: 
                        st.error(f"Skin Lesion Type: {pred_label}, {confidence:.2f}% confident")

                    # Probability chart
                    df_pred = pd.DataFrame({
                        "Skin Lesion Type": class_names,
                        "Probability (%)": pred * 100
                    })

                    prob_fig = px.bar(
                        df_pred.sort_values("Probability (%)"),
                        x="Probability (%)",
                        y="Skin Lesion Type",
                        orientation="h",
                    )
                    
                    st.divider()
                    
                    st.subheader('Probability of each Skin Cancer Type')
                    st.plotly_chart(prob_fig)

if page == 'Analytics': 
    col1, col2, col3 = st.columns([1,4,1])
    
    with col2: 
        st.header('Exploratory Data Analysis (EDA)')
        st.metric('Total records in dataset after cleaning', len(patient_df), 'Datasize', delta_arrow='off')

        tab1, tab2, tab3 = st.tabs(['Overall Distribution', 'Comparisons', 'Sample Data'])
        
        with tab1: 
            col4, col5 = st.columns(2)
            with col4: 
                age_dist = px.histogram(patient_df, x='age', title='Age Distribution')
                st.plotly_chart(age_dist)
            with col5: 
                sex_dist = px.histogram(patient_df, x='sex', title='Gender Distribution')
                st.plotly_chart(sex_dist)
                
            col6, col7 = st.columns(2)
            with col6: 
                cell_dist = px.histogram(patient_df, x='cell_type', title='Skin Lesion Type Distribution')
                st.plotly_chart(cell_dist)
            with col7: 
                local_dist = px.histogram(patient_df, x='localization', title='Localization Distribution')
                st.plotly_chart(local_dist)
                
            confirm_dist = px.histogram(patient_df, x='dx_type', title='Confirmation Examination Type Distribution')
            st.plotly_chart(confirm_dist)
        
        with tab2: 
            age_by_sex = px.box(patient_df, x='sex', y='age', title='Confirmation Examination Type Distribution')
            st.plotly_chart(age_by_sex)
            local_by_cell = px.histogram(patient_df, x='localization', color='cell_type', title='Confirmation Examination Type Distribution')
            st.plotly_chart(local_by_cell)
            age_by_cell = px.histogram(patient_df, x='age', color='cell_type', title='Confirmation Examination Type Distribution')
            st.plotly_chart(age_by_cell)
            cell_by_gender = px.histogram(patient_df, x='sex', color='cell_type',  title='Confirmation Examination Type Distribution')
            st.plotly_chart(cell_by_gender) 

        with tab3: 
            st.subheader('Raw data')
            st.dataframe(patient_df.sample(50))