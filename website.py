import tensorflow as tf
import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title='Skin Cancer ML App')

pd.read_csv('./skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

class_names = ['Actinic keratoses',
               'Basal cell carcinoma',
               'Benign keratosis-like lesions ',
               'Dermatofibroma',
               'Melanocytic nevi',
               'Melanoma',
               'Vascular lesions'
              ]

loaded_model = keras.models.load_model('skin_cancer.keras')

st.header('Skin Cancer Machine Learning Application')

uploaded_file = st.file_uploader(label='Upload Image', type=["jpg", "jpeg", "png"])
if (uploaded_file): 
    st.image(uploaded_file)
    img_array = image.img_to_array(uploaded_file)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    pred = loaded_model.predict(img_array, verbose=0)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    pred_label = class_names[pred_class_idx]
    st.header(pred_label)