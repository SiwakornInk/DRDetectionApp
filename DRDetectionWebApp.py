#import library
import cv2
import time
import scipy 

import numpy as np
import streamlit as st
import random as python_random

from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

################################# SETUP #################################

# Set app title and page configuration
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="👁️", layout="wide")

# Load images
NU = Image.open('NULOGO.png')
SCI = Image.open('SciLOGO.webp')
DRex = Image.open('DRLabel.jpg')
DRprepro = Image.open('PreproEX.png')

# Sidebar
st.sidebar.title('Contact me:')
st.sidebar.markdown('Siwakorn Inkong-ngam')
st.sidebar.markdown('Tel: (+66)9 3308 5230')
st.sidebar.markdown('Email: Siwakorni62@nu.ac.th')
st.sidebar.image([NU, SCI], width=150)

# App header and description
st.title('👁️ Diabetic Retinopathy Detection')
st.markdown("""
            This app provides an image classification to detect Diabetic Retinopathy in a fundus image using our trained CNN model
            and a dataset of fundus images from DeepDRiD website.
            """)

st.markdown("""
            Diabetic Retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). 
            It can cause blindness if left undiagnosed and untreated. However, it usually takes several years for Diabetic Retinopathy 
            to reach a stage where it could threaten your sight.
            """)

# Image examples and preprocessing demonstration
st.image(DRex, width=700, caption='Simple picture to explain Diabetic Retinopathy')
st.image(DRprepro, width=700, caption='Example of our preprocessing')

# File upload and prediction
input_image = st.file_uploader("Please upload your fundus image (.jpg, .jpeg, or .png):", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if input_image:
    st.image(input_image, caption='This is your fundus image.', width=256)
    image_str = input_image.read()
    nparr = np.fromstring(image_str, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = circle_crop(img_bgr)
    img_rgb_resized = cv2.resize(img_rgb, (WIDE, WIDE))
    img_rgb_resized = img_rgb_resized.reshape(BATCH_SIZE, WIDE, WIDE, 3)

    datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)
    testdata = datagen.flow(img_rgb_resized)
    
    if st.button('Click to check for Diabetic Retinopathy'):
        with st.spinner('Predicting...'):
            time.sleep(2)
        predict = model.predict(testdata)
        Classes = np.argmax(predict)
        if Classes == 0:
            st.success("Prediction: This image shows no signs of Diabetic Retinopathy. You are healthy.")
        else:
            st.error("Prediction: This image is suspected of Diabetic Retinopathy. We recommend you make an appointment with a specialized doctor for further investigation.")
