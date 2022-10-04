#import library
import cv2
import time

import numpy as np
import streamlit as st
import random as python_random

from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

################################# SETUP #################################

model = keras.models.load_model('Prepro94.75')

WIDE = 128
BATCH_SIZE = 1

def load_ben_color(img):
    image = cv2.addWeighted(img,2, cv2.GaussianBlur( img , (0,0) , WIDE/5) ,-2 ,100)
    return image

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img 

################################# Main #################################

#Title
st.title('👁️ Diabetic Retinopathy Detection')

#Images
NU = Image.open('NULOGO.png')
SCI = Image.open('SciLOGO.webp')
IMAGES = [NU,SCI]
DRex = Image.open('DRLabel.jpg')
DRprepro = Image.open('PreproEX.png')

#Sidebar
st.sidebar.title('Contact me :')
st.sidebar.markdown('Siwakorn Inkong-ngam')
st.sidebar.markdown('Tel : (+66)9 3308 5230')
st.sidebar.markdown('Email : Siwakorni62@nu.ac.th')
st.sidebar.image(IMAGES,width=150)

#Markdown
st.markdown("""
            This app provides an image classification to detect the Diabetic Retinopathy in a fundus image using our trained CNN model
            and a dataset of fundus images from DeepDRiD website. \n 
            * **Python libraries :** streamlit, cv2, pillow, numpy, pandas, keras, os
            * **Data source :** DeepDriD website, https://isbi.deepdr.org/data.html
            """)

st.markdown("""
            Diabetic Retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). 
            It can cause blindness if left undiagnosed and untreated. However, it usually takes several years for Diabetic Retinopathy 
            to reach a stage where it could threaten your sight.
            """)

st.image(DRex,width=700,caption='Simple picture to explain Diabetic Retinopathy')

st.markdown("""
            Before checking Diabetic Retinopathy from a fundus image, we first use our preprocessing method to 
            detect some signs of them easier. This is a demonstration from our preprocessing method : 
            """)

st.image(DRprepro,width=700,caption='Example of our preprocessing')

st.markdown("""
            Then we use trained CNN model to detect some signs of Diabetic Retinopathy and show you the result.
            """) 

#Input image and Predict
input_image = st.file_uploader("Please upload your fundus image (.jpg, .jpeg or .png) : ", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if input_image :
    st.image(input_image, caption = 'This is your fundus image.', width=256)
    image_str = input_image.read()
    nparr = np.fromstring(image_str, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = circle_crop(img_bgr)
    img_rgb_resized = cv2.resize(img_rgb,(WIDE,WIDE))
    img_rgb_resized  = img_rgb_resized.reshape(BATCH_SIZE,WIDE,WIDE,3)

    datagen = ImageDataGenerator(rescale= 1./255, preprocessing_function=load_ben_color)
    testdata = datagen.flow(img_rgb_resized)
    
    print(type(testdata))
   #if st.button('Click for checking the Diabetic Retinopathy'):
   #    with st.spinner('Predicting...'):
   #        time.sleep(2)
   #    predict = model.predict(testdata)
   #    Classes = np.argmax(predict)
   #    if Classes == 0 : 
   #        st.write("""
   #                    Prediction for this image : \n
   #                    This image has no DR sign. You are Healthy!! 
   #                """)
   #    else :
   #        st.write("""
   #                    Prediction for this image : \n
   #                    This image has Diabetic Retinopathy. You should see the doctor for treatment ASAP.
   #                """) 