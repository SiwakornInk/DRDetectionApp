import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

################################# SETUP #################################

model = keras.models.load_model('Prepro94.75')

WIDE = 128
BATCH_SIZE = 1

def load_ben_color(img):
    image = cv2.addWeighted(img, 2, cv2.GaussianBlur(img, (0, 0), WIDE / 5), -2, 100)
    return image

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img):
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img

################################# Main #################################

# Title
st.title('👁️ Diabetic Retinopathy Detection')

# Input image and Predict
input_image = st.file_uploader("Please upload your fundus image (.jpg, .jpeg or .png) : ", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

# Add a textbox for user feedback
user_feedback = st.text_area("Please provide feedback (optional):")

if input_image:
    st.image(input_image, caption='This is your fundus image.', width=256)
    image_str = input_image.read()
    nparr = np.frombuffer(image_str, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = circle_crop(img_bgr)
    img_rgb_resized = cv2.resize(img_rgb, (WIDE, WIDE))
    img_rgb_resized = img_rgb_resized.reshape(BATCH_SIZE, WIDE, WIDE, 3)

    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=load_ben_color)
    testdata = datagen.flow(img_rgb_resized)

    if st.button('Click for checking the Diabetic Retinopathy'):
        with st.spinner('Predicting...'):
            time.sleep(2)
        predict = model.predict(testdata)
        probability_dr = predict[0][1]  # Probability of having DR
        probability_no_dr = predict[0][0]  # Probability of not having DR

        Classes = np.argmax(predict)
        if Classes == 0:
            st.write(f"""
                Prediction for this image: \n
                This image has no DR sign. You are healthy. \n
                Probability of having DR: {probability_dr:.2f} \n
                Probability of not having DR: {probability_no_dr:.2f}
            """)
        else:
            st.write(f"""
                Prediction for this image: \n
                This image is suspected of Diabetic Retinopathy. We recommend you make an appointment with a specialized doctor for more investigation. \n
                Probability of having DR: {probability_dr:.2f} \n
                Probability of not having DR: {probability_no_dr:.2f}
            """)

        # Display user feedback
        if user_feedback:
            st.write("User Feedback:")
            st.write(user_feedback)
