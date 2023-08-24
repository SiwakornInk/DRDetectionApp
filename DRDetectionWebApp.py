import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# ... (rest of your code)

#Title
st.title('👁️ Diabetic Retinopathy Detection')

# ... (rest of your code)

#Input image and Predict
input_image = st.file_uploader("Please upload your fundus image (.jpg, .jpeg or .png) : ", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

# Add a textbox for user feedback
user_feedback = st.text_area("Please provide feedback (optional):")

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
                This image has no DR sign. You are healthy. 
                Probability of having DR: {probability_dr:.2f}
                Probability of not having DR: {probability_no_dr:.2f}
            """)
        else:
            st.write(f"""
                Prediction for this image: \n
                This image is suspected of Diabetic Retinopathy. We recommend you make an appointment with a specialized doctor for more investigation. 
                Probability of having DR: {probability_dr:.2f}
                Probability of not having DR: {probability_no_dr:.2f}
            """)

        # Display user feedback
        if user_feedback:
            st.write("User Feedback:")
            st.write(user_feedback)
