import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Constants
WIDE = 128
BATCH_SIZE = 1

# Load the trained model
model = keras.models.load_model('Prepro94.75')

# Image preprocessing
def load_ben_color(img, gaussian_weight):
    image = cv2.addWeighted(img, 2, cv2.GaussianBlur(img, (0, 0), WIDE / gaussian_weight), -2, 100)
    return image

def crop_image_from_gray(img, crop_tolerance):
    if img.ndim == 2:
        mask = img > crop_tolerance
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > crop_tolerance
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img, crop_tolerance, gaussian_weight):
    img = crop_image_from_gray(img, crop_tolerance)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img, crop_tolerance)
    img = load_ben_color(img, gaussian_weight)
    return img

# Streamlit app
st.title('👁️ Diabetic Retinopathy Detection')

# User-friendly instructions
st.write('Upload an eye fundus image to check for Diabetic Retinopathy.')

# Image preprocessing settings
gaussian_weight = st.slider('Gaussian Blur Weight', 0, 10, 2)
crop_tolerance = st.slider('Cropping Tolerance', 0, 50, 7)

# Input image and Predict
input_image = st.file_uploader("Upload Image (.jpg, .jpeg or .png)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if input_image:
    # Preprocess the uploaded image based on the settings
    st.image(input_image, caption='Uploaded Fundus Image', width=256)
    image_str = input_image.read()
    nparr = np.frombuffer(image_str, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = circle_crop(img_bgr, crop_tolerance, gaussian_weight)
    img_rgb_resized = cv2.resize(img_rgb, (WIDE, WIDE))
    img_rgb_resized = img_rgb_resized.reshape(BATCH_SIZE, WIDE, WIDE, 3)

    datagen = ImageDataGenerator(rescale=1. / 255)
    testdata = datagen.flow(img_rgb_resized)

    if st.button('Check for Diabetic Retinopathy'):
        with st.spinner('Predicting...'):
            time.sleep(2)
        predict = model.predict(testdata)
        probability_dr = predict[0][1]
        probability_no_dr = predict[0][0]

        Classes = np.argmax(predict)
        if Classes == 0:
            st.success(f"""
                **Prediction Result:** This image has no signs of Diabetic Retinopathy. You are healthy. 
                \nProbability of having DR: {probability_dr:.2f}
                \nProbability of not having DR: {probability_no_dr:.2f}
            """)
        else:
            st.warning(f"""
                **Prediction Result:** This image is suspected to have Diabetic Retinopathy. We recommend consulting a specialist for further evaluation.
                \nProbability of having DR: {probability_dr:.2f}
                \nProbability of not having DR: {probability_no_dr:.2f}
            """)

# Image Preprocessing Explanation
st.subheader('Image Preprocessing Explanation')
st.write("""
- **Gaussian Blur Weight:** Adjust this value to control the amount of Gaussian blur applied to the image.
- **Cropping Tolerance:** Tune this value to determine how much of the image is cropped during preprocessing.
""")

# Feedback and Credits
st.subheader('Feedback & Credits')
st.write("""
- If you encounter issues or have suggestions for improvement, please provide feedback.
- App developed by [Your Name].
""")
