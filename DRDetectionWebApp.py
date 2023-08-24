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

# Streamlit app
st.title('👁️ Diabetic Retinopathy Detection')
st.sidebar.title('Settings')

# User-friendly instructions
st.sidebar.markdown('Upload an eye fundus image to check for Diabetic Retinopathy.')

# Image preprocessing settings
st.sidebar.subheader('Image Preprocessing Settings')
gaussian_weight = st.sidebar.slider('Gaussian Blur Weight', 0, 10, 2)
crop_tolerance = st.sidebar.slider('Cropping Tolerance', 0, 50, 7)

# Input image and Predict
input_image = st.file_uploader("Upload Image (.jpg, .jpeg or .png)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if input_image:
    # Preprocess the uploaded image
    st.image(input_image, caption='Uploaded Fundus Image', width=256)
    image_str = input_image.read()
    nparr = np.frombuffer(image_str, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = circle_crop(img_bgr)
    img_rgb_resized = cv2.resize(img_rgb, (WIDE, WIDE))
    img_rgb_resized = img_rgb_resized.reshape(BATCH_SIZE, WIDE, WIDE, 3)

    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=load_ben_color)
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

# Add a section to explain image preprocessing settings
st.sidebar.subheader('Image Preprocessing Explanation')
st.sidebar.markdown("""
- **Gaussian Blur Weight:** Adjust this value to control the amount of Gaussian blur applied to the image.
- **Cropping Tolerance:** Tune this value to determine how much of the image is cropped during preprocessing.
""")

# Add a section for feedback and credits
st.sidebar.subheader('Feedback & Credits')
st.sidebar.markdown("""
- If you encounter issues or have suggestions for improvement, please provide feedback.
- App developed by [Your Name].
""")

# Deployment and additional improvements
# ...

