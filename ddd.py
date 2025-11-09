import streamlit as st
import cv2
import numpy as np
import joblib
from keras.models import Sequential, load_model

imgeee = "skin-cancer-ai/skin-cancer-ai/data/1/ISIC_0000013.jpg"

def extract_glcm_features_from_upload(uploaded_image):
    imggg = cv2.imread(uploaded_image)
    img_rgb = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    image_batch = np.expand_dims(img_normalized, axis=0)
    return image_batch



MODEL_FILENAME = 'skin-cancer-aiv2/SkinCancer_model_V1.keras'
model = load_model(MODEL_FILENAME)

features = extract_glcm_features_from_upload(imgeee)
prediction_probs = model.predict(features)
labels_dict = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
predicted_index = np.argmax(prediction_probs[0])
predicted_label = labels_dict[predicted_index]
print(f"โมเดลทำนายว่าเป็น: {predicted_label}")

##extract_glcm_features_from_upload(imgeee)

