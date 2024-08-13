import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Load the models
normal_abnormal_model = torch.load('model_abnormal.pth')
acl_tear_model = torch.load('model_acl1.pth')
other_condition_model = torch.load('model_acl2.pth')

# Ensure models are in evaluation mode
normal_abnormal_model.eval()
acl_tear_model.eval()
other_condition_model.eval()

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict using the normal/abnormal model
def predict_normal_abnormal(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        prediction = normal_abnormal_model(processed_image)
    return 'Normal' if prediction.item() < 0.5 else 'Abnormal'

# Function to predict using the ACL tear or other models
def predict_acl_or_other(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        acl_prediction = acl_tear_model(processed_image)
        other_prediction = other_condition_model(processed_image)
    
    if acl_prediction.item() > other_prediction.item():
        return 'ACL Tear'
    else:
        return 'Other Condition'

# Streamlit app
st.title("MRI Image Classification")

uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")

    st.write("Classifying the image...")
    normal_abnormal_result = predict_normal_abnormal(image)

    if normal_abnormal_result == 'Normal':
        st.success("The MRI is Normal.")
    else:
        st.warning("The MRI is Abnormal.")
        acl_or_other_result = predict_acl_or_other(image)
        st.write(f"The model predicts: **{acl_or_other_result}**")
