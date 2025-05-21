import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from src.modeling.predict import predict_single_image
from streamlit_drawable_canvas import st_canvas
from src.modeling.train import get_resnet18_mnist

def load_model():
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = get_resnet18_mnist().to(device)
    model.load_state_dict(torch.load("src/models/resnet18_mnist.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

st.title("MNIST Digit Recogniser")

canvas_result = st_canvas(
    fill_color="black",  
    stroke_width=15,
    stroke_color="white", 
    background_color="black",
    height=224,
    width=224,
    drawing_mode="freedraw",
    key="canvas",
)

preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert("L")
    
    st.image(img, caption="Your drawing", width=112)
    label = st.number_input("Enter the correct digit label (0-9):", min_value=0, max_value=9, step=1)
    
    if st.button("Predict"):
    
        pred = predict_single_image(img,model,device)

        st.write(f"Model prediction: **{pred}**")
        st.write(f"Your label: **{label}**")