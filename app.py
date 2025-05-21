import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from src.modeling.train import get_resnet18_mnist

st.title("MNIST Digit Recogniser")

canvas_result = st_canvas(
    fill_color="black",  # Black background
    stroke_width=15,
    stroke_color="white",  # White digit
    background_color="black",
    height=224,
    width=224,
    drawing_mode="freedraw",
    key="canvas",
)