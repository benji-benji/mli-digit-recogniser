import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import torch
import psycopg2
from datetime import datetime
import torchvision.transforms as transforms
from src.modeling.predict import predict_single_image
from streamlit_drawable_canvas import st_canvas
from src.modeling.train import get_resnet18_mnist

conn = psycopg2.connect(
    dbname="digitdb",
    user="digituser",
    password="digitpass",
    host="db"
)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_digit INTEGER,
    actual_digit INTEGER
);
""")
conn.commit()
cur.close()
conn.close()

st.sidebar.title("About")
st.sidebar.markdown("This is an digit recogniser to classify handwritten digits. It uses a ResNet18 model trained on the MNIST dataset and can predict handwritten digits from 0 to 9. The model accuracy is 99.19% on MNIST testing data. ")



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
pred =()
confidence=()

st.title("Benji's MNIST Digit Recogniser ✍️ 🤖")

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

def log_prediction(pred, actual, confidence):
    try:
        conn = psycopg2.connect(
            dbname="digitdb",
            user="digituser",
            password="digitpass",
            host="db"
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prediction_logs (predicted_digit, actual_digit, confidence) VALUES (%s, %s, %s)",
            (pred, actual, confidence)
        )
        conn.commit()
        cur.close()
        conn.close()
    
    except Exception as e:
        st.error(f"Failed to log prediction: {e}")

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert("L")
    
    
    label = st.number_input("Enter the correct digit label (0-9):", min_value=0, max_value=9, step=1)
    
    if st.button("Predict"):
    
        pred, confidence = predict_single_image(img,model,device)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Your drawing", width=200)

        with col2:
            st.success(f"""
            ### ✅ Prediction Results:
            - **Model prediction:** {pred}
            - **Confidence:** {confidence:.2%}
            - **Your label:** {label}
            """)
        
        log_prediction(pred, label, confidence)

def fetch_logs(limit=10):
    conn = psycopg2.connect(
        dbname="digitdb",
        user="digituser",
        password="digitpass",
        host="db",
        port="5432"
    )
    query = f"SELECT timestamp, predicted_digit, actual_digit, confidence FROM prediction_logs ORDER BY timestamp DESC LIMIT {limit};"
   
    df = pd.read_sql(query, conn)
    conn.close()
    return df



st.subheader("Last 10 predictions")
logs_df = fetch_logs(limit=10)
logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
st.dataframe(logs_df)
