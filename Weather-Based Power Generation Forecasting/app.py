import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# --- 1. Define the Correct Neural Network Architecture ---
# This architecture matches the one described in the error message.
class Weather(nn.Module):
    def __init__(self, input_dim):
        super(Weather, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        return self.net(x)

# --- 2. Load the Saved Model and Scaler ---
# Load the model's state dictionary
model = Weather(input_dim=4)
model.load_state_dict(torch.load('weather_model.pth'))
model.eval() # Set the model to evaluation mode

# Load the scaler
scaler = joblib.load('scaler.pkl')

# --- 3. Create the Streamlit Web Interface ---
st.title("⚡ Power Plant Energy Prediction")
st.write("Enter the weather conditions to predict the net hourly electrical energy output (PE).")

# Create sliders for user input
at_val = st.slider("Ambient Temperature (AT)", min_value=-2.0, max_value=38.0, value=20.0, step=0.1)
v_val = st.slider("Exhaust Vacuum (V)", min_value=25.0, max_value=82.0, value=60.0, step=0.1)
ap_val = st.slider("Ambient Pressure (AP)", min_value=990.0, max_value=1035.0, value=1015.0, step=0.1)
rh_val = st.slider("Relative Humidity (RH)", min_value=25.0, max_value=100.0, value=80.0, step=0.1)

# --- 4. Make a Prediction ---
if st.button("Predict Energy Output"):
    new_data = np.array([[at_val, v_val, ap_val, rh_val]])
    scaled_new_data = scaler.transform(new_data)
    new_data_tensor = torch.tensor(scaled_new_data, dtype=torch.float32)
    
    with torch.no_grad():
        predicted_pe = model(new_data_tensor)
    
    st.success(f"Predicted Power Output (PE): **{predicted_pe.item():.2f} MW**")