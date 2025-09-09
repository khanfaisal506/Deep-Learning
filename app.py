import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# --- 1. Recreate the Model Architecture ---
# The class definition must be available to load the state_dict
class CancerNet(nn.Module):
    def __init__(self, input_dim):
        super(CancerNet, self).__init__()
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

# --- 2. Load the Saved Artifacts ---
# Instantiate the model
model = CancerNet(input_dim=30) # Input dimension is 30 for this dataset

# Load the trained weights
model.load_state_dict(torch.load('cancer_model.pth'))
model.eval() # Set the model to evaluation mode

# Load the scaler and encoder
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

# --- 3. Create the Streamlit User Interface ---
st.title("Breast Cancer Diagnosis Prediction 🩺")
st.write("Enter the patient's measurements below to predict the diagnosis.")

# Create input fields in the sidebar for all 30 features
st.sidebar.header("Input Features")

# A dictionary to hold all the user inputs
input_data = {}

# Feature names (must be in the same order as the training data)
feature_names = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']

# Create sliders for each feature
for feature in feature_names:
    # Use reasonable min/max values based on your data analysis or general knowledge
    input_data[feature] = st.sidebar.slider(
        label=feature, 
        min_value=0.0, 
        max_value=1.0, # Assuming scaled data, or you can set realistic unscaled ranges
        value=0.5,
        step=0.01
    )

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict"):
    # Convert the dictionary of inputs into a Pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the user input using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Convert scaled data to a PyTorch tensor
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output)
        prediction = (probability >= 0.5).int().item()

    # Get the human-readable label
    diagnosis = encoder.inverse_transform([prediction])[0]
    
    st.subheader("Prediction Result")
    if diagnosis == 'M':
        st.error(f"Diagnosis: Malignant (M)")
    else:
        st.success(f"Diagnosis: Benign (B)")
        
    st.write(f"Confidence: {probability.item():.4f}")