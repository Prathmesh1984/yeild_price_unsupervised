import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Price_Agriculture_commodities_Week.csv")   # your dataset
    return data

data = load_data()

# --------------------------
# Preprocessing
# --------------------------
st.title("🌾 Crop Price Anomaly Detection System (Isolation Forest)")

# Encode categorical columns
label_encoders = {}
for col in ['District', 'Commodity', 'Grade']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Calculate % change in modal price
data['Pct_Change'] = data.groupby(['District', 'Commodity', 'Grade'])['Modal Price'].pct_change() * 100
data['Pct_Change'] = data['Pct_Change'].fillna(0)

# Features
features = ['District', 'Commodity', 'Grade', 'Min Price', 'Max Price', 'Modal Price', 'Pct_Change']
X = data[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest (unsupervised)
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

# --------------------------
# User Input for Prediction
# --------------------------
st.subheader("🔎 Test with Today's Price")

district = st.selectbox("Select District", label_encoders['District'].classes_)
commodity = st.selectbox("Select Commodity", label_encoders['Commodity'].classes_)
grade = st.selectbox("Select Grade", label_encoders['Grade'].classes_)

min_price = st.number_input("Enter Min Price", value=0.0)
max_price = st.number_input("Enter Max Price", value=0.0)
modal_price = st.number_input("Enter Today’s Modal Price", value=0.0)

if st.button("Predict"):
    # Encode selections
    district_enc = label_encoders['District'].transform([district])[0]
    commodity_enc = label_encoders['Commodity'].transform([commodity])[0]
    grade_enc = label_encoders['Grade'].transform([grade])[0]

    # Yesterday’s modal price
    yesterday_data = data[
        (data['District'] == district_enc)
        & (data['Commodity'] == commodity_enc)
        & (data['Grade'] == grade_enc)
    ]

    if not yesterday_data.empty:
        yesterday_modal = yesterday_data['Modal Price'].iloc[-1]
        pct_change = ((modal_price - yesterday_modal) / yesterday_modal) * 100
    else:
        yesterday_modal = modal_price
        pct_change = 0

    # Input row
    input_row = pd.DataFrame([[ 
        district_enc,
        commodity_enc,
        grade_enc,
        min_price,
        max_price,
        modal_price,
        pct_change
    ]], columns=features)

    # Scale input
    scaled_input = scaler.transform(input_row)

    # Predict anomaly (-1 = anomaly, 1 = normal)
    prediction = model.predict(scaled_input)[0]

    # --------------------------
    # Output Message
    # --------------------------
    if prediction == 1:
        st.success("✅ Normal Price Movement\n👉 Farmers can continue regular selling.")
    else:
        if pct_change > 0:
            st.error("🚨 Anomaly Detected: **SPIKE** in price\n💡 Good time to SELL your crop!")
        else:
            st.error("🚨 Anomaly Detected: **CRASH** in price\n⚠️ Avoid selling now, wait for better prices.")
