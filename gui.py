import streamlit as st
import py7zr
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# -------------------- LOAD MODELS & FILES --------------------
st.info("Loading Models & Features... Please wait...")

# Paths (update if needed)
DBSCAN_PATH = "dbscan_model.pkl"
SCALER_PATH = "feature_scaler (1).pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
FUSED_FEATURES_PATH = "merged_phenotype_embeddings.7z"  # Make it a compressed 7z file
ENCODER_PATH = "fused_encoder (1).h5"

# Load DBSCAN model
dbscan_model = joblib.load(DBSCAN_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load feature columns used in encoder
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

# Load encoder model
encoder_model = load_model(ENCODER_PATH)

# -------------------- EXTRACT CSV FROM 7Z --------------------
extract_path = "data_temp"
os.makedirs(extract_path, exist_ok=True)

with py7zr.SevenZipFile(FUSED_FEATURES_PATH, mode='r') as z:
    z.extractall(path=extract_path)

all_features = pd.read_csv(os.path.join(extract_path, "merged_phenotype_embeddings.csv"))

st.success("Models & Features Loaded ✔️")

# -------------------- DBSCAN PREDICTION FUNCTION --------------------
def dbscan_predict(model, X_new):
    core_samples = model.components_
    core_labels = model.labels_[model.core_sample_indices_]
    predictions = []
    for x in X_new:
        distances = np.linalg.norm(core_samples - x, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        if nearest_dist <= model.eps:
            predictions.append(core_labels[nearest_idx])
        else:
            predictions.append(-1)
    return np.array(predictions)

# -------------------- STREAMLIT UI --------------------
st.title("🌽 Corn Kernel Cluster Detector (DBSCAN)")

uploaded_file = st.file_uploader("Upload a Corn Kernel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    base_name = uploaded_file.name.split('.')[0]
    image_rows = all_features[all_features['full_image_name'].str.startswith(base_name)]

    if image_rows.empty:
        st.error("❌ No precomputed features found for this image!")
    else:
        features_df = image_rows.copy()

        # One-hot encode color_name if exists
        if 'color_name' in features_df.columns:
            features_df = pd.get_dummies(features_df, columns=['color_name'], prefix='colour')

        # Add missing columns
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Ensure correct order
        features_df = features_df[feature_columns]

        # Scale features
        numeric_features = features_df.values
        scaled_features = scaler.transform(numeric_features)

        # Encode features
        latent_features = encoder_model.predict(scaled_features)

        # Predict clusters
        cluster_ids = dbscan_predict(dbscan_model, latent_features)

        # Display results
        st.info("Cluster prediction for each kernel in the image:")
        for idx, cluster_id in enumerate(cluster_ids):
            st.write(f"Kernel {idx}: Cluster {cluster_id}" if cluster_id != -1 else f"Kernel {idx}: Outlier (-1)")

