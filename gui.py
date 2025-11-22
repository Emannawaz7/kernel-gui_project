import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------- LOAD MODELS & FILES --------------------
st.info("Loading Models & Features... Please wait...")

# Paths (update if needed)
DBSCAN_PATH = r"D:\Eman\Uni\Semester 8\GUI dbscan\models\dbscan_model.pkl"
SCALER_PATH = r"D:\Eman\Uni\Semester 8\GUI dbscan\models\feature_scaler (1).pkl"
FEATURE_COLUMNS_PATH = r"D:\Eman\Uni\Semester 8\GUI dbscan\feature_columns.pkl"
FUSED_FEATURES_PATH = r"D:\Eman\Uni\Semester 8\GUI dbscan\merged_phenotype_embeddings (1).csv"
ENCODER_PATH = r"D:\Eman\Uni\Semester 8\GUI dbscan\models\fused_encoder (1).h5"

# Load DBSCAN model
dbscan_model = joblib.load(DBSCAN_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load feature columns used in encoder
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

# Load encoder model
encoder_model = load_model(ENCODER_PATH)

# Load precomputed features
all_features = pd.read_csv(FUSED_FEATURES_PATH)

st.success("Models & Features Loaded ✔️")

# -------------------- DBSCAN PREDICTION FUNCTION --------------------
def dbscan_predict(model, X_new):
    """
    Predict clusters for DBSCAN using nearest core point.
    """
    core_samples = model.components_  # core sample vectors
    core_labels = model.labels_[model.core_sample_indices_]  # core labels

    predictions = []
    for x in X_new:
        distances = np.linalg.norm(core_samples - x, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        if nearest_dist <= model.eps:
            predictions.append(core_labels[nearest_idx])
        else:
            predictions.append(-1)  # outlier
    return np.array(predictions)

# -------------------- STREAMLIT UI --------------------
st.title("🌽 Corn Kernel Cluster Detector (DBSCAN)")

uploaded_file = st.file_uploader("Upload a Corn Kernel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Extract base name (to match with precomputed features)
    base_name = uploaded_file.name.split('.')[0]  # e.g., DSC_0001
    image_rows = all_features[all_features['full_image_name'].str.startswith(base_name)]

    if image_rows.empty:
        st.error("❌ No precomputed features found for this image!")
    else:
        # -------------------- PREPARE FEATURES --------------------
        features_df = image_rows.copy()

        # If 'color_name' exists, encode it
        if 'color_name' in features_df.columns:
            features_df = pd.get_dummies(features_df, columns=['color_name'], prefix='colour')

        # Add missing columns with zeros to match training
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Ensure correct column order
        features_df = features_df[feature_columns]

        # Extract numeric values
        numeric_features = features_df.values

        # Scale features
        scaled_features = scaler.transform(numeric_features)

        # -------------------- PASS THROUGH ENCODER --------------------
        latent_features = encoder_model.predict(scaled_features)

        # -------------------- PREDICT CLUSTERS --------------------
        cluster_ids = dbscan_predict(dbscan_model, latent_features)

        # Display results
        st.info("Cluster prediction for each kernel in the image:")
        for idx, cluster_id in enumerate(cluster_ids):
            st.write(f"Kernel {idx}: Cluster {cluster_id}" if cluster_id != -1 else f"Kernel {idx}: Outlier (-1)")
