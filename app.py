import streamlit as st
import pandas as pd
import pickle
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_final.csv")


# -----------------------------
# Load Selected Features
# -----------------------------
@st.cache_data
def load_features():
    with open("models/selected_features.txt", "r") as file:
        features = [line.strip() for line in file if line.strip()]
    return features


# -----------------------------
# Load Scaler
# -----------------------------
@st.cache_resource
def load_scaler():
    try:
        return joblib.load("models/scaler_standard.pkl")
    except Exception:
        with open("models/scaler_standard.pkl", "rb") as file:
            return pickle.load(file)


# -----------------------------
# Load K-Means Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/kmeans_model_fixed.pkl")
    except Exception:
        with open("models/kmeans_model_fixed.pkl", "rb") as file:
            return pickle.load(file)


# -----------------------------
# Load Everything
# -----------------------------
try:
    df = load_data()
    selected_features = load_features()
    scaler = load_scaler()
    kmeans = load_model()

    # Generate cluster predictions
    X = df[selected_features]
    X_scaled = scaler.transform(X)
    df["Cluster"] = kmeans.predict(X_scaled)

    # Segment names
    segment_names = {
        0: "Premium Buyers",
        1: "High-Value Customers",
        2: "Discount Seekers",
        3: "At-Risk Customers"
    }

    df["Segment"] = df["Cluster"].map(segment_names)

    model_loaded = True

except Exception as e:
    model_loaded = False
    error_message = str(e)


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Customer Segments",
        "Segment Analysis",
        "Customer Prediction",
        "Business Recommendations",
        "About Model"
    ]
)


# =========================================================
# HOME
# =========================================================
if page == "Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.write(
        "ML-based customer segmentation using the final K-Means model."
    )

    if model_loaded:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            len(df)
        )

        col2.metric(
            "Total Features",
            df.shape[1]
        )

        col3.metric(
            "Selected Features",
            len(selected_features)
        )

        col4.metric(
            "Customer Segments",
            4
        )

        st.success(
            "Dataset, scaler and K-Means model loaded successfully."
        )

        st.subheader("Final Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

    else:

        st.error(
            f"Model Loading Error: {error_message}"
        )


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================
elif page == "Customer Segments":

    st.title("👥 Customer Segments")

    if model_loaded:

        segment_counts = (
            df["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers"
        ]

        st.dataframe(
            segment_counts,
            use_container_width=True
        )

        st.subheader("Segment Distribution")

        st.bar_chart(
            segment_counts.set_index("Segment")
        )

    else:
        st.error(error_message)


# =========================================================
# SEGMENT ANALYSIS
# =========================================================
elif page == "Segment Analysis":

    st.title("📊 Segment Analysis")

    if model_loaded:

        st.subheader("Customer Distribution")

        segment_summary = (
            df.groupby("Segment")
            .size()
            .reset_index(name="Customers")
        )

        segment_summary["Percentage"] = (
            segment_summary["Customers"]
            / len(df) * 100
        ).round(2)

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        if "Income" in df.columns:
            st.subheader("Average Income by Segment")

            income_summary = (
                df.groupby("Segment")["Income"]
                .mean()
                .round(2)
            )

            st.bar_chart(income_summary)

        if "Total_Spend" in df.columns:
            st.subheader("Average Spending by Segment")

            spending_summary = (
                df.groupby("Segment")["Total_Spend"]
                .mean()
                .round(2)
            )

            st.bar_chart(spending_summary)

    else:
        st.error(error_message)


# =========================================================
# CUSTOMER PREDICTION
# =========================================================
elif page == "Customer Prediction":

    st.title("🎯 Customer Segment Prediction")

    if model_loaded:

        st.info(
            "Prediction interface will be enhanced during Day 3 "
            "with customer input fields."
        )

        st.write("Current model:")

        st.write("**Algorithm:** K-Means")
        st.write("**Clusters:** 4")

        st.write("**Selected Features:**")

        for feature in selected_features:
            st.write(f"- {feature}")

    else:
        st.error(error_message)


# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================
elif page == "Business Recommendations":

    st.title("💡 Business Recommendations")

    recommendations = {
        "Premium Buyers":
            "Focus on loyalty rewards, premium products and exclusive offers.",

        "High-Value Customers":
            "Use personalized marketing, cross-selling and retention campaigns.",

        "Discount Seekers":
            "Provide discounts, promotions and value-based product bundles.",

        "At-Risk Customers":
            "Use re-engagement campaigns, incentives and personalized offers."
    }

    for segment, recommendation in recommendations.items():

        st.subheader(segment)
        st.write(recommendation)


# =========================================================
# ABOUT MODEL
# =========================================================
elif page == "About Model":

    st.title("ℹ️ About Model")

    st.write("**Algorithm:** K-Means Clustering")
    st.write("**Number of Clusters:** 4")
    st.write("**Scaler:** StandardScaler")
    st.write("**Model:** kmeans_model_fixed.pkl")
    st.write("**Dataset:** cleaned_final.csv")

    if model_loaded:

        st.success("Model is ready for use.")

        st.subheader("Selected Features")

        for feature in selected_features:
            st.write(f"- {feature}")