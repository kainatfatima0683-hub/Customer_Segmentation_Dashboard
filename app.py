import streamlit as st
import pandas as pd
import pickle
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_final.csv")


# =========================================================
# LOAD SELECTED FEATURES
# =========================================================

@st.cache_data
def load_features():
    with open("models/selected_features.txt", "r") as file:
        features = [line.strip() for line in file if line.strip()]
    return features


# =========================================================
# LOAD SCALER
# =========================================================

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("models/scaler_standard.pkl")
    except Exception:
        with open("models/scaler_standard.pkl", "rb") as file:
            return pickle.load(file)


# =========================================================
# LOAD K-MEANS MODEL
# =========================================================

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/kmeans_model_fixed.pkl")
    except Exception:
        with open("models/kmeans_model_fixed.pkl", "rb") as file:
            return pickle.load(file)


# =========================================================
# LOAD EVERYTHING
# =========================================================

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


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

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
# SIDEBAR FILTERS
# =========================================================

if model_loaded:

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filters")

    # Segment filter
    segment_options = ["All"] + sorted(
        df["Segment"].dropna().unique().tolist()
    )

    selected_segment = st.sidebar.selectbox(
        "Select Segment",
        segment_options
    )

    # Age filter
    if "Age" in df.columns:

        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())

        age_range = st.sidebar.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

    # Income filter
    if "Income" in df.columns:

        min_income = float(df["Income"].min())
        max_income = float(df["Income"].max())

        income_range = st.sidebar.slider(
            "Income Range",
            min_value=min_income,
            max_value=max_income,
            value=(min_income, max_income)
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_segment != "All":
        filtered_df = filtered_df[
            filtered_df["Segment"] == selected_segment
        ]

    if "Age" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Age"] >= age_range[0]) &
            (filtered_df["Age"] <= age_range[1])
        ]

    if "Income" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Income"] >= income_range[0]) &
            (filtered_df["Income"] <= income_range[1])
        ]

else:

    filtered_df = pd.DataFrame()


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.write(
        "ML-based customer segmentation using the final K-Means model."
    )

    if model_loaded:

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            len(df)
        )

        col2.metric(
            "Filtered Customers",
            len(filtered_df)
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

        # -------------------------------------------------
        # SEGMENT DISTRIBUTION
        # -------------------------------------------------

        st.subheader("📈 Customer Segment Distribution")

        segment_counts = (
            filtered_df["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers"
        ]

        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            title="Customers by Segment",
            text="Customers"
        )

        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # DATASET PREVIEW
        # -------------------------------------------------

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            filtered_df.head(10),
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

        st.subheader("Segment Summary")

        segment_summary = (
            filtered_df
            .groupby("Segment")
            .size()
            .reset_index(name="Customers")
        )

        segment_summary["Percentage"] = (
            segment_summary["Customers"]
            / len(filtered_df)
            * 100
        ).round(2)

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        # -------------------------------------------------
        # PIE CHART
        # -------------------------------------------------

        st.subheader("🥧 Segment Distribution")

        fig = px.pie(
            segment_summary,
            names="Segment",
            values="Customers",
            title="Customer Distribution by Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        csv = filtered_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Filtered Customer Data",
            data=csv,
            file_name="filtered_customer_segments.csv",
            mime="text/csv"
        )

    else:

        st.error(error_message)


# =========================================================
# SEGMENT ANALYSIS
# =========================================================

elif page == "Segment Analysis":

    st.title("📊 Segment Analysis")

    if model_loaded:

        # -------------------------------------------------
        # CUSTOMER DISTRIBUTION
        # -------------------------------------------------

        st.subheader("Customer Distribution")

        segment_summary = (
            filtered_df
            .groupby("Segment")
            .size()
            .reset_index(name="Customers")
        )

        segment_summary["Percentage"] = (
            segment_summary["Customers"]
            / len(filtered_df)
            * 100
        ).round(2)

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        # -------------------------------------------------
        # INCOME CHART
        # -------------------------------------------------

        if "Income" in filtered_df.columns:

            st.subheader("💰 Average Income by Segment")

            income_summary = (
                filtered_df
                .groupby("Segment")["Income"]
                .mean()
                .reset_index()
            )

            income_summary["Income"] = income_summary[
                "Income"
            ].round(2)

            fig = px.bar(
                income_summary,
                x="Segment",
                y="Income",
                title="Average Income by Segment",
                text="Income"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # -------------------------------------------------
        # SPENDING CHART
        # -------------------------------------------------

        if "Total_Spend" in filtered_df.columns:

            st.subheader("🛍️ Average Spending by Segment")

            spending_summary = (
                filtered_df
                .groupby("Segment")["Total_Spend"]
                .mean()
                .reset_index()
            )

            spending_summary["Total_Spend"] = spending_summary[
                "Total_Spend"
            ].round(2)

            fig = px.bar(
                spending_summary,
                x="Segment",
                y="Total_Spend",
                title="Average Spending by Segment",
                text="Total_Spend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # -------------------------------------------------
        # AGE CHART
        # -------------------------------------------------

        if "Age" in filtered_df.columns:

            st.subheader("👤 Average Age by Segment")

            age_summary = (
                filtered_df
                .groupby("Segment")["Age"]
                .mean()
                .reset_index()
            )

            age_summary["Age"] = age_summary[
                "Age"
            ].round(2)

            fig = px.bar(
                age_summary,
                x="Segment",
                y="Age",
                title="Average Age by Segment",
                text="Age"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.error(error_message)


# =========================================================
# CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    st.title("🎯 Customer Segment Prediction")

    if model_loaded:

        st.info(
            "Select a customer from the dataset to view "
            "its predicted customer segment."
        )

        # Customer search
        search_text = st.text_input(
            "🔎 Search Customer",
            placeholder="Enter customer ID or row number"
        )

        display_df = filtered_df.copy()

        if search_text:

            # Search ID if available
            if "ID" in display_df.columns:

                display_df = display_df[
                    display_df["ID"]
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ]

            else:

                # Search by dataframe index
                display_df = display_df[
                    display_df.index
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ]

        st.subheader("Customer Results")

        st.dataframe(
            display_df.head(20),
            use_container_width=True
        )

        if len(display_df) > 0:

            selected_customer_index = st.selectbox(
                "Select Customer",
                display_df.index.tolist()
            )

            customer = display_df.loc[
                selected_customer_index
            ]

            st.subheader("Customer Details")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customer Age",
                customer["Age"]
                if "Age" in customer
                else "N/A"
            )

            col2.metric(
                "Income",
                round(customer["Income"], 2)
                if "Income" in customer
                else "N/A"
            )

            col3.metric(
                "Total Spend",
                round(customer["Total_Spend"], 2)
                if "Total_Spend" in customer
                else "N/A"
            )

            st.success(
                f"Predicted Segment: {customer['Segment']}"
            )

        else:

            st.warning("No customer found.")

    else:

        st.error(error_message)


# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================

elif page == "Business Recommendations":

    st.title("💡 Business Recommendations")

    recommendations = {

        "Premium Buyers":
            "Focus on loyalty rewards, premium products, "
            "exclusive offers and VIP programs.",

        "High-Value Customers":
            "Use personalized marketing, cross-selling, "
            "upselling and retention campaigns.",

        "Discount Seekers":
            "Provide discounts, promotions, coupons "
            "and value-based product bundles.",

        "At-Risk Customers":
            "Use re-engagement campaigns, incentives "
            "and personalized offers."
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

        st.subheader("Dashboard Features")

        st.write("✓ Sidebar Navigation")
        st.write("✓ Customer Filters")
        st.write("✓ Interactive Plotly Charts")
        st.write("✓ KPI Cards")
        st.write("✓ Customer Tables")
        st.write("✓ Customer Search")
        st.write("✓ Segment Selection")
        st.write("✓ CSV Download")


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Segmentation Dashboard | Day 2"
)