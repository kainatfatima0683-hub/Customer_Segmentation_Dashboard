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

        st.write(
            "Enter customer information below. "
            "The saved StandardScaler and K-Means model "
            "will automatically predict the customer segment."
        )

        st.subheader("📝 Customer Information")

        # -------------------------------------------------
        # CUSTOMER INPUT FORM
        # -------------------------------------------------

        with st.form("customer_prediction_form"):

            col1, col2, col3 = st.columns(3)

            with col1:

                income = st.number_input(
                    "Income",
                    min_value=0.0,
                    value=50000.0,
                    step=1000.0
                )

                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=50,
                    step=1
                )

                total_spend = st.number_input(
                    "Total Spend",
                    min_value=0.0,
                    value=500.0,
                    step=50.0
                )

                recency = st.number_input(
                    "Recency",
                    min_value=0,
                    value=30,
                    step=1
                )

                children = st.number_input(
                    "Children",
                    min_value=0,
                    max_value=10,
                    value=1,
                    step=1
                )

            with col2:

                web_purchases = st.number_input(
                    "Web Purchases",
                    min_value=0,
                    value=5,
                    step=1
                )

                store_purchases = st.number_input(
                    "Store Purchases",
                    min_value=0,
                    value=5,
                    step=1
                )

                catalog_purchases = st.number_input(
                    "Catalog Purchases",
                    min_value=0,
                    value=2,
                    step=1
                )

                web_visits = st.number_input(
                    "Web Visits per Month",
                    min_value=0,
                    value=5,
                    step=1
                )

                deals_purchases = st.number_input(
                    "Deals Purchases",
                    min_value=0,
                    value=2,
                    step=1
                )

            with col3:

                accepted_cmp1 = st.selectbox(
                    "Accepted Campaign 1",
                    [0, 1]
                )

                accepted_cmp2 = st.selectbox(
                    "Accepted Campaign 2",
                    [0, 1]
                )

                accepted_cmp3 = st.selectbox(
                    "Accepted Campaign 3",
                    [0, 1]
                )

                accepted_cmp4 = st.selectbox(
                    "Accepted Campaign 4",
                    [0, 1]
                )

                accepted_cmp5 = st.selectbox(
                    "Accepted Campaign 5",
                    [0, 1]
                )

            predict_button = st.form_submit_button(
                "🔮 Predict Customer Segment"
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        if predict_button:

            # Create input dictionary
            customer_input = {

                "Income": income,
                "Age": age,
                "Total_Spend": total_spend,
                "Recency": recency,
                "Children": children,
                "NumWebPurchases": web_purchases,
                "NumStorePurchases": store_purchases,
                "NumCatalogPurchases": catalog_purchases,
                "NumWebVisitsMonth": web_visits,
                "NumDealsPurchases": deals_purchases,
                "AcceptedCmp1": accepted_cmp1,
                "AcceptedCmp2": accepted_cmp2,
                "AcceptedCmp3": accepted_cmp3,
                "AcceptedCmp4": accepted_cmp4,
                "AcceptedCmp5": accepted_cmp5
            }

            # Convert input to DataFrame
            input_df = pd.DataFrame(
                [customer_input]
            )

            # Make sure feature order is exactly the same
            input_df = input_df[selected_features]

            # Apply saved preprocessing
            input_scaled = scaler.transform(input_df)

            # Generate prediction using trained K-Means model
            predicted_cluster = kmeans.predict(
                input_scaled
            )[0]

            # Convert cluster number to business segment
            predicted_segment = segment_names.get(
                predicted_cluster,
                "Unknown Segment"
            )

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.success(
                "Customer prediction completed successfully!"
            )

            st.subheader("🎯 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Predicted Cluster",
                predicted_cluster
            )

            result_col2.metric(
                "Customer Segment",
                predicted_segment
            )

            # -------------------------------------------------
            # SEGMENT CHARACTERISTICS
            # -------------------------------------------------

            st.subheader("📊 Segment Characteristics")

            segment_data = df[
                df["Segment"] == predicted_segment
            ]

            if len(segment_data) > 0:

                char_col1, char_col2, char_col3 = st.columns(3)

                if "Income" in segment_data.columns:

                    char_col1.metric(
                        "Average Income",
                        f"{segment_data['Income'].mean():,.2f}"
                    )

                if "Age" in segment_data.columns:

                    char_col2.metric(
                        "Average Age",
                        f"{segment_data['Age'].mean():.1f}"
                    )

                if "Total_Spend" in segment_data.columns:

                    char_col3.metric(
                        "Average Spending",
                        f"{segment_data['Total_Spend'].mean():,.2f}"
                    )

                st.write(
                    f"Historical customers belonging to "
                    f"**{predicted_segment}**: "
                    f"**{len(segment_data)}**"
                )

            # -------------------------------------------------
            # MARKETING RECOMMENDATION
            # -------------------------------------------------

            st.subheader("💡 Marketing Recommendation")

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

            recommendation = recommendations.get(
                predicted_segment,
                "No recommendation available."
            )

            st.info(recommendation)

            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            st.subheader("📋 Customer Input Summary")

            st.dataframe(
                input_df,
                use_container_width=True
            )

    else:

        st.error(
            f"Model Loading Error: {error_message}"
        )
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