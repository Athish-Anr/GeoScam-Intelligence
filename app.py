import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GeoScam Intelligence",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #172033;
}

.subtitle {
    font-size: 18px;
    color: #687386;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    color: #687386;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #172033;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-box {
    background: #eef5ff;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #3976e8;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "districtwise-cyber-crimes-2017-onwards.csv"
    )

    return df


data = load_data()


# =========================================================
# CYBERCRIME COLUMNS
# =========================================================

crime_columns = [
    "identity_theft",
    "cheating_by_personation_by_using_computer_resource",
    "data_theft",
    "credit_card_debit_card_fraud",
    "atms_fraud",
    "online_banking_fraud",
    "otp_frauds",
    "other_frauds",
    "cheating",
    "fake_profile",
    "cyber_blackmailing_threatening",
    "cyber_stalking_bullying_of_women_children",
    "fake_news_on_social_media",
    "forgery",
    "defamation_morphing",
    "ransom_ware",
    "cyber_terrorism",
    "other_offences"
]


available_crimes = [
    col for col in crime_columns
    if col in data.columns
]


for col in available_crimes:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    ).fillna(0)


# =========================================================
# AGGREGATE DATA
# =========================================================

group_columns = [
    "state_name",
    "district_name",
    "year"
]


if "state_code" in data.columns:

    group_columns.insert(
        1,
        "state_code"
    )


if "district_code" in data.columns:

    insert_position = 3 if "state_code" in group_columns else 2

    group_columns.insert(
        insert_position,
        "district_code"
    )


data = (
    data
    .groupby(
        group_columns,
        as_index=False
    )[available_crimes]
    .sum()
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

data["total_cybercrime"] = (
    data[available_crimes]
    .sum(axis=1)
)


fraud_columns = [
    "credit_card_debit_card_fraud",
    "atms_fraud",
    "online_banking_fraud",
    "otp_frauds",
    "other_frauds"
]


fraud_columns = [
    col for col in fraud_columns
    if col in data.columns
]


data["total_fraud"] = (
    data[fraud_columns]
    .sum(axis=1)
)


identity_columns = [
    "identity_theft",
    "data_theft",
    "fake_profile"
]


identity_columns = [
    col for col in identity_columns
    if col in data.columns
]


data["identity_related"] = (
    data[identity_columns]
    .sum(axis=1)
)


data["crime_diversity"] = (
    data[available_crimes] > 0
).sum(axis=1)


data = data.sort_values(
    [
        "district_name",
        "year"
    ]
).reset_index(drop=True)


data["previous_year_cases"] = (
    data
    .groupby("district_name")["total_cybercrime"]
    .shift(1)
)


data["growth_rate"] = np.where(

    data["previous_year_cases"] > 0,

    (
        (
            data["total_cybercrime"]
            -
            data["previous_year_cases"]
        )
        /
        data["previous_year_cases"]
    ) * 100,

    0
)


data["growth_rate"] = (
    data["growth_rate"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)


# =========================================================
# YEARLY PERCENTILE
# =========================================================

def yearly_percentile(series):

    if len(series) <= 1:

        return pd.Series(
            np.full(
                len(series),
                50.0
            ),
            index=series.index
        )

    return (
        series
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )


# =========================================================
# RISK SCORE
# =========================================================

data["volume_score"] = (
    data
    .groupby("year")["total_cybercrime"]
    .transform(yearly_percentile)
)


data["fraud_score"] = (
    data
    .groupby("year")["total_fraud"]
    .transform(yearly_percentile)
)


data["identity_score"] = (
    data
    .groupby("year")["identity_related"]
    .transform(yearly_percentile)
)


positive_growth = (
    data["growth_rate"]
    .clip(lower=0)
)


data["growth_score"] = (
    positive_growth
    .groupby(data["year"])
    .transform(yearly_percentile)
)


data["risk_score"] = (

    0.40 * data["volume_score"]

    +

    0.30 * data["fraud_score"]

    +

    0.20 * data["identity_score"]

    +

    0.10 * data["growth_score"]

)


def assign_risk(score):

    if score < 30:

        return "Low"

    elif score < 60:

        return "Medium"

    else:

        return "High"


data["risk_level"] = (
    data["risk_score"]
    .apply(assign_risk)
)


# =========================================================
# FUTURE RISK DATASET
# =========================================================

future_data = data.copy()


future_data["future_risk_level"] = (
    future_data
    .groupby("district_name")["risk_level"]
    .shift(-1)
)


future_data["future_year"] = (
    future_data
    .groupby("district_name")["year"]
    .shift(-1)
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🛡️ GeoScam Intelligence"
)

st.sidebar.markdown(
    "### Filters"
)


states = sorted(
    data["state_name"]
    .dropna()
    .unique()
)


selected_state = st.sidebar.selectbox(
    "Select State",
    ["All States"] + states
)


if selected_state != "All States":

    districts = sorted(
        data[
            data["state_name"]
            ==
            selected_state
        ]["district_name"]
        .dropna()
        .unique()
    )

else:

    districts = sorted(
        data["district_name"]
        .dropna()
        .unique()
    )


selected_district = st.sidebar.selectbox(
    "Select District",
    ["All Districts"] + districts
)


years = sorted(
    data["year"]
    .dropna()
    .unique()
)


selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + list(years)
)


filtered_data = data.copy()


if selected_state != "All States":

    filtered_data = filtered_data[
        filtered_data["state_name"]
        ==
        selected_state
    ]


if selected_district != "All Districts":

    filtered_data = filtered_data[
        filtered_data["district_name"]
        ==
        selected_district
    ]


if selected_year != "All Years":

    filtered_data = filtered_data[
        filtered_data["year"]
        ==
        selected_year
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">'
    '🛡️ GeoScam Intelligence'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'AI-powered cybercrime risk analysis, '
    'hotspot intelligence and future risk prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# KPI CARDS
# =========================================================

total_cases = int(
    filtered_data["total_cybercrime"].sum()
)


total_fraud = int(
    filtered_data["total_fraud"].sum()
)


average_risk = (
    filtered_data["risk_score"].mean()
)


if pd.isna(average_risk):

    average_risk = 0


average_risk = round(
    average_risk,
    1
)


high_risk_count = int(
    (
        filtered_data["risk_level"]
        ==
        "High"
    ).sum()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        Total Cybercrime
        </div>
        <div class="metric-value">
        {total_cases:,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        Fraud Activity
        </div>
        <div class="metric-value">
        {total_fraud:,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        Average Risk Score
        </div>
        <div class="metric-value">
        {average_risk}/100
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        High Risk Records
        </div>
        <div class="metric-value">
        {high_risk_count}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CURRENT RISK
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Current Risk Indicator'
    '</div>',
    unsafe_allow_html=True
)


st.progress(
    int(
        max(
            0,
            min(
                100,
                average_risk
            )
        )
    )
)


if average_risk < 30:

    st.success(
        f"Current relative risk indicator: "
        f"{average_risk:.1f}/100 — Low"
    )

elif average_risk < 60:

    st.warning(
        f"Current relative risk indicator: "
        f"{average_risk:.1f}/100 — Medium"
    )

else:

    st.error(
        f"Current relative risk indicator: "
        f"{average_risk:.1f}/100 — High"
    )


st.info(
    "The risk score is an engineered relative indicator "
    "based on crime volume, fraud activity, identity-related "
    "crime and positive growth. It is not a probability of crime."
)


# =========================================================
# AUTOMATED INSIGHTS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Automated Insights'
    '</div>',
    unsafe_allow_html=True
)


if len(filtered_data) > 0:

    highest_row = filtered_data.loc[
        filtered_data["total_cybercrime"].idxmax()
    ]


    highest_district = (
        highest_row["district_name"]
    )


    highest_cases = int(
        highest_row["total_cybercrime"]
    )


    dominant_crime = (
        filtered_data[available_crimes]
        .sum()
        .idxmax()
    )


    average_growth = (
        filtered_data["growth_rate"]
        .mean()
    )


    if pd.isna(average_growth):

        average_growth = 0


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            f"Highest observed cybercrime volume: "
            f"{highest_district} "
            f"({highest_cases:,} cases)"
        )


    with col2:

        st.info(
            f"Dominant crime category: "
            f"{dominant_crime}"
        )


    with col3:

        st.info(
            f"Average growth signal: "
            f"{average_growth:.1f}%"
        )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Trends",
        "🔥 Hotspots",
        "🔎 Crime Patterns",
        "🔮 Future Risk Prediction",
        "🧠 ML Intelligence"
    ]
)


# =========================================================
# TAB 1 — TRENDS
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        'Cybercrime Trend'
        '</div>',
        unsafe_allow_html=True
    )


    trend_data = (
        filtered_data
        .groupby("year")["total_cybercrime"]
        .sum()
        .reset_index()
    )


    if len(trend_data) > 0:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.plot(
            trend_data["year"],
            trend_data["total_cybercrime"],
            marker="o",
            linewidth=2
        )


        ax.set_title(
            "Cybercrime Cases Over Time"
        )

        ax.set_xlabel("Year")

        ax.set_ylabel(
            "Total Cybercrime Cases"
        )

        ax.grid(
            alpha=0.3
        )


        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No trend data available."
        )


# =========================================================
# TAB 2 — HOTSPOTS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        'District Hotspots'
        '</div>',
        unsafe_allow_html=True
    )


    hotspot = (
        filtered_data
        .groupby("district_name")
        ["total_cybercrime"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    if len(hotspot) > 0:

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )


        hotspot.sort_values().plot(
            kind="barh",
            ax=ax
        )


        ax.set_title(
            "Top 15 Districts by Cybercrime Volume"
        )

        ax.set_xlabel(
            "Cybercrime Cases"
        )

        ax.set_ylabel(
            "District"
        )


        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No hotspot data available."
        )


# =========================================================
# TAB 3 — CRIME PATTERNS
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        'Crime Category Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    category_totals = (
        filtered_data[available_crimes]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    if category_totals.sum() > 0:

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )


        category_totals.sort_values().plot(
            kind="barh",
            ax=ax
        )


        ax.set_title(
            "Top Cybercrime Categories"
        )

        ax.set_xlabel("Cases")

        ax.set_ylabel(
            "Crime Category"
        )


        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No crime-category data available "
            "for the selected filters."
        )


    st.markdown(
        '<div class="section-title">'
        'Fraud Composition'
        '</div>',
        unsafe_allow_html=True
    )


    fraud_totals = (
        filtered_data[fraud_columns]
        .sum()
    )


    # IMPORTANT:
    # Prevents matplotlib's
    # "All wedge sizes are zero" error.

    if (
        len(fraud_totals) > 0
        and fraud_totals.sum() > 0
    ):

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )


        ax.pie(
            fraud_totals,
            labels=fraud_totals.index,
            autopct="%1.1f%%"
        )


        ax.set_title(
            "Fraud Activity Composition"
        )


        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No fraud activity is available "
            "for the selected data."
        )


# =========================================================
# TAB 4 — FUTURE RISK PREDICTION
# =========================================================

with tab4:

    st.markdown(
        '<div class="section-title">'
        '🔮 Future Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        <b>How the Future Risk Model works</b>

        <br><br>

        The model uses a district's current and historical
        cybercrime characteristics to predict its
        <b>risk category in the following year</b>.

        <br><br>

        Example:

        <br>

        <b>2020 district data → predict 2021 risk</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    prediction_features = [

        "total_cybercrime",

        "total_fraud",

        "identity_related",

        "crime_diversity",

        "growth_rate",

        "previous_year_cases"

    ]


    prediction_data = (
        future_data
        .dropna(
            subset=[
                "future_risk_level"
            ]
        )
        .copy()
    )


    prediction_data = (
        prediction_data
        .dropna(
            subset=prediction_features
        )
    )


    st.markdown(
        "### 1. Prediction Dataset"
    )


    st.write(
        f"District-year records available "
        f"for future-risk modelling: "
        f"**{len(prediction_data):,}**"
    )


    # -----------------------------------------------------
    # TIME SPLIT
    # -----------------------------------------------------

    if len(years) >= 3:

        train_years = [
            year
            for year in years
            if year <= max(years) - 2
        ]


        validation_year = (
            max(years) - 1
        )


        test_year = max(years)


        train_data = prediction_data[
            prediction_data["year"]
            .isin(train_years)
        ]


        validation_data = prediction_data[
            prediction_data["year"]
            ==
            validation_year
        ]


        test_data = prediction_data[
            prediction_data["year"]
            ==
            test_year
        ]


        st.markdown(
            "### 2. Time-Based Training Strategy"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Training Years",
                f"{min(train_years)}–"
                f"{max(train_years)}"
            )


        with col2:

            st.metric(
                "Validation Year",
                validation_year
            )


        with col3:

            st.metric(
                "Latest Input Year",
                test_year
            )


        st.caption(
            "The model learns from earlier years "
            "rather than randomly mixing years."
        )


        # -------------------------------------------------
        # TRAIN MODEL
        # -------------------------------------------------

        if (
            len(train_data) > 0
            and len(test_data) > 0
            and
            train_data[
                "future_risk_level"
            ].nunique() >= 2
        ):


            X_train = train_data[
                prediction_features
            ]


            y_train = train_data[
                "future_risk_level"
            ]


            X_test = test_data[
                prediction_features
            ]


            y_test = test_data[
                "future_risk_level"
            ]


            future_model = RandomForestClassifier(

                n_estimators=250,

                random_state=42,

                class_weight="balanced"

            )


            future_model.fit(
                X_train,
                y_train
            )


            test_predictions = (
                future_model.predict(
                    X_test
                )
            )


            test_accuracy = (
                accuracy_score(
                    y_test,
                    test_predictions
                )
            )


            test_f1 = (
                f1_score(
                    y_test,
                    test_predictions,
                    average="macro"
                )
            )


            st.markdown(
                "### 3. Model Performance"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Test Accuracy",
                    f"{test_accuracy * 100:.1f}%"
                )


            with col2:

                st.metric(
                    "Macro F1 Score",
                    f"{test_f1:.2f}"
                )


            with col3:

                st.metric(
                    "Test Records",
                    len(y_test)
                )


            st.caption(
                f"These results evaluate how well "
                f"{test_year} district information "
                f"predicts the next-year risk label "
                f"available in the dataset."
            )


            # -------------------------------------------------
            # CONFUSION MATRIX
            # -------------------------------------------------

            st.markdown(
                "### 4. Future Risk Confusion Matrix"
            )


            labels = [
                "Low",
                "Medium",
                "High"
            ]


            cm = confusion_matrix(
                y_test,
                test_predictions,
                labels=labels
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )


            ax.set_xlabel(
                "Predicted Future Risk"
            )


            ax.set_ylabel(
                "Actual Future Risk"
            )


            ax.set_title(
                "Future Risk Prediction"
            )


            st.pyplot(fig)

            plt.close(fig)


            # -------------------------------------------------
            # FEATURE IMPORTANCE
            # -------------------------------------------------

            st.markdown(
                "### 5. Factors Influencing Future Risk"
            )


            importance = pd.DataFrame({

                "Feature":
                prediction_features,

                "Importance":
                future_model
                .feature_importances_

            }).sort_values(
                "Importance",
                ascending=False
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            ax.barh(
                importance["Feature"],
                importance["Importance"]
            )


            ax.invert_yaxis()


            ax.set_title(
                "Future Risk Feature Importance"
            )


            ax.set_xlabel(
                "Importance"
            )


            st.pyplot(fig)

            plt.close(fig)


            # -------------------------------------------------
            # DISTRICT PREDICTION
            # -------------------------------------------------

            st.markdown(
                "### 6. Predict Future Risk for a District"
            )


            prediction_districts = sorted(
                prediction_data[
                    "district_name"
                ]
                .unique()
            )


            if len(prediction_districts) > 0:


                chosen_district = (
                    st.selectbox(
                        "Choose District",
                        prediction_districts,
                        key="future_prediction_district"
                    )
                )


                district_history = (
                    prediction_data[
                        prediction_data[
                            "district_name"
                        ]
                        ==
                        chosen_district
                    ]
                    .sort_values("year")
                )


                if len(district_history) > 0:


                    latest_row = (
                        district_history
                        .iloc[-1]
                    )


                    latest_features = (
                        pd.DataFrame(
                            [
                                latest_row[
                                    prediction_features
                                ].values
                            ],
                            columns=prediction_features
                        )
                    )


                    predicted_future_risk = (
                        future_model
                        .predict(
                            latest_features
                        )[0]
                    )


                    probabilities = (
                        future_model
                        .predict_proba(
                            latest_features
                        )[0]
                    )


                    latest_year = int(
                        latest_row["year"]
                    )


                    st.markdown(
                        f"""
                        <div class="info-box">

                        <b>District:</b>
                        {chosen_district}

                        <br><br>

                        <b>Latest available year:</b>
                        {latest_year}

                        <br><br>

                        <b>Predicted next-year risk:</b>
                        {predicted_future_risk}

                        <br><br>

                        <b>Prediction year:</b>
                        {latest_year + 1}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        "#### Prediction Probability Distribution"
                    )


                    probability_df = pd.DataFrame({

                        "Risk Level":
                        future_model.classes_,

                        "Model Probability":
                        probabilities

                    })


                    probability_df[
                        "Model Probability"
                    ] = (
                        probability_df[
                            "Model Probability"
                        ] * 100
                    ).round(2)


                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )


                    # -------------------------------------------------
                    # DISTRICT TREND
                    # -------------------------------------------------

                    st.markdown(
                        "#### District Historical Pattern"
                    )


                    fig, ax = plt.subplots(
                        figsize=(10, 5)
                    )


                    ax.plot(
                        district_history["year"],
                        district_history[
                            "total_cybercrime"
                        ],
                        marker="o"
                    )


                    ax.set_title(
                        f"Cybercrime Trend — "
                        f"{chosen_district}"
                    )


                    ax.set_xlabel(
                        "Year"
                    )


                    ax.set_ylabel(
                        "Cybercrime Cases"
                    )


                    ax.grid(
                        alpha=0.3
                    )


                    st.pyplot(fig)

                    plt.close(fig)


        else:

            st.warning(
                "There is not enough class variation "
                "or historical data to train the "
                "future-risk model."
            )

    else:

        st.warning(
            "At least three years of data are required "
            "for time-based future risk modelling."
        )


# =========================================================
# TAB 5 — ML INTELLIGENCE
# =========================================================

with tab5:

    st.markdown(
        '<div class="section-title">'
        '🧠 Current Risk ML Classification'
        '</div>',
        unsafe_allow_html=True
    )


    ml_features = [

        "total_cybercrime",

        "total_fraud",

        "identity_related",

        "crime_diversity",

        "growth_rate"

    ]


    ml_data = (
        data
        .dropna(
            subset=
            ml_features
            +
            ["risk_level"]
        )
        .copy()
    )


    X = ml_data[
        ml_features
    ]


    y = ml_data[
        "risk_level"
    ]


    if (
        len(y.unique()) >= 2
        and
        len(ml_data) >= 10
    ):


        X_train, X_test, y_train, y_test = (
            __import__(
                "sklearn.model_selection",
                fromlist=["train_test_split"]
            )
            .train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )


        rf_model = RandomForestClassifier(

            n_estimators=150,

            random_state=42,

            class_weight="balanced"

        )


        rf_model.fit(
            X_train,
            y_train
        )


        predictions = (
            rf_model.predict(
                X_test
            )
        )


        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )


        f1 = (
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Classification Accuracy",
                f"{accuracy * 100:.1f}%"
            )


        with col2:

            st.metric(
                "Macro F1",
                f"{f1:.2f}"
            )


        st.markdown(
            "### Feature Importance"
        )


        importance = pd.DataFrame({

            "Feature":
            ml_features,

            "Importance":
            rf_model.feature_importances_

        }).sort_values(
            "Importance",
            ascending=False
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )


        ax.invert_yaxis()


        ax.set_title(
            "Current Risk Classification "
            "Feature Importance"
        )


        st.pyplot(fig)

        plt.close(fig)


        st.markdown(
            "### Confusion Matrix"
        )


        labels = [
            "Low",
            "Medium",
            "High"
        ]


        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )


        ax.set_xlabel(
            "Predicted Risk"
        )


        ax.set_ylabel(
            "Actual Risk"
        )


        ax.set_title(
            "Current Risk Classification"
        )


        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "Not enough data/classes for ML classification."
        )


# =========================================================
# PROCESSED DATA
# =========================================================

with st.expander(
    "📊 View Processed Dataset"
):

    st.dataframe(
        filtered_data,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.markdown(
    """
    <center>

    <b>GeoScam Intelligence</b>

    <br>

    Data-driven Cybercrime Risk & Future Risk Analysis

    </center>
    """,
    unsafe_allow_html=True
)