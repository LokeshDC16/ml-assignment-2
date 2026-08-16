import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing Website Classification",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #0f172a;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .app-header {
        padding: 24px 28px;
        border-radius: 14px;
        background: #172554;
        border: 1px solid #1e3a8a;
        margin-bottom: 24px;
    }

    .app-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .app-subtitle {
        font-size: 16px;
        color: #cbd5e1;
    }

    /* Section headings */
    .section-title {
        font-size: 23px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Information cards */
    .info-card {
        padding: 18px;
        border-radius: 12px;
        background: #111827;
        border: 1px solid #334155;
        min-height: 100px;
    }

    .card-label {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .card-value {
        font-size: 25px;
        font-weight: 700;
    }

    /* Model verdict */
    .verdict {
        padding: 18px 20px;
        border-radius: 12px;
        background: #132e25;
        border: 1px solid #166534;
        margin-top: 18px;
        margin-bottom: 20px;
    }

    .verdict-title {
        font-size: 18px;
        font-weight: 650;
    }

    .verdict-text {
        color: #cbd5e1;
        margin-top: 5px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🛡️ Phishing Website Classification</div>
        <div class="app-subtitle">
            Machine Learning Based Website Classification & Evaluation
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Model Bench")

    selected_model = st.selectbox(
        "Choose a classifier",
        list(MODEL_PATHS.keys())
    )

    st.markdown("---")

    st.markdown("### Classification")

    st.markdown(
        """
        **-1** → Phishing  
        **+1** → Legitimate
        """
    )

    st.markdown("---")

    st.caption(
        "Models trained during the ML Assignment 2 experiment."
    )


# ============================================================
# DATA UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Test Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the test dataset used for model evaluation."
)


if uploaded_file is None:

    st.info(
        "Upload the test CSV file to evaluate the selected machine learning model."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

if "result" not in data.columns:

    st.error(
        "The uploaded dataset must contain the 'result' target column."
    )
    st.stop()


feature_columns = [
    column for column in data.columns
    if column != "result"
]


# ============================================================
# DATASET SUMMARY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">TEST INSTANCES</div>
            <div class="card-value">{len(data):,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">FEATURES</div>
            <div class="card-value">{len(feature_columns)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">MISSING VALUES</div>
            <div class="card-value">{data.isnull().sum().sum()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">SELECTED MODEL</div>
            <div class="card-value">{selected_model}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("View uploaded dataset"):

    st.dataframe(
        data.head(10),
        width="stretch"
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATHS[selected_model])

except Exception as error:

    st.error(
        f"Unable to load {selected_model}: {error}"
    )
    st.stop()


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_test = data[feature_columns]
y_test = data["result"]


# ============================================================
# PREDICTION
# ============================================================

try:

    predictions = model.predict(X_test)

except Exception as error:

    st.error(
        f"Prediction failed. Please check that the uploaded dataset "
        f"contains the correct 30 features.\n\n{error}"
    )
    st.stop()


# ============================================================
# PROBABILITY / AUC
# ============================================================

try:

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        auc_score = roc_auc_score(
            y_test,
            predictions
        )

except Exception:

    auc_score = 0.0


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">2. Performance Snapshot</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with metric2:
    st.metric(
        "AUC",
        f"{auc_score:.4f}"
    )

with metric3:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )


metric4, metric5, metric6 = st.columns(3)

with metric4:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

with metric5:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with metric6:
    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# MODEL VERDICT
# ============================================================

correct_predictions = int(
    np.sum(predictions == y_test)
)

total_predictions = len(y_test)

if selected_model == "Random Forest":

    verdict_text = (
        "Random Forest achieved the strongest overall performance "
        "among the evaluated models in this experiment."
    )

elif selected_model == "Naive Bayes":

    verdict_text = (
        "Naive Bayes shows high precision but substantially lower "
        "recall compared with the other evaluated models."
    )

else:

    verdict_text = (
        f"{selected_model} provides a useful baseline for comparing "
        "classification performance on the phishing dataset."
    )


st.markdown(
    f"""
    <div class="verdict">
        <div class="verdict-title">
            {selected_model} — Model Verdict
        </div>
        <div class="verdict-text">
            {verdict_text}
        </div>
        <div class="verdict-text">
            Correct predictions: {correct_predictions:,} / {total_predictions:,}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">3. Classification Breakdown</div>',
    unsafe_allow_html=True
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[-1, 1]
)

left, right = st.columns([1, 1])

with left:

    st.markdown("### Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Phishing (-1)", "Legitimate (+1)"]
    )

    ax.set_yticklabels(
        ["Phishing (-1)", "Legitimate (+1)"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(selected_model)

    for row in range(2):
        for column in range(2):

            ax.text(
                column,
                row,
                cm[row, column],
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with right:

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=[-1, 1],
        target_names=[
            "Phishing (-1)",
            "Legitimate (+1)"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        width="stretch"
    )


# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">4. Prediction Summary</div>',
    unsafe_allow_html=True
)

phishing_count = int(
    np.sum(predictions == -1)
)

legitimate_count = int(
    np.sum(predictions == 1)
)

summary1, summary2 = st.columns(2)

with summary1:

    st.metric(
        "Predicted Phishing",
        phishing_count
    )

with summary2:

    st.metric(
        "Predicted Legitimate",
        legitimate_count
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Phishing Website Classification | Machine Learning Assignment 2"
)