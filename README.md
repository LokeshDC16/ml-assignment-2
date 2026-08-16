# Machine Learning Assignment 2

## Phishing Website Classification

This project implements and compares multiple machine learning classification models for detecting phishing and legitimate websites using the Phishing Websites dataset from the UCI Machine Learning Repository.

---

## 1. Problem Statement

The objective of this project is to build a binary classification system that can distinguish between phishing and legitimate websites based on website-related features.

Five machine learning classification algorithms are implemented and evaluated on the same test dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Gaussian Naive Bayes
5. Random Forest

The models are evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 2. Dataset Description

### Dataset

**Phishing Websites**

### Source

UCI Machine Learning Repository

Dataset ID: 327

The dataset contains:

- 11,055 original instances
- 30 input features
- 1 binary target variable
- No missing values

The target variable is `result`.

Target encoding:

- `-1` = Phishing
- `1` = Legitimate

### Dataset preprocessing

The original dataset contained 5,206 exact duplicate rows.

After removing exact duplicate records:

- Final instances: 5,849
- Features: 30

A stratified group-aware train-test split was then used.

The split produced:

- Training instances: 4,679
- Testing instances: 1,170

Identical feature patterns were prevented from appearing in both training and testing sets.

The final test dataset is provided as:

`test_data.csv`

---

## 3. Project Links

### GitHub Repository

[GitHub Repository](https://github.com/LokeshDC16/ml-assignment-2)

### Live Streamlit Application

[Phishing Website Classifier](https://ml-assignment-2-lokesh-2025ac05027-bits.streamlit.app/)

---

## 4. Machine Learning Models

### 4.1 Logistic Regression

Logistic Regression was used as a linear classification baseline.

StandardScaler was applied within a scikit-learn Pipeline so that scaling parameters were learned only from the training data.

### 4.2 Decision Tree Classifier

A Decision Tree classifier was implemented using the 30 input features.

Feature scaling was not required for the Decision Tree.

### 4.3 K-Nearest Neighbor

KNN is a distance-based algorithm, so StandardScaler was applied using a Pipeline before classification.

The model was configured with:

- `n_neighbors = 5`

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was used as the Naive Bayes classifier because the dataset features are numerical.

### 4.5 Random Forest

Random Forest was implemented using 300 decision trees.

Configuration:

- `n_estimators = 300`
- `random_state = 42`

---

## 5. Model Comparison

The following results were obtained using the same test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9248 | 0.9767 | 0.9065 | 0.9417 | 0.9237 | 0.8502 |
| Decision Tree | 0.9316 | 0.9419 | 0.9483 | 0.9081 | 0.9278 | 0.8637 |
| KNN | 0.9308 | 0.9766 | 0.9277 | 0.9293 | 0.9285 | 0.8614 |
| Naive Bayes | 0.6615 | 0.9700 | 0.9942 | 0.3021 | 0.4634 | 0.4240 |
| **Random Forest** | **0.9470** | **0.9930** | **0.9390** | **0.9523** | **0.9456** | **0.8940** |

### Metric convention

For Precision, Recall and F1 in the comparison table, the positive class is `+1`, which corresponds to Legitimate websites.

The dataset labels are:

- `-1` = Phishing
- `+1` = Legitimate

---

## 6. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.48% and an AUC of 0.9767.

The model provided strong baseline performance, with a recall of 94.17% for the positive class. Its performance was lower than Decision Tree, KNN and Random Forest on overall classification metrics.

---

### Decision Tree

Decision Tree achieved 93.16% accuracy.

It obtained the highest precision among the five models at 94.83%. However, its AUC of 0.9419 was lower than the AUC values obtained by Logistic Regression, KNN, Naive Bayes and Random Forest.

The model therefore demonstrated strong classification performance but comparatively weaker ranking performance.

---

### KNN

KNN achieved 93.08% accuracy and an AUC of 0.9766.

Its precision and recall were highly balanced at 92.77% and 92.93%, respectively.

Overall, KNN performed competitively with Decision Tree but remained below Random Forest.

---

### Naive Bayes

Naive Bayes produced the lowest overall classification performance, with an accuracy of 66.15%, F1 score of 0.4634 and MCC of 0.4240.

However, its AUC was relatively high at 0.9700 and its precision for the positive class was 99.42%.

The confusion matrix showed that the model correctly classified almost all phishing instances but classified a large number of legitimate websites as phishing. This resulted in very low recall for the positive class.

This illustrates the difference between probability-ranking performance measured by AUC and threshold-based classification metrics such as accuracy, recall and F1.

---

### Random Forest

Random Forest achieved the strongest overall performance.

It obtained:

- Accuracy: 94.70%
- AUC: 0.9930
- Precision: 93.90%
- Recall: 95.23%
- F1 Score: 0.9456
- MCC: 0.8940

Random Forest achieved the highest Accuracy, AUC, Recall, F1 Score and MCC among the five evaluated models.

Its confusion matrix also showed balanced performance between phishing and legitimate websites.

---

## 7. Overall Winner

### Random Forest

Based on the comparison of the six required evaluation metrics, **Random Forest is the overall best-performing model** for this experiment.

It achieved the highest:

- Accuracy
- AUC
- Recall
- F1 Score
- MCC

Although Decision Tree achieved slightly higher precision, Random Forest provided the strongest overall performance across the evaluation metrics.

---

## 8. Project Structure

```text
ml-assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   │
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
│
├── data/
│   ├── README.md
│   └── phishing_websites.csv
│
├── scripts/
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── train_models.py
│   └── analyze_models.py
│
└── outputs/
    ├── plots/
    │   ├── class_distribution.png
    │   ├── correlation_heatmap.png
    │   └── target_correlation.png
    │
    └── model_comparison.csv
```

---

## 9. How to Run

### 1. Clone the repository

```bash
git clone https://github.com/LokeshDC16/ml-assignment-2.git
cd ml-assignment-2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application allows the user to:

- Upload the test dataset
- Select one of the five trained models
- View model evaluation metrics
- View the confusion matrix
- View the classification report
- View the prediction summary

---

## 10. Reproducing the Experiment

The complete ML pipeline can also be reproduced using the scripts provided in the `scripts/` directory.

### Prepare the dataset

```bash
python scripts/prepare_data.py
```

### Train the models

```bash
python scripts/train_models.py
```

### Analyze model performance

```bash
python scripts/analyze_models.py
```

The trained models are saved in the `model/` directory and the model comparison results are saved in:

`outputs/model_comparison.csv`

The exploratory analysis plots are available in:

`outputs/plots/`