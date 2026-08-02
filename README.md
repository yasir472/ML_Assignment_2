# ML_Assignment_2

## Problem statement
This assignment requires building a machine-learning classification workflow that compares multiple models on a dataset, documents the evaluation metrics, and deploys a simple Streamlit dashboard that allows test-data upload and model selection.

## Dataset description
The project uses the Breast Cancer Wisconsin dataset from scikit-learn. It is a binary classification dataset with diagnostic features for tumor samples. The target variable is the malignant/benign label.

## GitHub Repository Link
Repository: https://github.com/yasir472/ML_Assignment_2

## Models used
The dashboard compares these classification models:
- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes
- Random Forest (Ensemble)
- Support Vector Machine

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9371 | 0.9186 | 0.9551 | 0.9444 | 0.9497 | 0.8657 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9893 | 0.9263 | 0.9778 | 0.9514 | 0.8650 |
| Random Forest (Ensemble) | 0.9580 | 0.9952 | 0.9667 | 0.9667 | 0.9667 | 0.9101 |
| Support Vector Machine | 0.9790 | 0.9969 | 0.9888 | 0.9778 | 0.9832 | 0.9553 |

### Observations on model performance
- Logistic Regression offers a strong and reliable baseline for binary classification.
- Decision Tree is easy to interpret, but it may overfit if its depth is not controlled.
- kNN depends heavily on scaling and neighborhood size.
- Naive Bayes is fast and simple, but its assumptions may not fit all datasets perfectly.
- Random Forest tends to provide robust performance and good generalization.
- Support Vector Machine can perform very well, especially when the data are scaled properly.

Overall winner for the chosen dataset: Logistic Regression, which achieved the highest accuracy on the default hold-out split.

## Deployment on Streamlit Community Cloud
1. Open https://streamlit.io/cloud
2. Sign in with a GitHub account
3. Click New App
4. Select the repository
5. Choose the main branch
6. Select app.py
7. Click Deploy

The app includes the following required features:
- CSV upload option for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix and classification report

## Submission checklist
- GitHub repo link works
- Streamlit app link opens correctly
- App loads without errors
- All required features are implemented
- README.md is updated and included in the PDF submission

