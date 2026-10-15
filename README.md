# 🛡️ Insurance Fraud Detection Using Machine Learning

**Detecting potentially fraudulent insurance claims using machine learning classification algorithms**

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![SMOTE](https://img.shields.io/badge/SMOTE-Class%20Imbalance-green)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 🧭 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Project Objectives](#-project-objectives)
* [Dataset](#-dataset)
* [Technologies Used](#-technologies-used)
* [Project Workflow](#-project-workflow)
* [Machine Learning Models](#-machine-learning-models)
* [Model Performance](#-model-performance)
* [Model Evaluation](#-model-evaluation)
* [Feature Importance](#-feature-importance)
* [Key Insights](#-key-insights)
* [Limitations](#-limitations)
* [Future Improvements](#-future-improvements)
* [Installation and Usage](#-installation-and-usage)
* [Project Structure](#-project-structure)
* [Conclusion](#-conclusion)
* [Author](#-author)

---

## 🔍 Overview

Insurance fraud is a major challenge for insurance companies because fraudulent claims can result in financial losses, increased investigation costs, and inefficient claims processing.

This project develops a machine learning classification pipeline to identify potentially fraudulent insurance claims using historical insurance data.

The dataset contains information about policyholders, insurance policies, incident characteristics, claim amounts, and reported fraud.

Six classification algorithms are trained and evaluated to compare their ability to distinguish fraudulent claims from legitimate claims.

The project also applies data preprocessing, exploratory data analysis (EDA), feature selection, SMOTE, hyperparameter tuning, and model evaluation techniques.

### 💡 Why This Project Matters

An effective fraud detection system can help insurance companies:

* 🔎 Identify suspicious claims for further investigation.
* 💰 Reduce potential financial losses from fraudulent claims.
* ⚡ Improve the efficiency of claims investigation.
* 📊 Understand factors associated with fraudulent claims.
* 🎯 Support data-driven insurance risk assessment.

**Note:** This project identifies patterns associated with reported fraud. Its predictions should support human investigation rather than automatically determine whether a claim is fraudulent.

---

## 🎯 Problem Statement

The objective is to develop a machine learning model that predicts whether an insurance claim has been reported as fraudulent.

| Component         | Description                 |
| ----------------- | --------------------------- |
| Problem Type      | Supervised Machine Learning |
| Learning Task     | Binary Classification       |
| Target Variable   | `fraud_reported`            |
| Negative Class    | `N` — No fraud reported     |
| Positive Class    | `Y` — Fraud reported        |
| Dataset Size      | 1,000 records               |
| Original Features | 40 columns                  |

The main challenge is identifying fraudulent claims without relying exclusively on overall accuracy, especially when fraudulent and legitimate claims are not equally represented.

---

## 🎯 Project Objectives

* Analyze the structure and characteristics of insurance claims data.
* Identify and handle missing values.
* Explore numerical variables and their correlations.
* Examine categorical features and their distinct-value counts.
* Remove irrelevant, redundant, and high-cardinality columns.
* Encode categorical variables into numerical representations.
* Standardize numerical features.
* Address class imbalance using SMOTE.
* Train and compare six classification algorithms.
* Tune Decision Tree and Random Forest hyperparameters.
* Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
* Investigate important features associated with reported fraud.
* Save and reload the trained Random Forest model using Joblib.

---

## 📁 Dataset

**Dataset file:** `insurance_claims.csv`

The notebook loads the dataset using Pandas.

```python
df = pd.read_csv("insurance_claims.csv")
```

### Dataset Summary

| Property                     | Details                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| Number of Records            | 1,000                                                             |
| Original Number of Columns   | 40                                                                |
| Target Variable              | `fraud_reported`                                                  |
| Numerical Features           | Policy, customer, incident, and claim-related measurements        |
| Categorical Features         | Policy details, customer attributes, and incident characteristics |
| Missing-Value Representation | `?`                                                               |
| Additional Empty Column      | `_c39`                                                            |

### Important Features

The dataset contains several types of information.

| Feature                       | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| `months_as_customer`          | Duration of the customer's relationship with the insurer |
| `age`                         | Age of the insured customer                              |
| `policy_state`                | State associated with the insurance policy               |
| `policy_csl`                  | Combined single-limit coverage information               |
| `policy_deductable`           | Policy deductible amount                                 |
| `policy_annual_premium`       | Annual insurance premium                                 |
| `umbrella_limit`              | Umbrella insurance coverage limit                        |
| `insured_sex`                 | Sex of the insured customer                              |
| `insured_education_level`     | Education level of the insured                           |
| `insured_occupation`          | Occupation of the insured                                |
| `insured_relationship`        | Relationship category of the insured                     |
| `incident_type`               | Type of insurance incident                               |
| `collision_type`              | Type of collision involved                               |
| `incident_severity`           | Severity of the reported incident                        |
| `authorities_contacted`       | Authorities contacted following the incident             |
| `incident_hour_of_the_day`    | Hour when the incident occurred                          |
| `number_of_vehicles_involved` | Number of vehicles involved                              |
| `bodily_injuries`             | Number of bodily injuries                                |
| `witnesses`                   | Number of witnesses                                      |
| `total_claim_amount`          | Total amount claimed                                     |
| `injury_claim`                | Amount associated with injury claims                     |
| `property_claim`              | Amount associated with property claims                   |
| `vehicle_claim`               | Amount associated with vehicle claims                    |
| `auto_make`                   | Vehicle manufacturer                                     |
| `auto_model`                  | Vehicle model                                            |
| `auto_year`                   | Vehicle manufacturing year                               |
| `fraud_reported`              | Target indicating whether fraud was reported             |

*The table highlights selected columns from the original dataset; it is not an exhaustive feature listing.*

### Target Variable Distribution

The notebook's analysis identifies an imbalance between fraudulent and legitimate claims.

* **Approximately 75%:** Legitimate claims (`N`)
* **Approximately 25%:** Fraudulent claims (`Y`)

This imbalance makes fraud-class recall and F1-score particularly important when evaluating model performance.

---

## 🛠️ Technologies Used

| Technology         | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| Python             | Programming and model development               |
| Pandas             | Data manipulation and preprocessing             |
| NumPy              | Numerical computations                          |
| Matplotlib         | Data visualization                              |
| Seaborn            | Statistical visualization and heatmaps          |
| Scikit-learn       | Machine learning, preprocessing, and evaluation |
| SMOTE              | Synthetic minority-class oversampling           |
| GridSearchCV       | Hyperparameter search for the Decision Tree     |
| RandomizedSearchCV | Hyperparameter search for the Random Forest     |
| Joblib             | Saving and loading the trained model            |
| Jupyter Notebook   | Interactive development and experimentation     |

---

## 🧪 Project Workflow

### 1. Data Loading and Understanding

* Loaded the insurance claims dataset using Pandas.
* Examined the first few records.
* Inspected column names, data types, and dataset dimensions.
* Identified numerical and categorical variables.
* Investigated the number of distinct values in each column.

### 2. Data Cleaning and Missing-Value Treatment

The dataset uses `?` to represent missing values in several categorical columns.

These placeholders were converted to proper missing values using NumPy.

```python
df.replace('?', np.nan, inplace=True)
```

The notebook identified missing values in the following columns:

| Column                    | Missing Values |
| ------------------------- | -------------: |
| `collision_type`          |            178 |
| `authorities_contacted`   |             91 |
| `property_damage`         |            360 |
| `police_report_available` |            343 |
| `_c39`                    |          1,000 |

The four partially missing categorical columns were imputed using their respective most frequent values (mode).

The fully empty `_c39` column was removed during feature reduction.

### 3. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand the data before model training.

The notebook includes:

* Missing-value visualization.
* Numerical-feature correlation heatmap.
* Distinct-value count analysis.
* Feature-level examination of categorical and numerical variables.
* Investigation of relationships among policy, incident, and claim-related features.

Correlation analysis and cardinality checks were used to guide the removal of redundant and non-predictive columns.

### 4. Feature Selection and Data Reduction

The original dataset contained 40 columns.

The notebook removed 15 non-predictive or redundant columns, leaving 24 features for modeling, including the target.

Examples of columns removed include:

* `policy_number`
* `insured_zip`
* `auto_model`
* `insured_hobbies`
* `_c39`

The reduction process considered identifier fields, high-cardinality variables, redundant information, and correlation analysis.

### 5. Categorical Encoding

Machine learning algorithms require numerical representations of input features.

Categorical variables were transformed using one-hot encoding.

```python
categorical_col = x.select_dtypes(include=['object'])

categorical_col = pd.get_dummies(
    categorical_col,
    drop_first=True,
    dtype=int
)
```

The encoded categorical features were combined with the numerical features to prepare the modeling dataset.

### 6. Feature Scaling

Numerical features were standardized using `StandardScaler`.

Standardization transforms numerical values to a common scale, which can be particularly useful for distance-based and gradient-based algorithms.

Scaling helps prevent variables with larger numerical magnitudes from dominating models that are sensitive to feature scale.

### 7. Train-Test Split

The prepared dataset was divided into training and testing sets.

| Dataset      | Proportion |
| ------------ | ---------: |
| Training Set |        80% |
| Testing Set  |        20% |

With 1,000 original records, the split corresponds to approximately 800 training records and 200 testing records before any oversampling.

The test set is retained for evaluating model performance on data not used for model fitting.

### 8. Handling Class Imbalance with SMOTE

The dataset contains fewer fraudulent claims than legitimate claims.

To address this issue, the project applies **Synthetic Minority Over-sampling Technique (SMOTE)** to the training data.

SMOTE generates synthetic examples of the minority class to help the model learn patterns associated with fraudulent claims.

The intended workflow is:

1. Split the dataset into training and testing sets.
2. Apply SMOTE to the training data.
3. Train the classification models.
4. Evaluate the models on the original, unresampled test set.

This approach avoids directly resampling the test set and helps preserve a more realistic evaluation distribution.

### 9. Model Training

Six classification algorithms were trained and compared:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. K-Nearest Neighbors (KNN)
4. Decision Tree Classifier
5. Random Forest Classifier
6. Gradient Boosting Classifier

Each algorithm offers a different approach to classification, allowing performance comparisons across linear, distance-based, tree-based, and ensemble models.

### 10. Hyperparameter Tuning

Hyperparameter tuning was performed for the Decision Tree and Random Forest models.

**Decision Tree**

GridSearchCV was used with five-fold cross-validation to evaluate candidate hyperparameter configurations.

**Random Forest**

RandomizedSearchCV was used with five-fold cross-validation.

The search considered parameters including:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* `criterion`

The Random Forest search used 20 sampled parameter combinations.

The tuned Random Forest was subsequently evaluated on the test set.

### 11. Model Evaluation

The models were compared using multiple evaluation metrics:

* Accuracy
* Precision for the fraud class
* Recall for the fraud class
* F1-score for the fraud class
* ROC-AUC

The notebook also generates confusion matrices and ROC curves to provide additional information about model behavior.

### 12. Model Saving and Loading

The tuned Random Forest model was saved using Joblib.

```python
import joblib

joblib.dump(best_rf, "insurance_fraud_model.pkl")
```

The saved model was subsequently loaded and evaluated.

```python
loaded_model = joblib.load("insurance_fraud_model.pkl")

prediction = loaded_model.predict(x_test)
```

The notebook reports an accuracy of 0.79 for the loaded model on the test set.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

A linear classification algorithm used to estimate the probability of a claim belonging to the fraudulent class.

### 2. Support Vector Classifier (SVC)

A classification algorithm that identifies a decision boundary separating the classes. Its effectiveness depends on the feature representation and model parameters.

### 3. K-Nearest Neighbors (KNN)

A distance-based algorithm that predicts a class based on nearby training examples.

The notebook evaluates different values of `k` and selects the value with the highest test accuracy.

### 4. Decision Tree Classifier

A tree-based algorithm that recursively divides the feature space using decision rules.

GridSearchCV is used to tune its hyperparameters.

### 5. Random Forest Classifier

An ensemble algorithm that combines multiple decision trees to generate predictions.

The notebook applies RandomizedSearchCV and examines feature importance to understand which variables contribute most to the model.

The tuned Random Forest is also saved as a reusable model artifact.

### 6. Gradient Boosting Classifier

An ensemble algorithm that builds trees sequentially, with each stage attempting to improve the model's predictions.

Gradient Boosting achieved the highest reported accuracy and fraud-class F1-score in the notebook's final model comparison.

---

## 📊 Model Performance

The following table uses the metrics reported in the notebook's final comparison output.

| Model                     | Accuracy | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC |
| ------------------------- | -------: | --------------: | -----------: | -------------: | ------: |
| Gradient Boosting         |    0.820 |           0.633 |        0.633 |          0.633 |   0.775 |
| Logistic Regression       |    0.820 |           0.644 |        0.592 |          0.617 |   0.763 |
| Decision Tree (Tuned)     |    0.765 |           0.516 |        0.673 |          0.584 |   0.737 |
| Random Forest (Tuned)     |    0.790 |           0.600 |        0.429 |          0.500 |   0.791 |
| Support Vector Classifier |    0.795 |           0.633 |        0.388 |          0.481 |   0.783 |
| K-Nearest Neighbors       |    0.505 |           0.284 |        0.673 |          0.400 |   0.562 |

*These values are taken from the notebook's final comparison outputs. Minor differences may occur if the notebook is rerun with different environments or configurations.*

### 🏆 Best Model Depends on the Evaluation Metric

**Highest accuracy: Gradient Boosting and Logistic Regression**

Both models achieved an accuracy of 82%.

**Highest fraud-class F1-score: Gradient Boosting**

Gradient Boosting achieved a fraud precision of 0.633, recall of 0.633, and F1-score of 0.633.

**Highest ROC-AUC: Tuned Random Forest**

The tuned Random Forest achieved a ROC-AUC of approximately 0.791, indicating the strongest ranking performance among the six models according to this metric.

**Saved model: Tuned Random Forest**

The tuned Random Forest was selected for model persistence in the notebook and saved as `insurance_fraud_model.pkl`.

The saved model is a useful artifact for further testing, but its selection should not be interpreted as proof that it is the best model for every fraud detection objective.

---

## 📈 Model Evaluation

### 1. Confusion Matrix

The tuned Random Forest produced the following confusion matrix on the 200-record test set:

| Actual / Predicted | Non-Fraud | Fraud |
| ------------------ | --------: | ----: |
| Non-Fraud          |       137 |    14 |
| Fraud              |        28 |    21 |

This corresponds to:

* **True Negatives:** 137
* **False Positives:** 14
* **False Negatives:** 28
* **True Positives:** 21

The model correctly classified 137 legitimate claims and identified 21 fraudulent claims.

However, it failed to identify 28 of the 49 fraudulent claims in the test set.

This highlights an important limitation: although the Random Forest achieved 79% accuracy, its fraud-class recall was approximately 42.9%.

### 2. Classification Report

The tuned Random Forest achieved the following fraud-class metrics:

| Metric    | Score |
| --------- | ----: |
| Precision |  0.60 |
| Recall    |  0.43 |
| F1-score  |  0.50 |

These metrics indicate that the model identifies some fraudulent claims successfully but still misses a considerable proportion of actual fraudulent cases.

### 3. ROC Curve

ROC curves were generated to compare the classification models at different decision thresholds.

The notebook also calculates ROC-AUC for each model.

ROC-AUC complements accuracy by evaluating how effectively a model ranks positive examples above negative examples across different thresholds.

### 4. Precision, Recall, and F1-score Comparison

The notebook generates a comparison chart of fraud-class precision, recall, and F1-score.

This visualization helps identify the trade-offs between:

* Correctly identifying fraudulent claims.
* Avoiding false fraud alerts.
* Balancing precision and recall.

---

## 🔎 Feature Importance

Feature importance was extracted from the tuned Random Forest model.

The notebook's feature-importance analysis identified the following as influential predictors:

1. `incident_severity`
2. `vehicle_claim`
3. `property_claim`

Other features also contributed to the model's predictions.

Feature importance can help guide additional investigation and improve understanding of the model's behavior.

However, feature importance indicates a model's reliance on a feature; it does not prove that the feature causes insurance fraud.

---

## 📌 Key Insights

### 1. Accuracy Alone Is Not Enough

A model can achieve relatively high accuracy while missing a substantial number of fraudulent claims.

For fraud detection, recall and F1-score should be considered alongside accuracy.

### 2. Gradient Boosting Provides Balanced Fraud Detection

Gradient Boosting achieved the highest reported fraud-class F1-score of 0.633.

Its precision and recall were both approximately 0.633, making it the strongest model in the reported comparison when balancing these two metrics.

### 3. Random Forest Has the Highest Reported ROC-AUC

The tuned Random Forest achieved a ROC-AUC of approximately 0.791.

However, its fraud recall of 0.429 indicates that the default classification predictions miss many fraudulent claims.

### 4. Decision Tree and KNN Show Different Trade-offs

The tuned Decision Tree achieved a fraud recall of 0.673 but lower precision and overall accuracy than the strongest models.

KNN also achieved a fraud recall of 0.673, but its low accuracy and ROC-AUC suggest weak overall discrimination in this experiment.

### 5. Claim-Related Features Matter

Feature-importance analysis highlighted incident severity and claim amounts as important predictors in the Random Forest model.

These patterns can support further investigation but should be validated on additional data before being used in operational decisions.

---

## ⚠️ Limitations

* The dataset contains only 1,000 records, which limits the amount of information available for model development.
* The dataset contains class imbalance between fraudulent and legitimate claims.
* The Random Forest's fraud recall is relatively low, meaning many fraudulent claims remain undetected.
* Model performance may vary with different train-test splits and hyperparameter configurations.
* Feature importance does not establish causality.
* The notebook does not demonstrate deployment in a production environment.
* Further validation is needed before using the model for real insurance claims decisions.

---

## 🚀 Future Improvements

* Optimize classification thresholds to improve fraud detection recall.
* Compare SMOTE with class-weighted learning and other imbalance-handling approaches.
* Apply stratified cross-validation to obtain more robust performance estimates.
* Investigate additional hyperparameter configurations for the classification models.
* Evaluate model calibration and probability-based decision-making.
* Perform more detailed error analysis on false-positive and false-negative predictions.
* Use explainability techniques such as SHAP to investigate individual predictions.
* Validate the model on a larger and more recent insurance claims dataset.
* Build an interactive Streamlit dashboard for exploring model predictions.
* Develop an API for integrating predictions into a claims review workflow.
* Monitor performance and potential data drift if the model is deployed.

---

## ⚙️ Installation and Usage

### Prerequisites

* Python 3.8 or later
* Jupyter Notebook
* The insurance claims dataset

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 3. Add the Dataset

Place `insurance_claims.csv` in the directory expected by the notebook.

The dataset is not included automatically with the notebook, so ensure that you have the required file before executing the code.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Project

Open the project notebook:

`project ML(update) (2)(1).ipynb`

Run the cells sequentially to reproduce the data preprocessing, model training, evaluation, and model-saving workflow.

**Important:** The saved Random Forest model expects data with the same preprocessing and feature structure used during training. For a production-ready application, preprocessing should be saved and applied consistently with the model.

---

## 📂 Project Structure

A suggested repository structure is:

```text
Insurance-Fraud-Detection/
│
├── README.md
├── project ML(update) (2)(1).ipynb
├── insurance_claims.csv
├── insurance_fraud_model.pkl
└── requirements.txt
```

The dataset and model file should be included only if you intend to distribute them and have permission to do so.

The `requirements.txt` file can contain the libraries required to reproduce the project.

---

## 📜 Conclusion

This project demonstrates the application of supervised machine learning to insurance fraud detection.

The workflow covers data cleaning, missing-value imputation, exploratory data analysis, feature reduction, categorical encoding, feature scaling, class-imbalance handling with SMOTE, model training, hyperparameter tuning, and performance evaluation.

Six classification algorithms were compared using accuracy, precision, recall, F1-score, and ROC-AUC.

Gradient Boosting achieved the highest reported accuracy and fraud-class F1-score, while the tuned Random Forest achieved the highest reported ROC-AUC and was saved for subsequent use.

The results demonstrate the importance of evaluating multiple metrics when developing fraud detection systems. Further improvements in fraud recall, validation, and model deployment would be necessary before applying the approach to real-world insurance claims.

---

## 👤 Author

**Sathish Adilla**

🎓 M.Sc. Statistics and Computing | Banaras Hindu University (BHU)

💻 Aspiring Data Analyst / Data Scientist

**Areas of Interest:**

* Data Analytics
* Machine Learning
* Statistical Modeling
* Predictive Analytics
* Data-Driven Decision-Making
