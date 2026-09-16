# 💰 Loan Approval Prediction using Decision Tree

## 📌 Project Overview

This project predicts whether a **loan application will be approved or rejected** using a **Decision Tree Classification** algorithm.

The model uses applicant-related information such as:

* Age
* Income
* Credit Score
* Loan Amount
* Employment Years
* Existing Loans
* Number of Dependents
* Loan Term

The target variable is `approval`, where:

* `1` → Loan Approved
* `0` → Loan Rejected

---

## 🎯 Objective

The main objective of this project is to build a machine learning classification model that can learn patterns from applicant information and predict the loan approval status of a new applicant.

---

## 📊 Dataset

The dataset is created using Python and Pandas.

It contains **32 loan application records**.

### Features

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| `age`              | Age of the applicant                        |
| `income`           | Annual/monthly income used in the dataset   |
| `credit_score`     | Applicant's credit score                    |
| `loan_amount`      | Requested loan amount                       |
| `employment_years` | Number of years employed                    |
| `existing_loans`   | Number of existing loans                    |
| `dependents`       | Number of dependents                        |
| `loan_term`        | Loan repayment period                       |
| `approval`         | Target variable: 0 = Rejected, 1 = Approved |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Decision Tree Classifier

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
```

Move into the project directory:

```bash
cd loan-approval-prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 🔄 Machine Learning Workflow

The project follows these steps:

```text
Dataset Creation
       ↓
DataFrame Creation
       ↓
Data Exploration
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Decision Tree Model
       ↓
Model Training
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Decision Tree Visualization
```

---

## 🧠 Algorithm Used

### Decision Tree Classifier

A **Decision Tree** is a supervised machine learning algorithm used for classification and regression.

In this project, the Decision Tree learns relationships between applicant information and the loan approval result.

For example, the model may learn patterns involving:

```text
Credit Score
     ↓
Income
     ↓
Existing Loans
     ↓
Loan Amount
     ↓
Approval
```

The actual decision rules are learned automatically from the training data.

---

## 💻 Basic Implementation

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Create the Dataset

```python
data = {
    "age": [...],
    "income": [...],
    "credit_score": [...],
    "loan_amount": [...],
    "employment_years": [...],
    "existing_loans": [...],
    "dependents": [...],
    "loan_term": [...],
    "approval": [...]
}

df = pd.DataFrame(data)
```

### 3. Separate Features and Target

```python
X = df.drop("approval", axis=1)
y = df["approval"]
```

### 4. Split the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 5. Create the Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)
```

### 6. Train the Model

```python
model.fit(X_train, y_train)
```

### 7. Make Predictions

```python
y_pred = model.predict(X_test)
```

### 8. Evaluate the Model

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

---

## 🌳 Decision Tree Visualization

The trained decision tree can be visualized using Matplotlib.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True
)

plt.show()
```

This helps understand how the model makes its predictions.

---

## 🔮 Predicting a New Applicant

A new applicant can be provided to the trained model:

```python
new_applicant = [[
    30,       # age
    50000,    # income
    680,      # credit score
    250000,   # loan amount
    5,        # employment years
    1,        # existing loans
    2,        # dependents
    20        # loan term
]]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
```

---

## 📈 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

## 📁 Project Structure

```text
loan-approval-prediction/
│
├── loan_approval.py
├── README.md
├── requirements.txt
└── images/
    └── decision_tree.png
```

---

## ⚠️ Important Note

This project uses a **small, manually created dataset** for educational and demonstration purposes.

The model should **not be used for real-world loan approval decisions** without a much larger, representative dataset, proper validation, fairness analysis, and appropriate financial/regulatory review.

---

## 🚀 Future Improvements

The project can be improved by:

* Using a real-world loan dataset
* Increasing the number of training records
* Handling missing values
* Performing exploratory data analysis
* Adding a confusion matrix visualization
* Using cross-validation
* Hyperparameter tuning
* Comparing Decision Tree with Random Forest
* Building a simple web interface using Flask or Streamlit
* Adding applicant input forms
* Saving the trained model using Joblib

---

## 👨‍💻 Author

**Ragul Balajee G.K**

Computer Science Engineering Student

---

## ⭐ Conclusion

This project demonstrates how a **Decision Tree Classification** model can be used to predict loan approval based on applicant information.

It provides a beginner-friendly example of the complete machine learning workflow, from dataset creation and preprocessing to model training, prediction, evaluation, and visualization.
