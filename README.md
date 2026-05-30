# Titanic-Data-Cleaning-Preprocessing
Titanic Dataset Data Cleaning and Preprocessing using Python, Pandas, Seaborn and Scikit-Learn
# Titanic Dataset Data Cleaning & Preprocessing

##  Project Overview

This project demonstrates a complete **Data Cleaning and Preprocessing Pipeline** using the Titanic Dataset. The goal is to transform raw and inconsistent data into a clean, structured, and machine-learning-ready dataset through industry-standard preprocessing techniques.

Data preprocessing is a crucial step in the Machine Learning workflow because model performance heavily depends on data quality.

---

##  Objective

The objectives of this project are:

* Explore and understand the dataset
* Identify and handle missing values
* Encode categorical variables into numerical form
* Detect and remove outliers
* Standardize numerical features
* Analyze relationships between features
* Export the cleaned dataset for future machine learning applications

---

##  Tools & Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

##  Dataset Information

### Original Dataset

| Metric   | Value |
| -------- | ----- |
| Records  | 891   |
| Features | 12    |

### Features Included

* PassengerId
* Survived
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

---

##  Data Preprocessing Workflow

```text
Raw Dataset
     │
     ▼
Dataset Exploration
     │
     ▼
Missing Value Analysis
     │
     ▼
Missing Value Treatment
     │
     ▼
Categorical Encoding
     │
     ▼
Outlier Detection
     │
     ▼
Outlier Removal
     │
     ▼
Feature Scaling
     │
     ▼
Correlation Analysis
     │
     ▼
Clean Dataset Export
```

---

## 1️⃣ Dataset Exploration

Performed:

* Dataset shape inspection
* Data type analysis
* Statistical summary generation
* Missing value identification

This helped understand the overall structure and quality of the dataset.

---

## 2️⃣ Missing Value Analysis

### Missing Values Detected

| Feature  | Missing Values | Percentage |
| -------- | -------------- | ---------- |
| Age      | 177            | 19.87%     |
| Cabin    | 687            | 77.10%     |
| Embarked | 2              | 0.22%      |

### Visualization

A heatmap was generated to visualize the distribution of missing values across the dataset.

---

## 3️⃣ Missing Value Treatment

### Age

Median Imputation was applied:

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

### Embarked

Mode Imputation was applied:

```python
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
```

### Cabin

The Cabin column was removed because over 77% of its values were missing.

```python
df.drop("Cabin", axis=1, inplace=True)
```

### Result

All missing values were successfully handled.

---

## 4️⃣ Categorical Feature Encoding

Machine Learning algorithms require numerical inputs. Therefore categorical features were converted using **One-Hot Encoding**.

### Encoded Features

* Sex → Sex_male
* Embarked → Embarked_Q, Embarked_S

Implementation:

```python
pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)
```

---

## 5️⃣ Outlier Detection

Outliers were identified in the Fare feature using boxplots.

### Observation

The Fare feature contained several extreme values, including values above 500, which could negatively affect model performance.

---

## 6️⃣ Outlier Removal

The **Interquartile Range (IQR)** method was used.

### Formula

* IQR = Q3 − Q1
* Lower Bound = Q1 − 1.5 × IQR
* Upper Bound = Q3 + 1.5 × IQR

Records outside these limits were removed.

### Dataset Size Comparison

| Stage                 | Records |
| --------------------- | ------- |
| Original Dataset      | 891     |
| After Outlier Removal | 775     |

---

## 7️⃣ Feature Scaling

Numerical features were standardized using **StandardScaler**.

### Scaled Features

* Age
* Fare

Implementation:

```python
from sklearn.preprocessing import StandardScaler
```

### Benefits

* Improves model performance
* Prevents feature dominance
* Ensures comparable feature scales

---

## 8️⃣ Correlation Analysis

A correlation matrix was generated to study relationships among features.

### Key Insights

* Sex_male vs Survived = -0.50

  * Male passengers were less likely to survive.

* Fare vs Survived = 0.23

  * Higher fare passengers had better survival rates.

* Pclass vs Fare = -0.59

  * Passenger class strongly influenced fare amount.

---

##  Visualizations Included

* Missing Values Heatmap
* Fare Distribution Before Outlier Removal
* Fare Distribution After Outlier Removal
* Feature Correlation Matrix

---

##  Final Results

| Metric                         | Value |
| ------------------------------ | ----- |
| Original Records               | 891   |
| Final Records                  | 775   |
| Missing Values Removed         | 100%  |
| Categorical Features Encoded   | Yes   |
| Outliers Removed               | Yes   |
| Feature Scaling Applied        | Yes   |
| Correlation Analysis Completed | Yes   |

---

##  Project Structure

```text
Titanic-Data-Cleaning/
│
├── Titanic_Data_Cleaning.ipynb
├── Cleaned_Titanic_Dataset.csv
├── README.md
└── screenshots/
```

---

##  Learning Outcomes

Through this project, the following concepts were implemented and understood:

* Data Cleaning
* Missing Value Treatment
* Exploratory Data Analysis (EDA)
* One-Hot Encoding
* Outlier Detection using IQR
* Outlier Removal
* Feature Scaling using StandardScaler
* Correlation Analysis
* Machine Learning Data Preparation

---

##  Conclusion

The Titanic dataset was successfully preprocessed using industry-standard data cleaning techniques.

The project included:

* Missing Value Handling
* Categorical Feature Encoding
* Outlier Detection and Removal
* Feature Scaling
* Correlation Analysis
* Dataset Export

The final cleaned dataset contains **775 rows and 12 columns** and is fully prepared for machine learning model development and predictive analytics.

---

## 👨‍💻 Author

Sonali Gupta

