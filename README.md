# House Price Prediction using Machine Learning

## Overview

This project focuses on predicting house prices using Machine Learning techniques. The objective is to build an accurate regression model capable of estimating the price of a house based on several important factors such as location, property type, area, number of rooms, and other housing attributes.

The project follows a complete end-to-end Machine Learning workflow starting from data cleaning and preprocessing to feature engineering, model building, and evaluation.

---

## Project Objective

* Analyze the housing dataset and understand the factors affecting house prices.
* Clean and preprocess the data to make it suitable for modeling.
* Build and compare multiple regression models.
* Predict house prices with better accuracy.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset Description

The dataset contains different features related to houses, such as:

* Location
* Type of House
* Area / Square Feet
* Number of Bedrooms
* Number of Bathrooms
* Price
* Other numerical and categorical attributes

---

## Project Workflow

### 1. Data Import

* Loaded the dataset using Pandas.
* Explored the structure of the dataset using:

  * `head()`
  * `info()`
  * `describe()`
  * `shape`

```python
import pandas as pd

df = pd.read_csv('house_price.csv')
print(df.head())
```

---

### 2. Data Cleaning

Data cleaning was performed to improve data quality and remove inconsistencies.

#### Missing Value Handling

* Checked null values in each column.
* Dropped columns having more than 80% missing values.
* Used forward fill (`ffill`) for columns having only 1–2 missing values.
* Replaced missing values in integer columns with the mean value.
* Replaced missing values in float columns with the mean value.
* Replaced missing values in categorical/object columns with the mode value.

```python
df.drop(columns = ['Pool QC','Fence','Misc Feature'],inplace=True)
df.drop(columns=['Fireplace Qu','Mas Vnr Type','Alley'], inplace=True)

# Removed null values with ffill() method
col=['BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF','Total Bsmt SF','Electrical','Bsmt Full Bath','Bsmt Half Bath','Garage Cars',
'Garage Area' ]
for i in col:
    df[i]=df[i].ffill()
    print(i,'->',df[i].isnull().sum())

# Replaced integer column null values with the mean of respective column
int_col = list(df.select_dtypes(include='int').columns)
#int_col
for i in int_col:
    df[i]=df[i].fillna(df[i].mean())
    print(i,'->',df[i].isnull().sum())

float_col = list(df.select_dtypes(include='float').columns)
#float_col
for i in float_col:
    df[i]=df[i].fillna(df[i].mean())
    print(i,'->',df[i].isnull().sum())



---

### 3. Duplicate Value Handling

* Checked for duplicate rows in the dataset.
* Removed duplicate records to avoid biased predictions.

```python
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
```

---

### 4. Outlier Detection and Treatment

Outliers can negatively affect regression model performance. Therefore, outlier detection was performed using the Interquartile Range (IQR) method.

#### Steps:

* Calculated Q1 and Q3.
* Computed IQR.
* Determined lower and upper bounds.
* Replaced values below the lower bound and above the upper bound with the respective bound values.

```python
# Replaced outliers with the lower and upper bound
num_cols = df.select_dtypes(include='number').columns
for i in num_cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    df[i] = df[i].clip(lower,upper)


```

---

### 5. Encoding Categorical Features

Machine Learning models cannot directly process categorical data. Therefore, categorical columns such as `Location` and `Type` were encoded.

* Used Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in str_col:
    df[i] = encoder.fit_transform(df[i])
```

---

### 6. Feature Scaling

Feature scaling was applied to normalize the numerical features so that all variables contribute equally to the model.

* Used `StandardScaler` for scaling.

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Model Building

Two Machine Learning regression models were trained and evaluated.

### 1. Linear Regression

Linear Regression was used as a baseline model to understand the linear relationship between features and house prices.

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
```

### 2. Decision Tree Regressor

A Decision Tree Regressor was used to capture non-linear relationships in the dataset.

```python
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
```

---

## Model Evaluation

The models were evaluated using the following metrics:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

```python
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print('R2 Score: ',r2_score(y_test,y_pred))
print('MAE: ',mean_absolute_error(y_test,y_pred))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
```

---

## Results

* Successfully cleaned and preprocessed the housing dataset.
* Reduced the impact of missing values, duplicates, and outliers.
* Built two regression models for house price prediction.
* Compared model performance to identify the better-performing algorithm.
* Observed that the Decision Tree model captured complex patterns better, while Linear Regression provided a simple and interpretable baseline.

---



---

## Author

**Revanth KS**
Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning
