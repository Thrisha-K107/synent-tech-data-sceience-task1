# synent-tech-data-sceience-task1
My first Internship with Task 1
# 🧹 Task 1: Data Cleaning & Preprocessing
### Synent Technologies — Data Science Internship

---

## 📌 Problem Statement

Raw datasets are rarely ready for analysis. The Titanic dataset contains missing values, duplicate records, inconsistent data types, and poorly named columns. This project demonstrates a complete end-to-end data cleaning pipeline to transform raw, messy data into a clean, analysis-ready dataset.

---

## 📂 Dataset Details

| Property | Details |
|---|---|
| **Name** | Titanic Dataset |
| **Source** | [Seaborn / Kaggle](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) |
| **Rows** | 891 |
| **Columns** | 12 |
| **Target Column** | `Survived` (0 = No, 1 = Yes) |

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- `pandas` — data manipulation and cleaning
- `numpy` — numerical operations
- `matplotlib` — data visualization
- `seaborn` — statistical visualization

---

## 🔍 Step-by-Step Code Explanation

### Step 1 — Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```
> Imports all required libraries. `warnings.filterwarnings("ignore")` suppresses irrelevant runtime warnings that clutter output.

---

### Step 2 — Load Dataset
```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
```
> Loads the Titanic dataset directly from a public URL using `pd.read_csv()`. `df` is our main DataFrame — a table-like structure.

---

### Step 3 — Initial Exploration
```python
print(df.shape)       # Outputs (891, 12) — rows × columns
print(df.dtypes)      # Shows data type of each column
print(df.head())      # Previews first 5 rows
```
> Before cleaning anything, we understand the data. `shape` tells us size, `dtypes` reveals type mismatches, `head()` shows sample rows.

---

### Step 4 — Check Missing Values
```python
missing_before = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
```
> `isnull().sum()` counts `NaN` (missing) values per column. We also compute missing percentage to decide the best strategy.

---

### Step 5 — Handle Missing Values
```python
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin"], inplace=True)
```
> - **Age** → Filled with **median** (not mean) because median is robust to outliers like extreme ages.
> - **Embarked** → Filled with **mode** (most frequent value: 'S') since it's a categorical column.
> - **Cabin** → **Dropped** entirely because over 77% of values are missing — imputing would introduce too much noise.

---

### Step 6 — Remove Duplicates
```python
df.drop_duplicates(inplace=True)
```
> Removes rows where every column has the same value. Duplicate records inflate counts and skew analysis.

---

### Step 7 — Fix Data Types
```python
df["Survived"] = df["Survived"].astype("category")
df["Pclass"]   = df["Pclass"].astype("category")
```
> Converts numerical columns that represent categories (0/1, 1/2/3) to the `category` dtype. This saves memory and enables correct statistical behavior.

---

### Step 8 — Rename Columns
```python
df.rename(columns={
    "PassengerId": "passenger_id",
    "SibSp": "siblings_spouses",
    "Parch": "parents_children",
    ...
}, inplace=True)
```
> All column names are converted to **snake_case** for consistency and readability. This is standard Python/pandas convention.

---

### Step 9 — Feature Engineering
```python
df["family_size"] = df["siblings_spouses"] + df["parents_children"] + 1
df["is_alone"]    = (df["family_size"] == 1).astype(int)
df["age_group"]   = pd.cut(df["age"], bins=[0,12,18,35,60,100], labels=[...])
```
> - **family_size**: Total passengers travelling together (self + siblings + parents).
> - **is_alone**: Binary flag. `1` if travelling alone, `0` otherwise.
> - **age_group**: Bins continuous age into meaningful categories using `pd.cut()`.

---

### Step 10 — Export Clean Dataset
```python
df.to_csv("titanic_cleaned.csv", index=False)
```
> Saves the cleaned DataFrame to a CSV file. `index=False` avoids writing the row numbers as a column.

---

## 📊 Results

| Metric | Before | After |
|---|---|---|
| Missing in Age | 177 | 0 |
| Missing in Embarked | 2 | 0 |
| Cabin column | Present (77% missing) | Removed |
| Column naming | Mixed case | Consistent snake_case |
| Feature columns | 12 | 14 (2 new features added) |

---

## 📁 Output Files

| File | Description |
|---|---|
| `titanic_cleaned.csv` | Final cleaned dataset |
| `titanic_cleaning_overview.png` | Visualization of age distribution and survival by gender |

---

## ▶️ How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn

# Run the script
python task1_data_cleaning.py
```

---

## 🔗 Repository

`synent-task1-datacleaning-<yourname>`

---

*Synent Technologies Data Science Internship — Task 1*
