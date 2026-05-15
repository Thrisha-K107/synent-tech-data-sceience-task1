# ============================================================
# Task 1: Data Cleaning & Preprocessing — Titanic Dataset
# Synent Technologies Data Science Internship
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")  # Suppress unnecessary warnings

# ============================================================
# Step 2: Load the Titanic Dataset
# Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# ============================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 60)
print("STEP 1: INITIAL DATA EXPLORATION")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")          # Number of rows and columns
print(f"\nColumn Names:\n{df.columns.tolist()}")  # List all column names
print(f"\nData Types:\n{df.dtypes}")             # Data type of each column
print(f"\nFirst 5 Rows:\n{df.head()}")           # Preview first 5 rows

# ============================================================
# Step 3: Check Missing Values BEFORE Cleaning
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: MISSING VALUES BEFORE CLEANING")
print("=" * 60)

missing_before = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_before,
    "Missing %": missing_percent
}).sort_values("Missing %", ascending=False)

print(missing_report[missing_report["Missing Count"] > 0])

# ============================================================
# Step 4: Handle Missing Values
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: HANDLING MISSING VALUES")
print("=" * 60)

# Fill 'Age' with median — robust to outliers
median_age = df["Age"].median()
df["Age"].fillna(median_age, inplace=True)
print(f"✔ 'Age' missing values filled with median: {median_age}")

# Fill 'Embarked' with mode (most frequent port)
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"].fillna(mode_embarked, inplace=True)
print(f"✔ 'Embarked' missing values filled with mode: {mode_embarked}")

# Drop 'Cabin' — too many missing values (>77%)
df.drop(columns=["Cabin"], inplace=True)
print("✔ 'Cabin' column dropped (too many missing values)")

# ============================================================
# Step 5: Remove Duplicate Rows
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: REMOVING DUPLICATES")
print("=" * 60)

before_dup = len(df)
df.drop_duplicates(inplace=True)  # Remove exact duplicate rows
after_dup = len(df)

print(f"✔ Rows before removing duplicates: {before_dup}")
print(f"✔ Rows after removing duplicates : {after_dup}")
print(f"✔ Duplicates removed: {before_dup - after_dup}")

# ============================================================
# Step 6: Convert and Fix Data Types
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DATA TYPE CONVERSION")
print("=" * 60)

# Convert 'Survived' and 'Pclass' to categorical
df["Survived"] = df["Survived"].astype("category")
df["Pclass"]   = df["Pclass"].astype("category")
df["Sex"]      = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print("✔ Converted 'Survived', 'Pclass', 'Sex', 'Embarked' → category")
print(f"\nUpdated Data Types:\n{df.dtypes}")

# ============================================================
# Step 7: Rename Columns for Clarity
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: RENAMING COLUMNS")
print("=" * 60)

df.rename(columns={
    "PassengerId" : "passenger_id",
    "Survived"    : "survived",
    "Pclass"      : "passenger_class",
    "Name"        : "full_name",
    "Sex"         : "gender",
    "Age"         : "age",
    "SibSp"       : "siblings_spouses",
    "Parch"       : "parents_children",
    "Ticket"      : "ticket_number",
    "Fare"        : "fare",
    "Embarked"    : "embarkation_port"
}, inplace=True)

print("✔ All columns renamed to snake_case format")
print(f"\nNew Column Names: {df.columns.tolist()}")

# ============================================================
# Step 8: Feature Engineering — Add Useful Columns
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 60)

# Create 'family_size' column
df["family_size"] = df["siblings_spouses"] + df["parents_children"] + 1
print("✔ 'family_size' column created (siblings + parents + self)")

# Create 'is_alone' binary flag
df["is_alone"] = (df["family_size"] == 1).astype(int)
print("✔ 'is_alone' column created (1 = traveling alone)")

# Create 'age_group' binned column
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)
print("✔ 'age_group' column created (Child/Teen/Young Adult/Adult/Senior)")

# ============================================================
# Step 9: Final Missing Value Check AFTER Cleaning
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: MISSING VALUES AFTER CLEANING")
print("=" * 60)

missing_after = df.isnull().sum()
print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "✔ No missing values remaining!")

# ============================================================
# Step 10: Statistical Summary of Clean Dataset
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: STATISTICAL SUMMARY (CLEAN DATASET)")
print("=" * 60)

print(df.describe(include="all"))

# ============================================================
# Step 11: Visualize Before/After Cleaning
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Titanic Dataset — Cleaning Overview", fontsize=14, fontweight="bold")

# Age distribution after cleaning
axes[0].hist(df["age"], bins=30, color="#2ecc71", edgecolor="black", alpha=0.8)
axes[0].set_title("Age Distribution (After Cleaning)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

# Survival by Gender
survival_gender = df.groupby(["gender", "survived"]).size().unstack()
survival_gender.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"],
                     edgecolor="black", alpha=0.85)
axes[1].set_title("Survival Count by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(["Did Not Survive", "Survived"])

plt.tight_layout()
plt.savefig("titanic_cleaning_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✔ Plot saved as 'titanic_cleaning_overview.png'")

# ============================================================
# Step 12: Export Clean Dataset
# ============================================================

df.to_csv("titanic_cleaned.csv", index=False)
print("\n✔ Clean dataset saved as 'titanic_cleaned.csv'")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE!")
print(f"Final Dataset Shape: {df.shape}")
print("=" * 60)
