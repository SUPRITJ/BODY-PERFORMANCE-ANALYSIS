import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "bodyPerformance.csv")
df = pd.read_csv(data_path)

# Display basic info
print(df.info())
print(df.describe())

# Data Cleaning & Encoding
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])  # Encode gender (0: Female, 1: Male)
df['class'] = encoder.fit_transform(df['class'])  # Encode class labels

# Descriptive Statistics & Distribution Analysis
plt.figure(figsize=(12, 6))
sns.histplot(df['body fat_%'], bins=30, kde=True)
plt.title('Distribution of Body Fat Percentage')
plt.show()

# Gender-Based Comparisons
grouped = df.groupby('gender').mean()
print("Average values by gender:\n", grouped)

# Perform t-test for gender differences in grip force
t_stat, p_val = ttest_ind(df[df['gender'] == 0]['gripForce'], df[df['gender'] == 1]['gripForce'])
print(f"T-test for grip force: t-stat = {t_stat}, p-value = {p_val}")

# Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Predictive Modeling - Classifying Performance Levels
features = df.drop(columns=['class'])
target = df['class']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()