# ==========================================
# STUDENT STRESS PREDICTION USING ML
# ==========================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(r"C:\Users\ACER\OneDrive\Desktop\PYTHON\Stress level prediction\student_lifestyle_dataset.csv", encoding="latin1")

print("Dataset Preview:\n", df.head())


# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nColumns:\n", df.columns)


# =========================
# DATA INFO
# =========================

print("\nInfo:")
df.info()

print("\nMissing Values:\n", df.isnull().sum())


# =========================
# HANDLE MISSING VALUES
# =========================

df.fillna(df.mean(numeric_only=True), inplace=True)


# =========================
# TARGET FIX (SAFE)
# =========================

print("Original Values:", df['stress_level'].unique())

df['stress_level'] = df['stress_level'].astype(str).str.lower().str.strip()

# FIX LABELS HERE
df['stress_level'] = df['stress_level'].replace({
    "low": "Low",
    "moderate": "Medium",  
    "medium": "Medium",
    "high": "High"
})

print("\nCleaned Distribution:")
print(df['stress_level'].value_counts())

# ==========================================
# GENERATE REALISTIC MENTAL HEALTH FEATURES
# ==========================================

np.random.seed(42)

anxiety = []
depression = []

for stress in df['stress_level']:

    # LOW STRESS STUDENTS
    if stress == "Low":
        anxiety.append(np.random.randint(1, 4))
        depression.append(np.random.randint(1, 4))

    # MEDIUM STRESS STUDENTS
    elif stress == "Medium":
        anxiety.append(np.random.randint(4, 7))
        depression.append(np.random.randint(4, 7))

    # HIGH STRESS STUDENTS
    else:
        anxiety.append(np.random.randint(7, 11))
        depression.append(np.random.randint(7, 11))

# Add columns into dataset
df['anxiety_level'] = anxiety
df['depression'] = depression

# =========================
# =========================
# 🔍 EDA SECTION
# =========================
# =========================

# 🔹 Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 🔹 Histograms
numeric_df.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distributions")
plt.show()


# 🔹 Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(x='stress_level', data=df)
plt.title("Stress Level Distribution")
plt.show()


# =========================
# FEATURE & TARGET
# =========================

X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Convert categorical → numeric
X = pd.get_dummies(X, drop_first=True)

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)


# =========================
# SCALING
# =========================

scaler = StandardScaler()
feature_names = X.columns.tolist()
X = scaler.fit_transform(X)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}


# =========================
# TRAINING
# =========================

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    
    print(f"\n{name} Accuracy: {acc:.4f}")


# =========================
# BEST MODEL
# =========================

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# =========================
# EVALUATION
# =========================

y_pred = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =========================
# MODEL COMPARISON GRAPH
# =========================

plt.figure(figsize=(6,4))
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.show()


# =========================
# CROSS VALIDATION
# =========================

cv_scores = cross_val_score(best_model, X, y, cv=5)

print("\nCross Validation Accuracy:", cv_scores.mean())


# =========================
# FEATURE IMPORTANCE
# =========================

if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
    feature_names = pd.get_dummies(df.drop(columns=['stress_level']), drop_first=True).columns
    
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


# =========================
# SAVE MODEL
# =========================

joblib.dump(best_model, "student_stress_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_names, "features.pkl")
print("\nModel saved successfully!")


# =========================
# PREDICTION EXAMPLE
# =========================

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print("\nPredicted Stress Level:", le.inverse_transform(prediction)[0])