# 🧠 Student Stress Prediction System Using Machine Learning

A Machine Learning based web application that predicts the stress level of a student using lifestyle and academic-related factors.

The project provides an interactive Streamlit dashboard where users can enter their daily lifestyle information and receive an AI-based stress prediction.

---

## 🚀 Project Overview

Student stress can be influenced by several lifestyle and academic factors such as:

- Study hours
- Sleep hours
- Social media usage
- Physical activity
- Extracurricular activities
- Social activities
- GPA

This project uses Machine Learning classification algorithms to predict the student's stress level as:

- 🟢 Low
- 🟡 Medium
- 🔴 High

The trained Machine Learning model is integrated into a Streamlit web application for real-time prediction.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze student lifestyle data.
2. Perform data preprocessing and cleaning.
3. Explore relationships between different features.
4. Train multiple Machine Learning classification models.
5. Compare model performance.
6. Select the best-performing model.
7. Save the trained model for deployment.
8. Build an interactive Streamlit dashboard.
9. Provide real-time stress predictions.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

### Machine Learning Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📊 Dataset

The project uses:

`student_lifestyle_dataset.csv`

The dataset contains student lifestyle and academic-related information used for stress-level prediction.

Important features include:

- Student ID
- Study hours per day
- Extracurricular hours per day
- Sleep hours per day
- Social hours per day
- Physical activity hours per day
- GPA
- Stress level

---

## 🔄 Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Feature Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Serialization
   ↓
Streamlit Deployment