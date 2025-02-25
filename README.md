# NDHU Data Mining 2021 - Heart Disease Prediction Model  

📌 **Heart Disease Prediction using Machine Learning**  

This repository contains a **heart disease prediction model** built for the **NDHU Data Mining course (2021)**. The model predicts the probability of heart disease based on 11 patient features from the **Heart Disease Dataset**. The goal is to assist doctors in making diagnostic decisions.

## 📊 Dataset & Preprocessing  
### 📌 Dataset: **Heart Disease Dataset**  
The dataset contains **918 samples** with **11 features** and a target variable indicating heart disease presence.

### 🔄 Data Preprocessing  
1. **Handle missing values** – Check and fill any missing data.  
2. **Load Data** – Read the `918 × 12` dataset and set the last column as the target variable `Y`.  
3. **Feature Encoding**  
   - **Ordinal categorical data** → Mapped to numeric values.  
   - **Nominal categorical data** → One-hot encoding.  
   - **Unordered numerical data** → One-hot encoding.  
4. **Feature Scaling** – Normalize all features to the range `[0,1]`.  
5. **Dimensionality Reduction** – Apply **PCA** to reduce feature dimensions.  
6. **Store Processed Data** – Features saved in `X`, target label saved in `Y`.  

## 🏗️ Model Building  
We trained and evaluated **six machine learning models** along with an ensemble **Voting Classifier**:

### 🔍 Models Used  
- **Decision Tree** 🌳  
- **K-Nearest Neighbors (KNN)** 📌  
- **Naive Bayes (Gaussian)** 📊  
- **Support Vector Machine (SVM)** 🛠️  
- **Random Forest** 🌲  
- **Linear Regression** 📈  
- **Voting Classifier** 🗳️ (Combining multiple models for better accuracy)

### ⚙️ Hyperparameter Optimization  
- We use `ParameterGrid` to systematically test and optimize hyperparameters for each model.  
- The model with the best performance is selected for final evaluation.  

