# Insurance Claim Prediction - MLOps Project

## 📌 Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting whether an insurance policy is likely to result in a claim.

The project integrates machine learning, feature engineering, MLflow experiment tracking, model registry, FastAPI deployment, Docker containerization, and a Streamlit user interface.

## 🎯 Objective

The main objective is to develop and deploy an insurance claim prediction system that can:

- Process customer and vehicle information
- Perform feature engineering and preprocessing
- Predict insurance claim probability
- Track experiments using MLflow
- Register and version the trained model
- Expose predictions through a REST API
- Containerize the application using Docker
- Provide an interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- MLflow
- FastAPI
- Uvicorn
- Streamlit
- Docker
- Git & GitHub
- Jupyter Notebook

## 🔄 MLOps Architecture

Dataset
↓
Data Cleaning & Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
MLflow Experiment Tracking
↓
MLflow Model Registry
↓
Saved Model Artifact
↓
Docker Container
↓
FastAPI REST API
↓
Streamlit UI
↓
Insurance Claim Prediction

## 🤖 Machine Learning

The project evaluates multiple classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The dataset contains an imbalanced target variable, so techniques such as SMOTE and class weighting were evaluated.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## 📊 MLflow

MLflow is used for:

- Experiment tracking
- Metric logging
- Model logging
- Model registration
- Model versioning

Registered model:

`InsuranceClaimPrediction`

Version:

`1`

## 🚀 FastAPI

The trained model is exposed through a REST API.

### Health Check

```text
GET /health