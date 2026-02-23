# 🏭 Predictive Maintenance for Factory Equipment

## 📌 Overview

This project implements an AI-driven Predictive Maintenance system for industrial equipment using machine learning and deep learning models.

The goal is to predict machine failure types and estimate remaining useful life (RUL) using sensor data, enabling proactive maintenance scheduling and minimizing unplanned downtime in manufacturing environments.

This project simulates an Industry 4.0 smart manufacturing use case.

---

## 🎯 Problem Statement

In industrial environments, unexpected equipment failures lead to:

- Production downtime
- Increased maintenance costs
- Supply chain disruption
- Reduced operational efficiency

Traditional maintenance approaches (reactive or scheduled) are inefficient.

This project applies Machine Learning techniques to predict failures before they occur using historical sensor data.

---

## 📊 Dataset

Source: Kaggle – Machine Predictive Maintenance Classification Dataset  
Type: Structured industrial sensor data  
Target Variable: Failure Type

### Features include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Quality Type (L, M, H)
- Failure Type (Label)

---

## 🧠 Solution Approach

The system follows a standard machine learning pipeline:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning & Encoding
4. Feature Engineering
5. Data Normalization
6. Train-Test Split (80/20)
7. Model Training
8. Model Evaluation
9. Failure Prediction & Life Estimation

---

## 🤖 Models Implemented

- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Convolutional Neural Network (CNN)

### 🏆 Best Performance

CNN achieved 97% classification accuracy.

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

Models were benchmarked to determine the most reliable failure prediction approach.

---

## 🔧 Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

### Development Environment
- Jupyter Notebook

---

## 🏗 Project Structure

Predictive-Maintenance/
│
├── dataset/
│   └── predictive_maintenance.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── performance_comparison.png
│
└── README.md

---

## 🔍 Key Features

- Multi-model benchmarking for industrial reliability analysis
- End-to-end ML pipeline implementation
- Failure type classification
- Remaining machine life percentage estimation
- Data visualization for industrial trend analysis

---

## ▶️ How to Run the Project

### 1. Clone the Repository

git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run the Notebook

jupyter notebook

Open and execute:
model_training.ipynb

---

## 📊 Example Output

Input: Sensor data values  
Output:
- Predicted Failure Type
- Remaining Machine Life (%)
- Maintenance Recommendation

If remaining life is below threshold → Maintenance should be scheduled.

---

## 🚀 Industry Relevance

This project demonstrates practical application of:

- Industry 4.0 concepts
- Smart manufacturing systems
- AI-driven reliability engineering
- Data-driven maintenance optimization

It simulates how predictive analytics can reduce downtime and improve operational efficiency in industrial environments.

---

## 🔮 Future Improvements

- Real-time IoT sensor integration
- Deployment as REST API service
- Docker-based containerization
- Cloud deployment (Azure)
- Remaining Useful Life regression modeling
- Integration with monitoring dashboard

---

