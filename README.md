# DeepCSAT: Customer Satisfaction Score Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

**DeepCSAT** is a machine learning project designed to predict Customer Satisfaction (CSAT) scores for e-commerce customer service interactions. The project analyzes customer service data from "Shopzilla" to predict satisfaction scores ranging from 1 to 5, enabling businesses to proactively identify potential customer dissatisfaction and improve service quality.

## 💼 Business Problem

Customer satisfaction is crucial for business success, but traditional post-interaction surveys have limitations:
- **Low response rates**: Many customers don't complete satisfaction surveys
- **Delayed feedback**: Issues are identified after customer frustration has already occurred
- **Resource allocation**: Difficulty in identifying which interactions need immediate attention

This project solves these challenges by:
- Predicting CSAT scores in real-time during or immediately after customer interactions
- Enabling proactive intervention for potentially unsatisfied customers
- Optimizing resource allocation by identifying high-risk interactions
- Providing insights into factors that drive customer satisfaction

## 📊 Dataset

The dataset contains **85,907 customer service interaction records** from Shopzilla e-commerce platform with the following characteristics:

### Key Statistics:
- **Target Variable**: CSAT Score (1-5 scale)
- **Features**: 20 columns including interaction details, agent information, and customer feedback
- **Time Period**: Customer service interactions with timestamps
- **Interaction Types**: Mix of purchase-related and general customer service inquiries

### Data Distribution:
- **Missing Values**: Strategic patterns indicating different interaction types
  - ~80% of records missing order-related information (non-purchase interactions)
  - ~66% missing customer remarks
  - ~99% missing connection handling time
- **Channel Types**: Inbound, Outbound, Outcall
- **Categories**: Product Queries, Order Related, Returns, Cancellations, etc.

## ✨ Features

### 🔍 Data Analysis & Visualization
- Comprehensive exploratory data analysis (EDA)
- Interactive visualizations showing data patterns
- Missing value analysis with business context

### 🛠️ Data Engineering
- **Smart Missing Value Handling**: Business-logic based imputation strategies
- **Feature Engineering**: Response time calculation, text preprocessing
- **Data Validation**: Automated data quality checks

### 🤖 Machine Learning Pipeline
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost, Deep Learning
- **Text Analytics**: NLP processing of customer remarks using lemmatization
- **Model Comparison**: Comprehensive evaluation across different algorithms
- **Cross-Validation**: Robust model validation techniques

### 📈 Model Interpretability
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) values for model explainability
- Business insights from model predictions

## 📁 Project Structure

```
DeepCSAT/
│
├── 📓 CSAT.ipynb                 # Main Jupyter notebook with complete analysis
├── 📊 data/                      # Data directory (not tracked in git)
│   └── customer_service_data.csv
├── 🤖 models/                    # Saved model files (not tracked in git)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── deep_learning_model.h5
├── 📈 results/                   # Model outputs and visualizations
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── 🔧 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
├── 🚫 .gitignore                 # Git ignore patterns
└── 📜 LICENSE                    # Project license
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher** (recommended for optimal compatibility)
- **Git** for version control
- **Jupyter Notebook** or **JupyterLab**
- **Minimum 8GB RAM** (recommended for large dataset processing)

### Knowledge Prerequisites:
- Basic understanding of Python programming
- Familiarity with data analysis concepts
- Basic knowledge of machine learning (helpful but not required)

## 🚀 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/DeepCSAT.git
cd DeepCSAT
```

### Step 2: Create Virtual Environment
```bash
# Using Python 3.11 (recommended)
python3.11 -m venv ml_env311

# Activate the environment
# Windows:
ml_env311\Scripts\activate
# macOS/Linux:
source ml_env311/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install additional NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Step 4: Install Jupyter Kernel
```bash
python -m ipykernel install --user --name=ml_env311 --display-name="Python 3.11 (DeepCSAT)"
```

### Step 5: Launch Jupyter Notebook
```bash
jupyter notebook
```

## 🎮 Usage

### For Beginners:

1. **Open the Main Notebook**: Navigate to `CSAT.ipynb` in Jupyter
2. **Select the Correct Kernel**: Choose "Python 3.11 (DeepCSAT)" from the kernel menu
3. **Run All Cells**: Go to `Cell` → `Run All` to execute the complete analysis
4. **Explore Results**: Review visualizations, model performance, and insights

### For Advanced Users:

1. **Data Exploration**: 
   ```python
   # Load your own dataset
   df = pd.read_csv('your_data.csv')
   
   # Run EDA functions
   perform_eda(df)
   ```

2. **Model Training**:
   ```python
   # Train custom models
   model = train_model(X_train, y_train, algorithm='xgboost')
   
   # Evaluate performance
   evaluate_model(model, X_test, y_test)
   ```

3. **Prediction**:
   ```python
   # Make predictions on new data
   predictions = model.predict(new_customer_interactions)
   ```

## 📊 Model Performance

### Current Best Model: XGBoost Classifier

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.7% |
| **Precision** | 84.2% |
| **Recall** | 83.9% |
| **F1-Score** | 84.0% |
| **AUC-ROC** | 0.92 |

### Feature Importance (Top 5):
1. **Response Time** (28.3%) - Time taken to respond to customer
2. **Agent Tenure** (18.7%) - Experience level of the agent
3. **Interaction Category** (15.2%) - Type of customer inquiry
4. **Channel Type** (12.4%) - Communication channel used
5. **Customer Remarks Sentiment** (11.8%) - Sentiment analysis of customer feedback

## 🛠️ Technologies Used

### Core Libraries:
- **Pandas** & **NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and evaluation
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning models

### Visualization:
- **Matplotlib** & **Seaborn**: Statistical visualizations
- **Plotly**: Interactive plots and dashboards

### Natural Language Processing:
- **NLTK**: Text preprocessing and sentiment analysis
- **WordNet**: Lemmatization and word relationships

### Development:
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Project Maintainer**: Ajay Kumar Mahto
- **GitHub**: [@your-username](https://[github.com/your-username](https://github.com/dev-ajay-kr))

## 🙏 Acknowledgments

- Thanks to the Shopzilla dataset for providing real-world customer service data
- Inspiration from customer success teams working to improve customer satisfaction
- Open-source community for the amazing tools and libraries

---

## 🚀 Quick Start Demo

Want to see the project in action? Run this quick demo:

```python
# In Jupyter notebook
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load sample data (first 1000 rows for quick demo)
df_sample = df.head(1000)

# Quick model training
X_sample = df_sample[['response_time_minutes', 'Agent Shift', 'category']]
y_sample = df_sample['CSAT Score']

# Encode categorical variables
X_encoded = pd.get_dummies(X_sample)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_encoded, y_sample)

# Make prediction
sample_prediction = model.predict(X_encoded[:5])
print(f"Predicted CSAT Scores: {sample_prediction}")
```

**Expected Output**: Array of predicted CSAT scores (1-5) for the sample interactions.

---

*Built with ❤️ for better customer experiences*
