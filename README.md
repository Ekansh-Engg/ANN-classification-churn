# Customer Churn Prediction using ANN

A binary classification web app that predicts whether a bank customer is likely to churn (leave the bank), built with an Artificial Neural Network (TensorFlow/Keras) and deployed with Streamlit.

**Live demo:** https://ann-classification-churn-kydggkewefu2rppckvvzyx.streamlit.app/

## Overview

Banks lose significant revenue when customers close their accounts. This project trains a feedforward neural network on historical customer data to predict churn probability, then serves the model through an interactive Streamlit UI where you can plug in customer details and get an instant prediction.

## Tech Stack

- **Model:** TensorFlow / Keras (Sequential ANN)
- **Preprocessing:** scikit-learn (`StandardScaler`, `LabelEncoder`, `OneHotEncoder`)
- **App / Deployment:** Streamlit
- **Data:** [`Churn_Modelling.csv`](Churn_Modelling.csv) (10,000 bank customer records)

## Project Structure

```
ANN-classification-churn/
├── app.py                      # Streamlit web app
├── experiments.ipynb           # Data preprocessing + model training
├── prediction.ipynb            # Standalone inference notebook
├── Churn_Modelling.csv         # Training dataset
├── model.h5                    # Trained ANN weights
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography
├── requirements.txt
└── runtime.txt
```

## How It Works

1. **Data preprocessing** (`experiments.ipynb`)
   - Drops non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
   - Label-encodes `Gender` (binary)
   - One-hot encodes `Geography` (France / Germany / Spain)
   - Scales all features with `StandardScaler`
   - 80/20 train-test split

2. **Model architecture**

   | Layer | Units | Activation |
   |-------|-------|------------|
   | Input → Hidden 1 | 64 | ReLU |
   | Hidden 2 | 32 | ReLU |
   | Output | 1 | Sigmoid |

   - Optimizer: Adam (lr = 0.01)
   - Loss: Binary Crossentropy
   - Callbacks: `EarlyStopping` (patience=10, restores best weights) + `TensorBoard` logging

3. **Inference** (`app.py`)
   - Loads the saved model and the three fitted preprocessing objects (`scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`)
   - Collects customer details via Streamlit widgets (geography, gender, age, balance, credit score, tenure, etc.)
   - Applies the same encoding/scaling pipeline used in training
   - Outputs a churn probability and a likely/not-likely verdict (threshold: 0.5)

## Input Features

| Feature | Description |
|---|---|
| Credit Score | Customer's credit score |
| Geography | France / Germany / Spain |
| Gender | Male / Female |
| Age | 18–92 |
| Tenure | Years as a bank customer (0–10) |
| Balance | Account balance |
| Number of Products | Bank products used (1–4) |
| Has Credit Card | Yes/No |
| Is Active Member | Yes/No |
| Estimated Salary | Annual estimated salary |

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Ekansh-Engg/ANN-classification-churn.git
cd ANN-classification-churn

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Requirements

```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
```

Python version: 3.11 (see `runtime.txt`)

## Future Improvements

- Add model evaluation metrics (accuracy, precision/recall, ROC-AUC) to the README
- Hyperparameter tuning (units, learning rate, batch size) with `keras-tuner`
- Add SHAP/feature-importance explanations to the app for prediction interpretability
- Containerize with Docker for reproducible deployment

