# Customer Churn Prediction & Salary Estimation using ANN
 
Two web apps built on the same bank customer dataset and sharing the same feature-engineering pipeline, powered by Artificial Neural Networks (TensorFlow/Keras) and deployed with Streamlit:
 
1. **Churn Classifier** — predicts whether a customer is likely to leave the bank (binary classification)
**Live demo (churn classifier):**https://ann-classification-churn-kydggkewefu2rppckvvzyx.streamlit.app/
2. **Salary Regressor** — predicts a customer's estimated salary (regression)
**Live demo (Salary Regressor):** https://ann-classification-churn-gqrmgrgrpj9hhfbvmbpv6c.streamlit.app/
 
## Overview
 
Banks lose significant revenue when customers close their accounts, and understanding customer profiles (like expected salary) helps with segmentation and product targeting. This project trains two feedforward neural networks — one classifier, one regressor — on the same historical customer data, then serves each through its own interactive Streamlit UI.
 
## Tech Stack
 
- **Model:** TensorFlow / Keras (Sequential ANN)
- **Preprocessing:** scikit-learn (`StandardScaler`, `LabelEncoder`, `OneHotEncoder`)
- **App / Deployment:** Streamlit
- **Data:** [`Churn_Modelling.csv`](Churn_Modelling.csv) (10,000 bank customer records)
## Project Structure
 
```
ANN-classification-churn/
├── app.py                      # Streamlit app — churn classifier
├── salaryregression_app.py     # Streamlit app — salary regressor
├── experiments.ipynb           # Data preprocessing + churn model training
├── salaryregression.ipynb      # Salary regression model training
├── prediction.ipynb            # Standalone inference notebook
├── Churn_Modelling.csv         # Training dataset
├── model.h5                    # Trained churn classification ANN
├── regressionmodel.h5          # Trained salary regression ANN
├── scaler.pkl                  # StandardScaler fit for churn classification (features incl. EstimatedSalary)
├── scaler1.pkl                 # StandardScaler fit for salary regression (features incl. Exited)
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender (shared by both apps)
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography (shared by both apps)
├── requirements.txt
└── runtime.txt
```
 
> **Note on the two scalers:** the classifier's target is `Exited`, so `scaler.pkl` was fit on all other columns including `EstimatedSalary`. The regressor's target is `EstimatedSalary`, so `scaler1.pkl` was fit on all other columns including `Exited`. Each app must load its own matching scaler — mixing them up causes a `feature names should match those passed during fit` error at inference time.
 
## How It Works
 
1. **Data preprocessing** (`experiments.ipynb` / `salaryregression.ipynb`)
   - Drops non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
   - Label-encodes `Gender` (binary)
   - One-hot encodes `Geography` (France / Germany / Spain)
   - Scales all features with `StandardScaler` (a separate scaler per model, see note above)
   - 80/20 train-test split
2. **Model architecture** (both models)
   | Layer | Units | Activation |
   |-------|-------|------------|
   | Input → Hidden 1 | 64 | ReLU |
   | Hidden 2 | 32 | ReLU |
   | Output | 1 | Sigmoid (classifier) / Linear (regressor) |
   - Optimizer: Adam
   - Loss: Binary Crossentropy (classifier) / Mean Absolute Error (regressor)
   - Callbacks: `EarlyStopping` (restores best weights) + `TensorBoard` logging
3. **Inference**
   - `app.py` loads `model.h5` + `scaler.pkl` and outputs a churn probability with a likely/not-likely verdict (threshold: 0.5)
   - `salaryregression_app.py` loads `regressionmodel.h5` + `scaler1.pkl` and outputs an estimated salary
   - Both load the shared `label_encoder_gender.pkl` and `onehot_encoder_geo.pkl`, and apply the same encoding/scaling pipeline used in training
## Input Features
 
| Feature | Description | Used by |
|---|---|---|
| Credit Score | Customer's credit score | Both |
| Geography | France / Germany / Spain | Both |
| Gender | Male / Female | Both |
| Age | 18–92 | Both |
| Tenure | Years as a bank customer (0–10) | Both |
| Balance | Account balance | Both |
| Number of Products | Bank products used (1–4) | Both |
| Has Credit Card | Yes/No | Both |
| Is Active Member | Yes/No | Both |
| Estimated Salary | Annual estimated salary | Classifier only (input) |
| Exited | Whether the customer churned | Regressor only (input) |
 
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
 
# Run the churn classifier app
streamlit run app.py
 
# ...or run the salary regressor app
streamlit run salaryregression_app.py
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
 
- Add model evaluation metrics (accuracy, precision/recall, ROC-AUC for the classifier; MAE/RMSE for the regressor) to the README
- Hyperparameter tuning (units, learning rate, batch size) with `keras-tuner`
- Add SHAP/feature-importance explanations to both apps for prediction interpretability
- Rename `scaler.pkl` / `scaler1.pkl` to self-documenting names (e.g. `churn_scaler.pkl` / `salary_scaler.pkl`) to prevent future mix-ups
- Containerize with Docker for reproducible deployment
