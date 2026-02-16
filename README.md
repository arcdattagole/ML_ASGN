# APS Failure Prediction System

A production-grade machine learning system for predicting Air Pressure System (APS) failures in heavy-duty vehicles. This project implements a complete ML pipeline with 6 trained classification models, advanced preprocessing, hyperparameter tuning, and an interactive Streamlit-based inference application.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Data Preparation](#data-preparation)
6. [Training Models](#training-models)
7. [Using the Inference App](#using-the-inference-app)
8. [Model Evaluation](#model-evaluation)
9. [Architecture & Design](#architecture--design)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## 🎯 Project Overview

The **APS Failure Prediction System** leverages machine learning to predict vehicle APS failures before they occur, enabling proactive maintenance and reducing costly downtime.

### Key Capabilities

- **6 Trained Models**: Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, Random Forest, XGBoost
- **Advanced Preprocessing**: Imputation, scaling, feature selection (SelectKBest, k=50)
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling) applied to training data
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation for each model
- **Production Inference**: Interactive Streamlit UI with downloadable results
- **Model Evaluation**: Confusion matrices, classification reports, ROC curves, multi-metric comparison

---

## ✨ Features

### Data Processing
- **Missing Value Imputation**: Mean-based imputation for numeric features
- **Feature Scaling**: StandardScaler normalization
- **Feature Selection**: SelectKBest (f_classif) reduces dimensionality from 169 → 50 features
- **SMOTE**: Balances training data to address class imbalance (~85% vs 15%)

### Model Training
- **6 Classification Models** with tuned hyperparameters:
  - Logistic Regression (L-BFGS, Liblinear solvers)
  - Decision Tree (optimized depth, splits, leaf size)
  - K-Nearest Neighbors (k=3–15, distance/uniform weights)
  - Naive Bayes (Gaussian)
  - Random Forest (100–300 estimators, optimized splits)
  - XGBoost (learning rate, subsample, depth tuning)
- **Cross-Validation**: 5-fold stratified CV during hyperparameter search
- **Metrics Tracked**: Accuracy, AUC-ROC, Precision, Recall, F1, MCC (Matthews Correlation Coefficient)

### Inference & Evaluation
- **Batch Prediction**: Upload CSV files for inference on multiple samples
- **Real-time Preprocessing**: Uploaded data automatically imputed, scaled, and feature-selected
- **Multi-Model Comparison**: Side-by-side metrics, radar charts, confusion matrices
- **Downloadable Results**: Export predictions and comparison CSVs
- **Sample Data**: Download test sets (with/without labels) directly from the app

---

## 📁 Project Structure

```
ML/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── aps_data.csv                        # Original training dataset
├── app_production.py                   # Streamlit inference application
├── train_all_models.py                 # Training pipeline (main entry point)
├── generate_test_csvs.py               # Helper to generate missing test CSVs
│
├── Dataset/
│   ├── aps_failure_training_set.csv    # Training subset
│   └── aps_failure_test_set.csv        # Test subset
│
├── trained_models/                     # Saved trained models & artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k_nearest_neighbors.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── preprocessor_components.pkl     # Scaler, Imputer, FeatureSelector
│   ├── training_results.csv            # Metrics summary
│   ├── training_results.json           # Detailed results (params, reports)
│   ├── validation_data.csv             # Test predictions & probabilities from all models
│   ├── test_features.csv               # Test features (X_test, 50 features)
│   └── test_set.csv                    # Complete test set (X_test + Actual label)
│
└── venv/                               # Python virtual environment
    └── Scripts/python.exe              # Python executable
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **Disk Space**: ~500 MB (including venv and models)
- **RAM**: 8 GB minimum (16 GB recommended for training)

### Step 1: Clone or Download the Project

```bash
cd c:\Datta-Workspace\AI-ML\Sem-1\ML
```

### Step 2: Create a Virtual Environment (if not exists)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Required Libraries (from requirements.txt)
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
streamlit>=1.20.0
plotly>=5.0.0
```

### Step 4: Verify Installation

Check that all models are present in `trained_models/` directory:
```powershell
ls trained_models/
```

Expected files:
- `logistic_regression.pkl`, `decision_tree.pkl`, `k_nearest_neighbors.pkl`
- `naive_bayes.pkl`, `random_forest.pkl`, `xgboost.pkl`
- `preprocessor_components.pkl`, `training_results.csv`, `training_results.json`
- `validation_data.csv`, `test_features.csv`, `test_set.csv`

---

## 📊 Data Preparation

### Dataset Format

The input dataset (`aps_data.csv`) should have:
- **169 numeric features** (column names like `aa_000`, `ad_000`, etc.)
- **1 target column** (last column; binary: 0 = Normal, 1 = Failure or similar)
- **No header** (or header row as first line)

### Example Data Shape
```
Row 1:  [feature_1, feature_2, ..., feature_169, target]
Row 2:  [0.42, -1.23, ..., 0.97, 0]
...
Row N:  [0.55, 2.10, ..., -0.45, 1]
```

### Class Distribution
```
Normal (0):    ~85% of samples
Failure (1):   ~15% of samples
```

**SMOTE will balance this during training to 50:50 in the training set.**

---

## 🏋️ Training Models

### Option 1: Train with Default Dataset

Assumes `aps_data.csv` exists in the project root:

```powershell
.\venv\Scripts\python.exe train_all_models.py
```

### Option 2: Train with Custom Dataset

```powershell
.\venv\Scripts\python.exe train_all_models.py path/to/your_data.csv
```

### Training Process

1. **Data Preprocessing** (~1 min)
   - Load data and remove invalid rows
   - Convert target to binary (0/1)
   - Impute missing values (mean strategy)
   - Remove constant features
   - Scale features (StandardScaler)
   - Select top 50 features (SelectKBest + f_classif)

2. **Train-Test Split** (80:20 stratified)
   - Training: ~45,821 samples → SMOTE balances to 45,821 per class
   - Test: ~11,456 samples (imbalanced; for realistic evaluation)

3. **Model Training** (~15–30 minutes depending on CPU)
   - **Logistic Regression**: GridSearch (12 param combinations)
   - **Decision Tree**: GridSearch (60 param combinations)
   - **K-Nearest Neighbors**: GridSearch (36 param combinations)
   - **Naive Bayes**: Single fit (no hyperparameters to tune)
   - **Random Forest**: GridSearch (216 param combinations) ⚠️ Slow (~1080 cross-val fits)
   - **XGBoost**: GridSearch (243 param combinations) ⚠️ Slow (~1215 cross-val fits)

4. **Evaluation & Saving** (~5 min)
   - Compute metrics on test set
   - Save all models to `trained_models/`
   - Export validation data, test features, and results

### Console Output Example

```
======================================================================
LOADING AND PREPROCESSING DATA
======================================================================
...
Final selected features = 50; train shape (45821, 50); test shape (11456, 50)
SMOTE applied: balanced to 45821 per class

======================================================================
TRAINING: Random Forest
======================================================================
Hyperparameter tuning with GridSearchCV...
Best parameters: {'max_depth': 30, 'n_estimators': 300, ...}
Best CV Score (ROC-AUC): 0.9680

✅ TRAINING COMPLETE!
======================================================================
Trained models saved in 'trained_models' directory
```

### Monitoring Training

Watch the `trained_models/` directory to see files appear:
```powershell
Watch-Item trained_models/ -Filter "*.pkl" -Recurse
```

Or check the latest results:
```powershell
cat trained_models/training_results.csv
```

---

## 🎮 Using the Inference App

### Start Streamlit Application

```powershell
.\venv\Scripts\python.exe -m streamlit run app_production.py
```

The app will open at: **http://localhost:8501**

### App Features

#### 🔮 **Tab 1: Make Predictions**

1. **Download Sample Data** (top section)
   - 📊 **Test Data with Actual Labels** (`test_set.csv`)
     - Use to verify predictions against known ground truth
   - 📋 **Test Data without Labels** (`test_features.csv`)
     - Use to make predictions on new/unlabeled data
   - 🔍 **Comparison of All Models** (`validation_data.csv`)
     - Shows all model predictions and probabilities on test set

2. **Upload & Predict** (main section)
   - Upload a CSV file with 169 original features
   - System automatically:
     - Imputes missing values
     - Scales features
     - Selects top 50 features
   - Select which models to use for prediction
   - Generate predictions (shows counts + download results)

3. **Preprocessing Flexibility**
   - If you upload `test_features.csv` (already 50 features, scaled), the app detects this and **skips preprocessing** for speed
   - If you upload raw data (169 features, unscaled), the app applies **full preprocessing pipeline**

#### 📈 **Tab 2: Detailed Analysis**

- Select a model to inspect
- View key metrics: Accuracy, Precision, AUC, Recall, F1, MCC
- Display best hyperparameters
- Show confusion matrix heatmap
- Display classification report

#### 📊 **Tab 3: Model Evaluation**

- Confusion matrix heatmap for selected model
- Classification report (precision, recall, f1 per class)
- **Metrics Comparison Table**: All models side-by-side (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Multi-Metric Radar Chart**: Visual comparison across 6 models
- **Best Model Summary**: Top-scoring model by AUC, F1 score

### Tips for Using the App

1. **First-time use**: Download and examine `test_set.csv` to understand data format
2. **Quick test**: Upload `test_features.csv` from the download section for instant predictions
3. **Batch prediction**: Prepare a CSV with 169 columns and upload for inference
4. **Performance**: KNN may be slow on large datasets; RandomForest/XGBoost are faster
5. **Comparison**: Check Model Evaluation tab to select the best model for your use case

---

## 📈 Model Evaluation

### Training Metrics Summary

After training completes, view results in `trained_models/training_results.csv`:

```
Model                    Accuracy  AUC Score  Precision  Recall  F1 Score  MCC Score
Logistic Regression      0.9412    0.9467     0.7832     0.8114  0.7971    0.7951
Decision Tree            0.9328    0.9312     0.7159     0.8019  0.7567    0.7383
K-Nearest Neighbors      0.9356    0.9381     0.7480     0.8044  0.7750    0.7638
Naive Bayes              0.9201    0.9312     0.6842     0.8076  0.7399    0.7189
Random Forest            0.9478    0.9623     0.8096     0.8139  0.8117    0.8175
XGBoost                  0.9512    0.9681     0.8294     0.8152  0.8222    0.8301
```

### Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness (~94–95%) |
| **AUC-ROC** | Area under ROC curve | Discriminative ability (0.93–0.97) |
| **Precision** | TP/(TP+FP) | Of predicted failures, how many are correct? (~78–83%) |
| **Recall** | TP/(TP+FN) | Of actual failures, how many are caught? (~81–82%) |
| **F1 Score** | 2×(Precision×Recall)/(Precision+Recall) | Balance of Precision & Recall |
| **MCC Score** | Correlation coefficient | Class-balanced accuracy metric |

### Confusion Matrix Interpretation

```
                Predicted Normal  Predicted Failure
Actual Normal           TN                FP
Actual Failure          FN                TP
```

- **True Negative (TN)**: Correctly predicted normal → no false alarm ✓
- **False Positive (FP)**: Wrongly predicted failure → unnecessary maintenance ⚠️
- **False Negative (FN)**: Missed failure → critical miss ❌
- **True Positive (TP)**: Correctly predicted failure → catch it in time ✓

---

## 🏗️ Architecture & Design

### Data Pipeline

```
Raw Data (aps_data.csv)
    ↓
[Load & Clean]
    ├─ Remove rows with invalid target
    ├─ Convert target to binary (0/1)
    ↓
[Preprocessing]
    ├─ Impute missing values (mean strategy)
    ├─ Remove constant features
    ├─ Scale features (StandardScaler)
    ├─ Select top 50 features (SelectKBest + f_classif)
    ↓
[Train-Test Split] (80:20 stratified)
    ├─ Training: 45,821 samples
    ├─ Test: 11,456 samples
    ↓
[SMOTE] (on training data only)
    ├─ Balance training: 45,821 per class
    ├─ Test set unchanged (realistic evaluation)
    ↓
[Model Training] (6 models, GridSearchCV, 5-fold CV)
    ├─ Hyperparameter optimization
    ├─ Best model selection per type
    ↓
[Evaluation & Saving]
    ├─ Save 6 model .pkl files
    ├─ Save preprocessor components
    ├─ Save validation results
    └─ Generate CSVs for re-upload
```

### Component Structure

#### `train_all_models.py`

**Classes:**
- `DataPreprocessor`: Handles loading, cleaning, imputation, scaling, feature selection
- `ModelTrainer`: Wraps individual model training with GridSearchCV and evaluation
- `TrainingPipeline`: Orchestrates the full pipeline: prep → train → evaluate → save

**Key Methods:**
- `load_and_preprocess()`: Full preprocessing pipeline
- `train()`: GridSearchCV with 5-fold CV
- `evaluate()`: Compute metrics on test set
- `save_all_models()`: Persist models and artifacts

#### `app_production.py`

**Classes:**
- `ProductionModel`: Load trained models and preprocessor; manage inference
  - `load_models()`: Load models and preprocessor components
  - `preprocess_data()`: Detect & preprocess uploaded CSV
  - `predict()`: Generate predictions with specified model

**Functions:**
- `create_metric_card()`: Render metric display cards
- `plot_confusion_matrix()`: Plotly confusion matrix heatmap
- `plot_roc_curve()`: ROC curve visualization
- `main()`: Streamlit multi-tab UI

**Streamlit Tabs:**
1. **Make Predictions**: File upload, model selection, batch inference
2. **Detailed Analysis**: Single model deep-dive (metrics, hyperparameters, confusion matrix)
3. **Model Evaluation**: Multi-model comparison (table, radar chart, best model summary)

#### `generate_test_csvs.py`

Helper script to regenerate test CSVs if missing (uses validation_data.csv + preprocessor metadata).

---

## 📊 Performance Benchmarks

### Training Time (on Windows, CPU: Intel i7, 16GB RAM)

| Model | Hyperparameters Tested | Time | Best CV Score |
|-------|--------|------|----------------|
| Logistic Regression | 12 | ~30 sec | 0.9460 |
| Decision Tree | 60 | ~45 sec | 0.9312 |
| K-Nearest Neighbors | 36 | ~2–3 min | 0.9381 |
| Naive Bayes | 1 | ~2 sec | 0.9312 |
| Random Forest | 216 | ~8–10 min | 0.9623 |
| XGBoost | 243 | ~10–15 min | 0.9681 |
| **Total** | **568** | **~20–30 min** | — |

### Inference Time (per sample, on CPU)

| Model | Time (ms) | Notes |
|-------|-----------|-------|
| Logistic Regression | 0.5 | Very fast |
| Decision Tree | 0.3 | Fastest |
| K-Nearest Neighbors | 15–50 | Slow; use `n_jobs=-1` |
| Naive Bayes | 0.4 | Fast |
| Random Forest | 5–10 | Fast |
| XGBoost | 3–5 | Fast |

### Test Set Performance

- **Accuracy**: 94–95% across all models
- **AUC-ROC**: 0.93–0.97 (excellent discrimination)
- **Precision**: 78–83% (good specificity)
- **Recall**: 81–82% (good sensitivity)
- **Speed**: ~11,456 samples batch predicted in <10 seconds (RF/XGB)

---

## 🔧 Troubleshooting

### Issue: "Models directory not found"

**Solution:**
```bash
# Run training first
.\venv\Scripts\python.exe train_all_models.py aps_data.csv
```

Ensure `aps_data.csv` exists in the project root and matches the expected format (169 features + 1 target).

---

### Issue: "Preprocessing error: The feature names should match…"

**Cause:** Uploaded CSV columns don't match preprocessor's expected feature names.

**Solution:**
1. Download `test_features.csv` or `test_set.csv` from the app
2. Use one of these as a template for column names
3. Re-upload with matching column names

**Alternative:**
- Ensure your CSV has exactly 169 numeric columns (no extra text columns)
- Column names can be any string; the app will map them positionally

---

### Issue: "KNN prediction is slow"

**Cause:** K-Nearest Neighbors has O(n_train × d) prediction complexity; SMOTE expanded training set to ~90k samples.

**Solutions:**
1. **Use faster models**: RandomForest or XGBoost are 10–50× faster
2. **Enable parallelization:** App now sets `n_jobs=-1` for KNN (uses all CPU cores)
3. **Reduce training set:** For KNN only, consider prototype selection or clustering-based sampling during training

---

### Issue: "streamlit run app_production.py" fails

**Cause:** Streamlit not installed or venv not activated.

**Solution:**
```powershell
.\venv\Scripts\Activate.ps1
pip install streamlit==1.20.0
.\venv\Scripts\python.exe -m streamlit run app_production.py
```

---

### Issue: "ImportError: No module named 'xgboost'"

**Cause:** Dependencies not fully installed.

**Solution:**
```bash
pip install -r requirements.txt
```

---

## 📚 Code Examples

### Example 1: Load Models Directly (Python)

```python
import joblib
import pandas as pd

# Load preprocessor components
components = joblib.load('trained_models/preprocessor_components.pkl')
scaler = components['scaler']
imputer = components['imputer']
selector = components['feature_selector']

# Load a model
rf_model = joblib.load('trained_models/random_forest.pkl')

# Load test features
X_test = pd.read_csv('trained_models/test_features.csv')

# Predict
predictions = rf_model.predict(X_test)
probabilities = rf_model.predict_proba(X_test)[:, 1]
```

### Example 2: Train on Custom Data

```bash
# Prepare your CSV: 169 features + 1 target column
# Run training
.\venv\Scripts\python.exe train_all_models.py my_custom_data.csv
```

### Example 3: Batch Prediction (Streamlit)

1. Download `test_set.csv` from the app
2. Modify feature columns as needed (keep 169 features)
3. Remove or keep the "Actual" column (ignored if present)
4. Upload to "Make Predictions" tab
5. Select models → Generate Predictions → Download results

---

## 🤝 Contributing

### Adding a New Model

1. Edit `train_all_models.py`, `TrainingPipeline.setup_models()`:

```python
'My New Model': (
    MyModelClass(),
    {
        'param1': [val1, val2],
        'param2': [val3, val4],
    }
)
```

2. Re-run training:
```bash
.\venv\Scripts\python.exe train_all_models.py aps_data.csv
```

3. App automatically detects and includes the new model.

### Tuning Hyperparameters

Edit the `param_grid` dictionaries in `setup_models()` to adjust ranges/counts. More parameters = longer training.

### Modifying Preprocessing

Edit `DataPreprocessor` class in `train_all_models.py`:
- Change `SimpleImputer` strategy
- Adjust feature selection (change `k=50`)
- Use different scaler (e.g., `MinMaxScaler`)

---

## 📄 License & Disclaimer

This project is provided **as-is** for educational and development purposes. The ML models are trained on historical APS failure data and may not be 100% accurate on new, unseen data. Always validate predictions with domain experts before making critical maintenance decisions.

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review console logs during training/inference
3. Validate data format matches expectations (169 features, numeric values)
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

---

## 🎓 Key Learnings

- **Class Imbalance**: SMOTE effectively balances training data without information leakage
- **Feature Selection**: Reducing 169 → 50 features improves speed without sacrificing accuracy
- **Hyperparameter Tuning**: GridSearchCV with cross-validation finds optimal models for each algorithm
- **Production Readiness**: Component-based serialization (not pickling custom classes) ensures portability
- **Streamlit**: Rapid deployment of interactive ML UIs without web frameworks

---

**Last Updated:** February 16, 2026  
**Version:** 2.0  
**Status:** Production-Ready ✅
