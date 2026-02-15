"""
Complete Training Pipeline with SMOTE and Hyperparameter Tuning
Trains all 6 classification models and saves them for production use
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoder = None
        self.feature_selector = None
        self.selected_columns = None
        self.imputed_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.smote = None
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess the dataset"""
        print(f"\n{'='*70}")
        print(f"LOADING AND PREPROCESSING DATA")
        print(f"{'='*70}")
        print(f"Loading dataset from {file_path}...")
        
        # Load data
        df = pd.read_csv(file_path, na_values=['NA', 'na', 'N.A', 'N.A.', '', ' '])
        print(f"Original dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Separate features and target
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()
        
        print(f"Target column name: {df.columns[-1]}")
        
        # Step 0: Clean target variable
        print("\n--- Step 0: Cleaning target variable ---")
        target_missing_before = y.isnull().sum()
        print(f"Missing values in target: {target_missing_before}")
        
        y_numeric = pd.to_numeric(y, errors='coerce')
        valid_mask = ~y_numeric.isnull()
        print(f"Rows with valid target: {valid_mask.sum()}/{len(valid_mask)}")
        
        X = X[valid_mask].reset_index(drop=True)
        y = y_numeric[valid_mask].reset_index(drop=True)
        
        print(f"Shape after removing rows with NaN target: {X.shape}")
        
        # Convert to binary classification: 0 (Normal) vs 1 (Failure)
        print("\n--- Converting to Binary Classification ---")
        print(f"Unique target values: {sorted(y.unique())}")
        y_binary = (y != 0).astype(int)
        
        print(f"Binary class distribution:")
        print(f"  - Class 0 (Normal): {(y_binary == 0).sum()}")
        print(f"  - Class 1 (Failure): {(y_binary == 1).sum()}")
        
        y = np.array(y_binary)
        
        # Step 1: Convert features to numeric
        print("\n--- Step 1: Converting to numeric ---")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        print(f"Shape: {X.shape}")
        
        # Step 2: Drop non-numeric columns
        print("\n--- Step 2: Dropping non-numeric columns ---")
        X = X.select_dtypes(include=[np.number])
        print(f"Shape: {X.shape}")
        
        # Step 2b: Drop all-NaN columns
        print("\n--- Step 2b: Dropping all-NaN columns ---")
        all_nan_cols = X.columns[X.isnull().all()].tolist()
        if all_nan_cols:
            print(f"Dropping {len(all_nan_cols)} all-NaN columns")
            X = X.drop(columns=all_nan_cols)
        print(f"Shape: {X.shape}")
        
        # Step 3: Impute missing values
        print("\n--- Step 3: Imputing missing values ---")
        col_names_before_impute = X.columns.tolist()
        
        X_imputed_array = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed_array, columns=col_names_before_impute)
        print(f"Shape: {X.shape}")
        
        # Step 4: Remove constant features
        print("\n--- Step 4: Removing constant features ---")
        nunique = X.nunique()
        constant_cols = nunique[nunique == 1].index.tolist()
        non_constant_cols = nunique[nunique > 1].index.tolist()
        
        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns")
            X = X[non_constant_cols]
        print(f"Shape: {X.shape}")
        
        # Step 5: Scale features
        print("\n--- Step 5: Scaling features ---")
        col_names_before_scale = X.columns.tolist()
        
        X_scaled_array = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled_array, columns=col_names_before_scale)
        print(f"Shape: {X.shape}")
        
        # Step 6: Feature selection
        print("\n--- Step 6: Feature selection (top 50) ---")
        n_features_to_select = min(50, X.shape[1])
        
        col_names_before_selection = X.columns.tolist()
        self.feature_selector = SelectKBest(f_classif, k=n_features_to_select)
        
        X_selected_array = self.feature_selector.fit_transform(X, y)
        
        selected_mask = self.feature_selector.get_support()
        selected_col_names = [col_names_before_selection[i] for i, selected in enumerate(selected_mask) if selected]
        self.selected_columns = selected_col_names
        self.imputed_columns = col_names_before_scale
        
        X = pd.DataFrame(X_selected_array, columns=selected_col_names)
        print(f"Shape: {X.shape}")
        
        # Step 7: Train-test split
        print("\n--- Step 7: Train-test split (80-20) ---")
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train set shape: {X_train_temp.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Step 8: Apply SMOTE to training data only
        print("\n--- Step 8: Applying SMOTE to training data ---")
        print(f"Before SMOTE - Class 0: {(y_train_temp == 0).sum()}, Class 1: {(y_train_temp == 1).sum()}")
        
        self.smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_train, y_train = self.smote.fit_resample(X_train_temp, y_train_temp)
        
        print(f"After SMOTE - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
        print(f"SMOTE train set shape: {X_train.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n✅ Preprocessing completed!")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_new_data(self, X):
        """Preprocess new data using fitted transformers"""
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Keep only imputed columns that exist
        existing_cols = [col for col in self.imputed_columns if col in X.columns]
        X = X[existing_cols]
        
        # Impute
        X_imputed = self.imputer.transform(X)
        X = pd.DataFrame(X_imputed, columns=existing_cols)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=existing_cols)
        
        # Select features
        X_selected = self.feature_selector.transform(X)
        X = pd.DataFrame(X_selected, columns=self.selected_columns)
        
        return X


class ModelTrainer:
    """Train individual models with hyperparameter tuning"""
    
    def __init__(self, model_name, model, param_grid, random_state=42):
        self.model_name = model_name
        self.base_model = model
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.cv_scores = None
        
    def train(self, X_train, y_train):
        """Train model with hyperparameter tuning"""
        print(f"\n{'='*70}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'='*70}")
        
        # GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(
            self.base_model,
            self.param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Hyperparameter tuning with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.cv_results_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV Score (ROC-AUC): {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'mcc_score': matthews_corrcoef(y_test, y_pred)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return metrics, cm, report, y_pred, y_pred_proba
    
    def save_model(self, file_path):
        """Save trained model"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.best_model, file_path)
        print(f"✅ Model saved: {file_path}")


class TrainingPipeline:
    """Complete training pipeline for all models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}
        self.results_df = None
        self.X_test = None
        self.y_test = None
        
    def setup_models(self):
        """Setup all models with hyperparameter grids"""
        print(f"\n{'='*70}")
        print(f"SETTING UP MODELS WITH HYPERPARAMETER GRIDS")
        print(f"{'='*70}")
        
        model_configs = {
            'Logistic Regression': (
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear'],
                }
            ),
            'Decision Tree': (
                DecisionTreeClassifier(random_state=self.random_state),
                {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            ),
            'K-Nearest Neighbors': (
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree']
                }
            ),
            'Naive Bayes': (
                GaussianNB(),
                {}
            ),
            'Random Forest': (
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            ),
            'XGBoost': (
                XGBClassifier(random_state=self.random_state, verbosity=0),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            )
        }
        
        for name, (model, params) in model_configs.items():
            self.models[name] = ModelTrainer(name, model, params, self.random_state)
            print(f"✅ {name} configured")
    
    def run_training(self, file_path):
        """Run complete training pipeline"""
        print(f"\n{'#'*70}")
        print(f"# COMPLETE TRAINING PIPELINE")
        print(f"{'#'*70}")
        
        # Step 1: Preprocess data
        self.preprocessor = DataPreprocessor(test_size=0.2, random_state=self.random_state)
        X_train, X_test, y_train, y_test = self.preprocessor.load_and_preprocess(file_path)
        
        # Store validation/test data for later saving
        self.X_test = X_test
        self.y_test = y_test
        
        # Step 2: Setup models
        self.setup_models()
        
        # Step 3: Train all models
        print(f"\n{'#'*70}")
        print(f"# TRAINING ALL MODELS")
        print(f"{'#'*70}")
        
        for model_name, trainer in self.models.items():
            trainer.train(X_train, y_train)
        
        # Step 4: Evaluate all models
        print(f"\n{'#'*70}")
        print(f"# EVALUATING ALL MODELS")
        print(f"{'#'*70}")
        
        for model_name, trainer in self.models.items():
            print(f"\nEvaluating {model_name}...")
            metrics, cm, report, y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)
            
            self.results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'best_params': trainer.best_params
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC Score: {metrics['auc_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  MCC Score: {metrics['mcc_score']:.4f}")
        
        # Step 5: Create results dataframe
        print(f"\n{'#'*70}")
        print(f"# RESULTS SUMMARY")
        print(f"{'#'*70}")
        
        results_list = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            results_list.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'AUC Score': metrics['auc_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'MCC Score': metrics['mcc_score']
            })
        
        self.results_df = pd.DataFrame(results_list)
        print("\n" + self.results_df.to_string(index=False))
        
        return self.results_df
    
    def save_all_models(self, models_dir='trained_models'):
        """Save all trained models"""
        print(f"\n{'='*70}")
        print(f"SAVING TRAINED MODELS")
        print(f"{'='*70}")
        
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, trainer in self.models.items():
            file_name = f"{model_name.lower().replace(' ', '_').replace('-', '_')}.pkl"
            file_path = os.path.join(models_dir, file_name)
            trainer.save_model(file_path)
        
        # Save preprocessor transformers separately (not the whole object)
        # This avoids pickle class reference issues
        preprocessor_components = {
            'scaler': self.preprocessor.scaler,
            'imputer': self.preprocessor.imputer,
            'feature_selector': self.preprocessor.feature_selector,
            'selected_columns': self.preprocessor.selected_columns,
            'imputed_columns': self.preprocessor.imputed_columns
        }
        
        preprocessor_path = os.path.join(models_dir, 'preprocessor_components.pkl')
        joblib.dump(preprocessor_components, preprocessor_path)
        print(f"✅ Preprocessor components saved: {preprocessor_path}")
        
        # Save results
        results_path = os.path.join(models_dir, 'training_results.csv')
        self.results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved: {results_path}")
        
        # Save results JSON with detailed metrics
        results_json_path = os.path.join(models_dir, 'training_results.json')
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'metrics': result['metrics'],
                'best_params': result['best_params'],
                'confusion_matrix': result['confusion_matrix'].tolist(),
                'classification_report': result['classification_report']
            }
        
        with open(results_json_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        print(f"✅ Detailed results saved: {results_json_path}")
        
        # Save validation data with predictions from all models
        print(f"\n{'='*70}")
        print(f"SAVING VALIDATION DATA")
        print(f"{'='*70}")
        
        validation_data = pd.DataFrame()
        validation_data['Actual'] = self.y_test
        
        # Add predictions from each model
        for model_name, result in self.results.items():
            pred_col_name = f'{model_name}_Pred'
            proba_col_name = f'{model_name}_Proba'
            validation_data[pred_col_name] = result['predictions']
            validation_data[proba_col_name] = result['probabilities']
        
        validation_path = os.path.join(models_dir, 'validation_data.csv')
        validation_data.to_csv(validation_path, index=False)
        print(f"✅ Validation data saved: {validation_path}")
        print(f"   Size: {len(validation_data)} samples")
        print(f"   Columns: {', '.join(validation_data.columns.tolist())}")
        
        # Save test features (X_test) for re-upload in Streamlit
        print(f"\n{'='*70}")
        print(f"SAVING TEST FEATURES FOR STREAMLIT UPLOAD")
        print(f"{'='*70}")
        
        # X_test is already a DataFrame with proper column names, use directly
        test_features_path = os.path.join(models_dir, 'test_features.csv')
        self.X_test.to_csv(test_features_path, index=False)
        print(f"✅ Test features saved: {test_features_path}")
        print(f"   Size: {len(self.X_test)} samples × {len(self.X_test.columns)} features")
        print(f"   Use this CSV to test models in Streamlit UI")
        
        # Save complete test set (features + actual labels)
        print(f"\n{'='*70}")
        print(f"SAVING COMPLETE TEST SET (FEATURES + ACTUAL LABELS)")
        print(f"{'='*70}")
        
        test_set_df = self.X_test.copy()
        test_set_df.insert(0, 'Actual', self.y_test)
        test_set_path = os.path.join(models_dir, 'test_set.csv')
        test_set_df.to_csv(test_set_path, index=False)
        print(f"✅ Complete test set saved: {test_set_path}")
        print(f"   Size: {len(test_set_df)} samples × {len(test_set_df.columns)} columns")
        print(f"   Columns: Actual + 169 features")
        print(f"   Use this to verify predictions in Streamlit UI")
        
        print(f"\n✅ All models saved to '{models_dir}' directory")


if __name__ == "__main__":
    import sys
    
    # Default dataset path
    dataset_path = 'aps_data.csv'
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        print(f"Usage: python train_all_models.py <dataset_path>")
        sys.exit(1)
    
    # Run training pipeline
    pipeline = TrainingPipeline(random_state=42)
    pipeline.run_training(dataset_path)
    pipeline.save_all_models('trained_models')
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Trained models saved in 'trained_models' directory")
    print(f"Run 'streamlit run app_production.py' to use the models for prediction")
