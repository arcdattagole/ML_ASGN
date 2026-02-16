"""
Production Streamlit App for APS Failure Prediction
Loads pre-trained models and performs inference on new data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="APS Failure Prediction System",
    page_icon="🚛",
    layout="wide",
    menu_items={'About': "Production ML Prediction System for APS Failures"}
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-title {
        color: #404040;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #08519c;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class ProductionModel:
    """Wrapper for production model inference"""
    
    def __init__(self, models_dir='trained_models'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.selected_columns = None
        self.imputed_columns = None
        self.results = {}
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models"""
        if not os.path.exists(self.models_dir):
            st.error(f"❌ Models directory not found: {self.models_dir}")
            st.info("Please run 'python train_all_models.py' to train and save models first.")
            st.stop()
        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        try:
            # Load preprocessor components
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor_components.pkl')
            if os.path.exists(preprocessor_path):
                components = joblib.load(preprocessor_path)
                self.scaler = components['scaler']
                self.imputer = components['imputer']
                self.feature_selector = components['feature_selector']
                self.selected_columns = components['selected_columns']
                self.imputed_columns = components['imputed_columns']
            else:
                st.error(f"❌ Preprocessor components not found: {preprocessor_path}")
                st.info("Please train models first with: python train_all_models.py aps_data.csv")
                st.stop()
            
            # Load models
            for model_name, file_name in model_files.items():
                file_path = os.path.join(self.models_dir, file_name)
                if os.path.exists(file_path):
                    self.models[model_name] = joblib.load(file_path)

            # If KNN is loaded, enable parallel neighbor search to speed up predictions
            if 'K-Nearest Neighbors' in self.models:
                try:
                    knn_model = self.models['K-Nearest Neighbors']
                    # Set n_jobs to -1 to use all cores for neighbor search/prediction
                    if hasattr(knn_model, 'n_jobs'):
                        knn_model.n_jobs = -1
                        self.models['K-Nearest Neighbors'] = knn_model
                except Exception:
                    pass
            
            # Load results
            results_path = os.path.join(self.models_dir, 'training_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def preprocess_data(self, X):
        """Preprocess new data using fitted transformers"""
        try:
            # Convert to numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Select only numeric columns
            X = X.select_dtypes(include=[np.number])
            # Normalize column names: strip whitespace
            X.columns = X.columns.str.strip()

            # If the uploaded data already contains the selected features (post-selection & scaled),
            # then skip imputation/scaling/selection and use directly.
            if self.selected_columns is not None:
                selected_set = set([c.strip() for c in self.selected_columns])
                uploaded_set = set(X.columns.tolist())

                if selected_set.issubset(uploaded_set) and len(X.columns) == len(self.selected_columns):
                    # Reindex to ensure correct column order and return
                    X = X.reindex(columns=[c.strip() for c in self.selected_columns])
                    return X

            # Ensure all imputed columns are present and in the same order as during training.
            # Missing columns will be filled with NaN so the imputer can handle them.
            expected_cols = list(self.imputed_columns) if self.imputed_columns is not None else list(X.columns)
            expected_cols = [c.strip() for c in expected_cols]
            X = X.reindex(columns=expected_cols)

            # Impute (the imputer expects the same feature names as during fit)
            X_imputed = self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=expected_cols)

            # Scale
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=expected_cols)

            # Select features
            X_selected = self.feature_selector.transform(X)
            X = pd.DataFrame(X_selected, columns=self.selected_columns)
            
            return X
        except Exception as e:
            st.error(f"❌ Preprocessing error: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def predict(self, model_name, X):
        """Make predictions with specified model"""
        if model_name not in self.models:
            return None, None
        
        try:
            model = self.models[model_name]
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            return y_pred, y_pred_proba
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None


def create_metric_card(label, value, format_str="{:.4f}"):
    """Create a metric card for display"""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{label}</div>
        <div class="metric-value">{format_str.format(value)}</div>
    </div>
    """


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Failure'],
        y=['Actual Normal', 'Actual Failure'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure(data=go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#08519c', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #08519c;'>
        🚛 APS Failure Prediction System
    </h1>
    <p style='text-align: center; color: #666;'>
        Production-grade ML System with 6 Trained Models
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Initialize session state
    if 'model_system' not in st.session_state:
        st.session_state.model_system = ProductionModel('trained_models')
    
    model_system = st.session_state.model_system
    
    # Check if models are loaded
    if not model_system.models:
        st.error("❌ No models found!")
        st.info("""
        Please train the models first:
        ```bash
        python train_all_models.py aps_data.csv
        ```
        """)
        st.stop()
    
    # Main tabs
    tab_prediction, tab_details, tab_evaluation = st.tabs([
        "🔮 Make Predictions",
        "📈 Detailed Analysis",
        "📊 Model Evaluation"
    ])
    
    # ==================== TAB 1: PREDICTIONS ====================
    with tab_prediction:
        st.header("Make Predictions on New Data")
        
        # Download sample data files
        st.subheader("📥 Download Sample Test Data")
        st.write("Use Test Data with-out Labels for making predictions. Use Test Data with Actual Labels for comparison and evaluation.")
        
        col1, col2, col3 = st.columns(3)
        
        models_dir = 'trained_models'
        
        with col1:
            if os.path.exists(os.path.join(models_dir, 'test_set.csv')):
                with open(os.path.join(models_dir, 'test_set.csv'), 'rb') as f:
                    st.download_button(
                        label="📊 Test Data with Actual Labels",
                        data=f.read(),
                        file_name="test_set.csv",
                        mime="text/csv"
                    )
            else:
                st.info("test_set.csv not available")
        
        with col2:
            if os.path.exists(os.path.join(models_dir, 'test_features.csv')):
                with open(os.path.join(models_dir, 'test_features.csv'), 'rb') as f:
                    st.download_button(
                        label="📋 Test Data without Labels",
                        data=f.read(),
                        file_name="test_features.csv",
                        mime="text/csv"
                    )
            else:
                st.info("test_features.csv not available")
        
        with col3:
            if os.path.exists(os.path.join(models_dir, 'validation_data.csv')):
                with open(os.path.join(models_dir, 'validation_data.csv'), 'rb') as f:
                    st.download_button(
                        label="🔍 Comparison of All Models",
                        data=f.read(),
                        file_name="validation_data.csv",
                        mime="text/csv"
                    )
            else:
                st.info("validation_data.csv not available")
        
        st.divider()
        
        # File upload
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📁 Upload CSV file for predictions",
                type=['csv'],
                help="CSV file should have same number of features as training data (169 features)"
            )
        
        with col2:
            st.write("")  # Spacing
        
        if uploaded_file is not None:
            try:
                # Load data
                st.info("Loading and preprocessing data...")
                data = pd.read_csv(uploaded_file, na_values=['NA', 'na', 'N.A', '', ' '])
                
                st.success(f"✅ Loaded {len(data)} samples with {len(data.columns)} columns")
                
                # Remove target column if present
                if data.shape[1] > 169:
                    data = data.iloc[:, :-1]
                
                st.write(f"**Data shape:** {data.shape}")
                
                # Preprocess
                with st.spinner("Preprocessing data..."):
                    X_processed = model_system.preprocess_data(data)
                
                if X_processed is not None:
                    st.success(f"✅ Preprocessed data shape: {X_processed.shape}")
                    
                    # Model selection
                    st.subheader("Select Models for Prediction")
                    
                    selected_models = st.multiselect(
                        "Choose models to predict with:",
                        list(model_system.models.keys()),
                        default=list(model_system.models.keys())
                    )
                    
                    if st.button("🔮 Generate Predictions", use_container_width=True):
                        st.divider()
                        
                        predictions_summary = []
                        
                        for model_name in selected_models:
                            with st.spinner(f"Predicting with {model_name}..."):
                                y_pred, y_pred_proba = model_system.predict(model_name, X_processed)
                                
                                if y_pred is not None:
                                    # Count predictions
                                    normal_count = (y_pred == 0).sum()
                                    failure_count = (y_pred == 1).sum()
                                    
                                    predictions_summary.append({
                                        'Model': model_name,
                                        'Normal': normal_count,
                                        'Failure': failure_count,
                                        'Failure %': f"{(failure_count/len(y_pred)*100):.2f}%"
                                    })
                                    
                                    # Display predictions
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-title">{model_name}</div>
                                            <div class="metric-title">Normal: {normal_count}</div>
                                            <div class="metric-value">Failure: {failure_count}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        # Summary table
                        st.divider()
                        st.subheader("Prediction Summary")
                        predictions_df = pd.DataFrame(predictions_summary)
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        st.divider()
                        predictions_csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Prediction Summary",
                            data=predictions_csv,
                            file_name="predictions_summary.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # ==================== TAB 2: DETAILED ANALYSIS ====================
    with tab_details:
        st.header("Detailed Model Analysis")
        
        # Model selection
        selected_model = st.selectbox(
            "Select a model for detailed analysis:",
            list(model_system.models.keys())
        )
        
        if selected_model and model_system.results:
            result = model_system.results[selected_model]
            metrics = result['metrics']
            
            # Metrics display
            st.subheader(f"{selected_model} - Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_metric_card("Accuracy", metrics['accuracy']), unsafe_allow_html=True)
                st.markdown(create_metric_card("Precision", metrics['precision']), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card("AUC Score", metrics['auc_score']), unsafe_allow_html=True)
                st.markdown(create_metric_card("Recall", metrics['recall']), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card("F1 Score", metrics['f1_score']), unsafe_allow_html=True)
                st.markdown(create_metric_card("MCC Score", metrics['mcc_score']), unsafe_allow_html=True)
            
            st.divider()
            
            # Hyperparameters
            st.subheader("Best Hyperparameters")
            
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)}
                for k, v in result['best_params'].items()
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            # Confusion Matrix and ROC Curve
            st.divider()
            st.subheader("Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm = np.array(result['confusion_matrix'])
                fig_cm = plot_confusion_matrix(
                    [0]*cm[0,0] + [1]*cm[0,1] + [0]*cm[1,0] + [1]*cm[1,1],
                    [0]*cm[0,0] + [0]*cm[0,1] + [1]*cm[1,0] + [1]*cm[1,1],
                    f"{selected_model} - Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("**ROC Curve** - Shows model's discriminative ability")
                st.info("Note: ROC curve generated from test set predictions")
                st.write("")
            
            # Classification Report
            st.subheader("Classification Report")
            st.text(result['classification_report'])
    
    # ==================== TAB 3: MODEL EVALUATION ====================
    with tab_evaluation:
        st.header("Model Evaluation & Comparison")
        
        if not model_system.results:
            st.error("No model results available")
            st.stop()
        
        # Model selector
        selected_model = st.selectbox(
            "Select Model for Detailed Evaluation:",
            list(model_system.results.keys()),
            key="eval_model_selector"
        )
        
        if selected_model:
            result = model_system.results[selected_model]
            metrics = result['metrics']
            
            # Confusion Matrix
            st.subheader(f"📊 Confusion Matrix - {selected_model}")
            
            cm = np.array(result['confusion_matrix'])
            
            # Create confusion matrix heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Normal', 'Predicted Failure'],
                y=['Actual Normal', 'Actual Failure'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ))
            fig_cm.update_layout(
                title=f"{selected_model} - Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader(f"📋 Classification Report - {selected_model}")
            st.text(result['classification_report'])
            
            st.divider()
            
            # Metrics comparison across all models
            st.subheader("📈 All Models Metrics Comparison")
            
            models_list = list(model_system.results.keys())
            metrics_names = ['accuracy', 'auc_score', 'precision', 'recall', 'f1_score', 'mcc_score']
            
            # Create comparison table
            comparison_data = []
            for model_name in models_list:
                metrics = model_system.results[model_name]['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'AUC Score': f"{metrics['auc_score']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1_score']:.4f}",
                    'MCC Score': f"{metrics['mcc_score']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Performance metrics radar chart
            st.subheader("🎯 Multi-Metric Comparison (Radar Chart)")
            
            fig_radar = go.Figure()
            
            for model_name in models_list:
                metrics = model_system.results[model_name]['metrics']
                values = [metrics[m] * 100 for m in metrics_names]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score'],
                    fill='toself',
                    name=model_name,
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="All Models - Multi-Metric Comparison",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.divider()
            
            # Best model summary
            st.subheader("🏆 Best Model Summary")
            
            best_model_idx = comparison_df['AUC Score'].astype(float).idxmax()
            best_model_name = comparison_df.loc[best_model_idx, 'Model']
            best_model_auc = comparison_df.loc[best_model_idx, 'AUC Score']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Model", best_model_name)
            
            with col2:
                st.metric("AUC Score", best_model_auc)
            
            with col3:
                best_f1 = comparison_df.loc[best_model_idx, 'F1 Score']
                st.metric("F1 Score", best_f1)


if __name__ == "__main__":
    main()
