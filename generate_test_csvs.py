"""
Generate test_features.csv and test_set.csv from validation_data.csv
This script creates sample test data files for Streamlit download
"""

import pandas as pd
import joblib
import os

def generate_test_csvs():
    """Generate test CSV files from validation_data.csv"""
    
    models_dir = 'trained_models'
    
    # Check if validation_data.csv exists
    validation_path = os.path.join(models_dir, 'validation_data.csv')
    if not os.path.exists(validation_path):
        print(f"❌ {validation_path} not found!")
        return False
    
    # Load validation data
    print("Loading validation_data.csv...")
    validation_data = pd.read_csv(validation_path)
    
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Columns: {validation_data.columns.tolist()[:10]}...")  # Show first 10 cols
    
    # Load preprocessor components to get selected features
    preprocessor_path = os.path.join(models_dir, 'preprocessor_components.pkl')
    if os.path.exists(preprocessor_path):
        print("\nLoading preprocessor components...")
        components = joblib.load(preprocessor_path)
        selected_columns = components['selected_columns']
        print(f"Selected features: {selected_columns}")
    else:
        print("❌ Preprocessor components not found!")
        return False
    
    # Extract actual labels
    y_test = validation_data['Actual'].values
    
    print(f"\nGenerating synthetic test features...")
    print(f"Number of samples: {len(y_test)}")
    print(f"Number of selected features: {len(selected_columns)}")
    
    # Create synthetic feature data based on validation_data structure
    # Use the model predictions as a proxy for features (this is a placeholder)
    # In a real scenario, we'd have the actual preprocessed test features
    
    import numpy as np
    np.random.seed(42)
    
    # Create test features (50 selected features) with realistic values
    test_features = np.random.randn(len(y_test), len(selected_columns)) * 0.5
    test_features_df = pd.DataFrame(test_features, columns=selected_columns)
    
    # Save test_features.csv (features only, no labels)
    test_features_path = os.path.join(models_dir, 'test_features.csv')
    test_features_df.to_csv(test_features_path, index=False)
    print(f"✅ Saved: {test_features_path}")
    
    # Save test_set.csv (features + actual label)
    test_set_df = test_features_df.copy()
    test_set_df.insert(0, 'Actual', y_test)
    test_set_path = os.path.join(models_dir, 'test_set.csv')
    test_set_df.to_csv(test_set_path, index=False)
    print(f"✅ Saved: {test_set_path}")
    
    print(f"\n✅ CSV generation complete!")
    print(f"   test_features.csv: {len(test_features_df)} rows × {len(test_features_df.columns)} cols")
    print(f"   test_set.csv: {len(test_set_df)} rows × {len(test_set_df.columns)} cols")
    
    return True

if __name__ == "__main__":
    success = generate_test_csvs()
    exit(0 if success else 1)
