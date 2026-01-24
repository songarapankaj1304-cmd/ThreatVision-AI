import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def main():
    print("Loading processed data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cicids_combined_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    data = pd.read_csv(data_path)
    
    print(f"Data shape: {data.shape}")
    
    # Prepare features and target
    # Drop non-feature columns
    drop_cols = ['label', 'dataset_year', 'attack_type']
    # Also drop any other columns that are not PC columns just in case
    feature_cols = [col for col in data.columns if col not in drop_cols]
    
    X = data[feature_cols]
    y = data['label']
    
    print(f"Features: {X.columns.tolist()}")
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    print("Initializing Random Forest model...")
    # Use fewer estimators to speed up training for this demonstration if dataset is huge
    # But for quality, let's stick to a reasonable number. 
    # Since the dataset might be large (900k rows), let's limit max_depth or n_estimators to avoid long wait times
    # user wants "perfect", so let's try a balanced approach.
    model = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42)
    
    # Train
    print("Training model (this may take a moment)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'threatvision_rf_model.pkl')
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Done!")

if __name__ == "__main__":
    main()
