
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Set paths for datasets
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cicids2017_path = os.path.join(base_dir, 'data', 'MachineLearningCSV (2017)', 'MachineLearningCVE')
cicids2018_path = os.path.join(base_dir, 'data', 'MachineLearningCSV (2018)', 'MachineLearningCVE')
output_path = os.path.join(base_dir, 'data')

def load_and_process_dataset(file_paths, year):
    """Load and process multiple CSV files from a dataset."""
    all_data = []
    
    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}...")
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
            
        # Clean column names (remove leading/trailing spaces)
        df.columns = [col.strip() for col in df.columns]
        
        # Add dataset year as a feature
        df['Dataset_Year'] = year
        
        # Add attack type based on filename
        filename = os.path.basename(file_path).lower()
        if 'ddos' in filename:
            df['Attack_Type'] = 'DDoS'
        elif 'portscan' in filename:
            df['Attack_Type'] = 'PortScan'
        elif 'infilteration' in filename or 'infiltration' in filename:
            df['Attack_Type'] = 'Infiltration'
        elif 'webattacks' in filename:
            df['Attack_Type'] = 'WebAttack'
        else:
            df['Attack_Type'] = 'Normal'
            
        all_data.append(df)
        
    # Combine all dataframes
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined shape for {year}: {combined_df.shape}")
    
    return combined_df

def standardize_labels(df, label_col):
    if label_col not in df.columns:
        return df
    df['standardized_label'] = df[label_col].apply(lambda x: 0 if str(x).lower() == 'benign' or str(x).lower() == 'normal' else 1)
    return df

def main():
    print("Starting data processing...")
    
    # Get all CSV files
    cicids2017_files = glob.glob(os.path.join(cicids2017_path, '*.csv'))
    cicids2018_files = glob.glob(os.path.join(cicids2018_path, '*.csv'))
    
    if not cicids2017_files and not cicids2018_files:
        print("No data files found!")
        return

    # Use a sample of files to avoid memory issues (taking first 1 from each for demo)
    # In production, you might want to process all or a larger subset
    print("Processing sample files...")
    cicids2017_sample = cicids2017_files[:1] if cicids2017_files else []
    cicids2018_sample = cicids2018_files[:1] if cicids2018_files else []
    
    df_2017 = load_and_process_dataset(cicids2017_sample, 2017)
    df_2018 = load_and_process_dataset(cicids2018_sample, 2018)
    
    # Standardize labels
    if not df_2017.empty:
        label_col_2017 = 'Label' if 'Label' in df_2017.columns else ' Label'
        df_2017 = standardize_labels(df_2017, label_col_2017)
        
    if not df_2018.empty:
        label_col_2018 = 'Label' if 'Label' in df_2018.columns else ' Label'
        df_2018 = standardize_labels(df_2018, label_col_2018)
    
    # Combine datasets
    print("Combining datasets...")
    if df_2017.empty and df_2018.empty:
        print("No data loaded.")
        return
    elif df_2017.empty:
        combined_df = df_2018
    elif df_2018.empty:
        combined_df = df_2017
    else:
        # Find common numeric columns
        common_cols = list(set(df_2017.columns).intersection(set(df_2018.columns)))
        numeric_cols = []
        for col in common_cols:
            if df_2017[col].dtype in ['int64', 'float64'] and df_2018[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        features_to_use = numeric_cols + ['standardized_label', 'Dataset_Year', 'Attack_Type']
        df_2017_subset = df_2017[features_to_use]
        df_2018_subset = df_2018[features_to_use]
        combined_df = pd.concat([df_2017_subset, df_2018_subset], ignore_index=True)
    
    # Handle missing values
    combined_df = combined_df.fillna(0)
    
    # Replace Infinity with large number or 0
    combined_df = combined_df.replace([np.inf, -np.inf], 0)
    
    # Prepare features for PCA
    X = combined_df.drop(['standardized_label', 'Dataset_Year', 'Attack_Type'], axis=1)
    y = combined_df['standardized_label']
    
    # Limit samples for PCA to avoid memory errors if dataset is huge
    if len(X) > 100000:
        print("Downsampling for PCA/Scaler fitting...")
        indices = np.random.choice(len(X), 100000, replace=False)
        X_fit = X.iloc[indices]
    else:
        X_fit = X
        
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    scaler.fit(X_fit)
    X_scaled = scaler.transform(X)
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=min(20, X.shape[1]))
    pca.fit(scaler.transform(X_fit))
    X_pca = pca.transform(X_scaled)
    
    # Create DataFrame with PCA features
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    # We need to ensure we use the same indices if we downsampled for PCA fitting
    # But wait, we applied PCA transform to X_scaled which is the FULL dataset
    # So X_pca should have same length as original combined_df
    
    # However, earlier 'combined_df = combined_df.replace([np.inf, -np.inf], 0)' might have created a copy?
    # Or 'y = combined_df['standardized_label']' was done BEFORE the replace?
    # Let's re-extract y from combined_df just to be safe and ensure alignment
    
    y = combined_df['standardized_label']
    
    # Ensure y is 1D
    if hasattr(y, 'values'):
        y_vals = y.values.ravel()
    else:
        y_vals = np.ravel(y)
    
    print(f"PCA shape: {pca_df.shape}")
    print(f"Labels shape: {y_vals.shape}")
    
    # If shapes still don't match, trim to the smaller one (though they should match)
    min_len = min(len(pca_df), len(y_vals))
    pca_df = pca_df.iloc[:min_len]
    y_vals = y_vals[:min_len]
    
    pca_df['label'] = y_vals
    
    # Ensure other columns are 1D
    year_vals = combined_df['Dataset_Year'].values
    if hasattr(year_vals, 'ravel'):
        year_vals = year_vals.ravel()
    else:
        year_vals = np.ravel(year_vals)
        
    attack_vals = combined_df['Attack_Type'].values
    if hasattr(attack_vals, 'ravel'):
        attack_vals = attack_vals.ravel()
    else:
        attack_vals = np.ravel(attack_vals)
    
    pca_df['dataset_year'] = year_vals[:min_len]
    pca_df['attack_type'] = attack_vals[:min_len]
    
    # Save processed data
    output_file = os.path.join(output_path, 'cicids_combined_data.csv')
    pca_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # Save models
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'threatvision_scaler.pkl'))
    joblib.dump(pca, os.path.join(models_dir, 'threatvision_pca.pkl'))
    print("Saved scaler and PCA models")

if __name__ == "__main__":
    main()
