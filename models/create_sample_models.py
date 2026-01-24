import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

# Create sample data
np.random.seed(42)
n_samples = 1000
n_features = 78  # CICIDS2017 dataset features

# Generate random features
X = np.random.randn(n_samples, n_features)
# Generate binary labels (normal=0, attack=1)
y = np.random.randint(0, 2, size=n_samples)

# Create and fit a scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit PCA
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# Create and fit a simple Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_pca, y)

# Save the models
joblib.dump(scaler, 'threatvision_scaler.pkl')
joblib.dump(pca, 'threatvision_pca.pkl')
joblib.dump(model, 'threatvision_rf_model.pkl')

print("Sample models created successfully!")
print(f"Model files saved to: {os.path.abspath('.')}")