import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# === Load data (supports both CSV and XLSX) ===
data_path = "data\liver_cirrhosis_edited.csv"  # Change to .csv if needed
if data_path.endswith(".xlsx"):
    df = pd.read_excel(data_path)
else:
    df = pd.read_csv(data_path)

# === Drop irrelevant columns ===
df.drop(columns=['N_Days', 'Status'], inplace=True)

# === Separate features and target ===
X = df.drop(columns=['Stage'])
y = df['Stage']

# === Identify numeric & categorical features ===
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# === Create transformers ===
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# === Combine into column transformer ===
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# === Fit & transform ===
X_processed = preprocessor.fit_transform(X)

# === Save preprocessor ===
os.makedirs('models', exist_ok=True)
joblib.dump(preprocessor, 'models/preprocessor.joblib')

print(f"✅ Preprocessing complete. Shape of processed features: {X_processed.shape}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")
