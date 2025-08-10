import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os

# === Load data (supports both CSV and XLSX) ===
data_path = "data\liver_cirrhosis_edited.csv"  # Change to .csv if needed
if data_path.endswith(".xlsx"):
    df = pd.read_excel(data_path)
else:
    df = pd.read_csv(data_path)

# === Separate features & target ===
X = df.drop(columns=['Stage'])
y = df['Stage']

# === Load preprocessor ===
preprocessor = joblib.load('models/preprocessor.joblib')

# === Preprocess features ===
X_processed = preprocessor.transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# === Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"✅ Accuracy: {accuracy:.3f}")
print(f"✅ F1 Score (macro): {f1_macro}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save model ===
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/liver_stage_model.joblib')
print("\n💾 Saved trained model to models/liver_stage_model.joblib")
