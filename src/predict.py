import pandas as pd
import joblib

# === Load preprocessor & model ===
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/liver_stage_model.joblib")

# === Define defaults from training data ===
default_values = {
    "Prothrombin": 10.6,
    "Platelets": 251.0,
    "Albumin": 3.5,
    "Age_years": 50.7,
    "Bilirubin": 1.3,
    "Sex": "M",            # default category
    "Drug": "D-penicil"    # default category
}

# === Stage descriptions ===
stage_descriptions = {
    1: "Early stage – mild liver damage, close monitoring recommended.",
    2: "Moderate stage – noticeable liver impairment, active treatment needed.",
    3: "Advanced stage – significant liver damage, intensive care required."
}

# === Function to predict ===
def predict_from_df(df):
    X_processed = preprocessor.transform(df)
    pred = model.predict(X_processed)
    proba = model.predict_proba(X_processed)
    return pred, proba

# === Function to print probability bars ===
def print_probability_bar(stage, probability):
    bar_length = int(probability * 20)  # Scale bar length
    bar = "█" * bar_length + "-" * (20 - bar_length)
    print(f"Stage {stage}: {probability:.2%} | {bar}")

# === User input ===
print("Choose input method:")
print("1. CSV or XLSX file (must contain all required features)")
print("2. Enter only 5 key values (rest will be defaulted)")
choice = input("Enter choice (1/2): ")

if choice == "1":
    file_path = input("Enter file path: ").strip()
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

elif choice == "2":
    user_data = {}
    for col in ["Prothrombin", "Platelets", "Albumin", "Age_years", "Bilirubin"]:
        val = input(f"Enter {col} (default={default_values[col]}): ")
        user_data[col] = float(val) if val.strip() != "" else default_values[col]

    # Fill in missing categorical values
    for col in ["Sex", "Drug"]:
        user_data[col] = default_values[col]

    # Create DataFrame with exact feature order from training
    all_columns = preprocessor.feature_names_in_
    df = pd.DataFrame([{col: user_data.get(col, default_values.get(col)) for col in all_columns}])

else:
    raise ValueError("Invalid choice! Please enter 1 or 2.")

# === Prediction ===
pred, proba = predict_from_df(df)
pred_stage = pred[0]

# === Output ===
print("\n=== Input Summary ===")
print(df.to_string(index=False))

print(f"\n✅ Predicted Stage: {pred_stage}")
print(f"ℹ️ {stage_descriptions.get(pred_stage, 'No description available.')}\n")

print("🔍 Probabilities per stage:")
for stage, p in zip(model.classes_, proba[0]):
    print_probability_bar(stage, p)
