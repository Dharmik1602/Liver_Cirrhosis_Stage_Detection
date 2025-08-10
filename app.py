from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load preprocessor & model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/liver_stage_model.joblib")

# Default values from training data
default_values = {
    "Prothrombin": 10.6,
    "Platelets": 251.0,
    "Albumin": 3.5,
    "Age_years": 50.7,
    "Bilirubin": 1.3,
    "Sex": "M",
    "Drug": "D-penicil"
}

# Stage descriptions
stage_descriptions = {
    1: "Early stage – mild liver damage, close monitoring recommended.",
    2: "Moderate stage – noticeable liver impairment, active treatment needed.",
    3: "Advanced stage – significant liver damage, intensive care required."
}

def predict_from_df(df):
    """Function to make predictions from DataFrame"""
    try:
        X_processed = preprocessor.transform(df)
        pred = model.predict(X_processed)
        proba = model.predict_proba(X_processed)
        
        results = []
        for i, (p, probs) in enumerate(zip(pred, proba)):
            pred_stage = int(p)
            result = {
                'stage': pred_stage,
                'description': stage_descriptions.get(pred_stage, 'No description available.'),
                'probabilities': {int(stage): float(prob) for stage, prob in zip(model.classes_, probs)}
            }
            results.append(result)
        
        return results
    except Exception as e:
        return [{'error': str(e)}]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        
        try:
            if input_type == 'file':
                # Handle file upload
                file = request.files['file']
                if file.filename == '':
                    return render_template('index.html', error="No file selected")
                
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)
                else:
                    return render_template('index.html', error="Please upload a CSV or XLSX file")
                
                results = predict_from_df(df)
                
            elif input_type == 'manual':
                # Handle manual input
                user_data = {
                    "Prothrombin": float(request.form['Prothrombin']),
                    "Platelets": float(request.form['Platelets']),
                    "Albumin": float(request.form['Albumin']),
                    "Age_years": float(request.form['Age_years']),
                    "Bilirubin": float(request.form['Bilirubin']),
                    "Sex": default_values["Sex"],
                    "Drug": default_values["Drug"]
                }
                
                # Create DataFrame with exact feature order
                all_columns = preprocessor.feature_names_in_
                df = pd.DataFrame([{col: user_data.get(col, default_values.get(col)) for col in all_columns}])
                
                results = predict_from_df(df)
                
            else:
                return render_template('index.html', error="Invalid input type")
                
            return render_template('index.html', results=results)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    try:
        # Check if file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Please upload a CSV or XLSX file'}), 400
        else:
            # Handle JSON data
            data = request.get_json()
            
            # Use provided values or defaults
            user_data = {
                "Prothrombin": float(data.get('Prothrombin', default_values['Prothrombin'])),
                "Platelets": float(data.get('Platelets', default_values['Platelets'])),
                "Albumin": float(data.get('Albumin', default_values['Albumin'])),
                "Age_years": float(data.get('Age_years', default_values['Age_years'])),
                "Bilirubin": float(data.get('Bilirubin', default_values['Bilirubin'])),
                "Sex": default_values["Sex"],
                "Drug": default_values["Drug"]
            }
            
            # Create DataFrame
            all_columns = preprocessor.feature_names_in_
            df = pd.DataFrame([{col: user_data.get(col, default_values.get(col)) for col in all_columns}])
        
        results = predict_from_df(df)
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
