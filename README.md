# Liver Cirrhosis Stage Detection - Flask App

A simple Flask web application for predicting liver cirrhosis stages using a pre-trained machine learning model.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app:**
   ```bash
   python app.py
   ```

3. **Access the app:**
   - Open your browser and go to: `http://localhost:5000`
   - Fill in the patient data and click "Predict Stage"

## API Usage

You can also use the JSON API endpoint:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Prothrombin": 10.6,
    "Platelets": 251.0,
    "Albumin": 3.5,
    "Age_years": 50.7,
    "Bilirubin": 1.3,
    "Sex": "M",
    "Drug": "D-penicil"
  }'
```

## Features

- Simple web form for manual input
- JSON API for programmatic access
- Real-time predictions using pre-trained model
- Probability scores for each stage
- Clean, responsive UI

## Project Structure

- `app.py` - Main Flask application
- `models/` - Pre-trained model and preprocessor
- `src/` - Original ML scripts (training, preprocessing, etc.)
- `data/` - Training datasets
