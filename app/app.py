from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Create Flask app
app = Flask(__name__)

# Load saved model and preprocessor
model = joblib.load("../models/churn_model.joblib")
preprocessor = joblib.load("../models/preprocessor.joblib")

@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.get_json()

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # =========================
    # Feature Engineering
    # =========================

    input_df['TotalServices'] = (
        input_df[
            [
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies'
            ]
        ] == 'Yes'
    ).sum(axis=1)

    input_df['AvgMonthlyCharges'] = np.where(
        input_df['tenure'] == 0,
        input_df['MonthlyCharges'],
        input_df['TotalCharges'] / input_df['tenure']
    )

    # Tenure Group
    bins = [-1, 12, 24, 48, np.inf]
    labels = ["New", "Early", "Established", "Long-term"]

    input_df["TenureGroup"] = pd.cut(
        input_df["tenure"],
        bins=bins,
        labels=labels
    )

    # =========================
    # Preprocessing
    # =========================

    transformed_data = preprocessor.transform(input_df)

    # =========================
    # Prediction
    # =========================

    churn_probability = model.predict_proba(transformed_data)[:, 1][0]

    prediction = int(churn_probability >= 0.40)

    # =========================
    # Return Response
    # =========================

    return jsonify({
        "churn_probability": round(float(churn_probability), 4),
        "prediction": prediction,
         "prediction_label": "Churn" if prediction == 1 else "No Churn"
    })


if __name__ == "__main__":
    app.run(debug=True)