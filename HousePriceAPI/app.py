from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model        = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler       = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

    try:
        req_body         = request.get_json()
        input_df         = pd.DataFrame([req_body])
        input_df_encoded = pd.get_dummies(input_df)
        final_features   = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        features_scaled  = scaler.transform(final_features)
        predicted_price  = round(float(model.predict(features_scaled)[0]), 2)

        response = jsonify({
            "status": "success",
            "predicted_price": predicted_price,
            "formatted_price": f"${predicted_price:,.2f}"
        })
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 200

    except Exception as e:
        response = jsonify({"status": "error", "message": str(e)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)