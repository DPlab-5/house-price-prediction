# import azure.functions as func
# #import datetime
# #import json
# #import logging

# #app = func.FunctionApp()

# #import azure.functions as func
# import json
# import pickle
# import numpy as np
# import os
# import logging
# import os
# import pandas as pd
# import joblib

# # Initialize the v4 Function App
# app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# # BASE_DIR targets the exact folder where function_app.py lives
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Get the folder path where model is saved
# #BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Load model, scaler, and the Ames training column matrix layout when function starts
# def load_artifacts():
#     model_path = os.path.join(BASE_DIR, 'model.pkl')
#     scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
#     columns_path = os.path.join(BASE_DIR, 'model_columns.pkl')
    
#     # Using joblib as it handles large numpy/pandas models more efficiently than standard pickle
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     model_columns = joblib.load(columns_path)
    
#     return model, scaler, model_columns

# # Load model and scaler when function starts
# # def load_model():
# #     model_path = os.path.join(BASE_DIR, 'model.pkl')
# #     scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
# #     COLUMNS_PATH = os.path.join(BASE_DIR, 'model_columns.pkl')
    
# #     with open(model_path, 'rb') as f:
# #         model = pickle.load(f)
    
# #     with open(scaler_path, 'rb') as f:
# #         scaler = pickle.load(f)
    
# #     return model, scaler

# model, scaler, model_columns = load_artifacts()
# logging.info("✅ Ames Housing Model, Scaler, and Column Rules loaded successfully")

# @app.route(route="predict", methods=["GET", "POST", "OPTIONS"])
# def predict(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info("🏠 House price prediction request received")
    
#     # Handle CORS (allows frontend to call this API)
#     headers = {
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
#         "Access-Control-Allow-Headers": "Content-Type"
#     }
    
#     # Handle OPTIONS request (preflight)
#     if req.method == "OPTIONS":
#         return func.HttpResponse(
#             status_code=200,
#             headers=headers
#         )
    
#     try:
#         # Get the input data from request
#         req_body = req.get_json()
#         logging.info(f"Input received: {req_body}")
        
#         # 2. Turn the raw payload into a base DataFrame
#         input_df = pd.DataFrame([req_body])
        
#         # 3. Replicate the One-Hot Encoding used during training
#         input_df_encoded = pd.get_dummies(input_df)
        
#         # 4. Reindex the columns to perfectly match the 297 training columns.
#         # Missing columns become 0, and irrelevant columns passed in are automatically dropped.
#         final_features = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
#         # 5. Scale features using the saved Ames training distribution rules
#         features_scaled = scaler.transform(final_features)
        
#         # 6. Run Inference
#         predicted_price = model.predict(features_scaled)[0]
#         predicted_price = round(float(predicted_price), 2)

#         # Extract house details from input
#         # longitude          = float(req_body.get('longitude', -119.5))
#         # latitude           = float(req_body.get('latitude', 35.6))
#         # housing_median_age = float(req_body.get('housing_median_age', 20))
#         # total_rooms        = float(req_body.get('total_rooms', 2000))
#         # total_bedrooms     = float(req_body.get('total_bedrooms', 400))
#         # population         = float(req_body.get('population', 1000))
#         # households         = float(req_body.get('households', 350))
#         # median_income      = float(req_body.get('median_income', 4.0))
#         # ocean_proximity    = req_body.get('ocean_proximity', 'INLAND')
        
#         # Convert ocean_proximity text to numbers
#         # (same way we did during training)
#         # ocean_map = {
#         #     'INLAND': [1, 0, 0, 0, 0],
#         #     'NEAR BAY': [0, 1, 0, 0, 0],
#         #     'NEAR OCEAN': [0, 0, 1, 0, 0],
#         #     '<1H OCEAN': [0, 0, 0, 1, 0],
#         #     'ISLAND': [0, 0, 0, 0, 1]
#         # }
#         #ocean_encoded = ocean_map.get(ocean_proximity, [1, 0, 0, 0, 0])
        
#         # Combine all features into one array
#         # features = [
#         #     longitude,
#         #     latitude,
#         #     housing_median_age,
#         #     total_rooms,
#         #     total_bedrooms,
#         #     population,
#         #     households,
#         #     median_income
#         # ] + ocean_encoded
        
#         # # Scale the features
#         # features_array = np.array(features).reshape(1, -1)
#         # features_scaled = scaler.transform(features_array)
        
#         # # Make prediction
#         # predicted_price = model.predict(features_scaled)[0]
#         # predicted_price = round(float(predicted_price), 2)
        
#         response_data = {
#             "status": "success",
#             "predicted_price": predicted_price,
#             "formatted_price": f"${predicted_price:,.2f}",
#             "input_received": req_body
#         }
        
#         return func.HttpResponse(
#             body=json.dumps(response_data),
#             status_code=200,
#             mimetype="application/json",
#             headers=headers
#         )
    
#     except ValueError as e:
#         error = {"status": "error", "message": f"Data conversion or type processing failure: {str(e)}"}
#         return func.HttpResponse(
#             body=json.dumps(error),
#             status_code=400,
#             mimetype="application/json",
#             headers=headers
#         )
    
#     except Exception as e:
#         error = {"status": "error", "message": f"Server Pipeline Error: {str(e)}"}
#         return func.HttpResponse(
#             body=json.dumps(error),
#             status_code=500,
#             mimetype="application/json",
#             headers=headers
#         )

#         # Return the result
#     #     response_data = {
#     #         "status": "success",
#     #         "predicted_price": predicted_price,
#     #         "formatted_price": f"${predicted_price:,.2f}",
#     #         "input_received": req_body
#     #     }
        
#     #     return func.HttpResponse(
#     #         body=json.dumps(response_data),
#     #         status_code=200,
#     #         mimetype="application/json",
#     #         headers=headers
#     #     )
    
#     # except ValueError as e:
#     #     # Input was wrong format
#     #     error = {"status": "error", "message": f"Invalid input: {str(e)}"}
#     #     return func.HttpResponse(
#     #         body=json.dumps(error),
#     #         status_code=400,
#     #         mimetype="application/json",
#     #         headers=headers
#     #     )
    
#     # except Exception as e:
#     #     # Something else went wrong
#     #     error = {"status": "error", "message": str(e)}
#     #     return func.HttpResponse(
#     #         body=json.dumps(error),
#     #         status_code=500,
#     #         mimetype="application/json",
#     #         headers=headers
#     #     )

# @app.route(route="predict", auth_level=func.AuthLevel.ANONYMOUS)
# def predict(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Python HTTP trigger function processed a request.')

#     name = req.params.get('name')
#     if not name:
#         try:
#             req_body = req.get_json()
#         except ValueError:
#             pass
#         else:
#             name = req_body.get('name')

#     if name:
#         return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#     else:
#         return func.HttpResponse(
#              "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
#              status_code=200
#         )


import azure.functions as func
import json
import numpy as np
import os
import logging
import pandas as pd
import joblib

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    model   = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
    scaler  = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))
    return model, scaler, columns

model, scaler, model_columns = load_artifacts()

@app.route(route="predict", methods=["GET", "POST", "OPTIONS"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("House price prediction request received")

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }

    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=200, headers=headers)

    try:
        req_body         = req.get_json()
        input_df         = pd.DataFrame([req_body])
        input_df_encoded = pd.get_dummies(input_df)
        final_features   = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        features_scaled  = scaler.transform(final_features)
        predicted_price  = round(float(model.predict(features_scaled)[0]), 2)

        return func.HttpResponse(
            body=json.dumps({
                "status": "success",
                "predicted_price": predicted_price,
                "formatted_price": f"${predicted_price:,.2f}",
                "input_received": req_body
            }),
            status_code=200,
            mimetype="application/json",
            headers=headers
        )

    except Exception as e:
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json",
            headers=headers
        )