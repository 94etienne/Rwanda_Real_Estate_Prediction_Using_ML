# SERVER SIDE FOR MODEL DEPLOYMENT
from flask import Flask, request, jsonify
import pickle
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and data columns
with open("../model_corrected/kigali_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("../json_corrected/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Define the location and property type mappings
location_mapping = {
    'gacuriro': 1,
    'kacyiru': 2,
    'kanombe': 3,
    'kibagabaga': 4,
    'kicukiro': 5,
    'kimironko': 6,
    'nyamirambo': 7,
    'nyarutarama': 8
}

property_type_mapping = {
    'apartment': 1,
    'bungalow': 2,
    'house': 3,
    'villa': 4
}

# Function to transform input data
def transform_data(size_sqm, number_of_bedrooms, number_of_bathrooms, number_of_floors, parking_space, location, property_type):
    x = np.zeros(len(data_columns))
    x[0] = size_sqm
    x[1] = number_of_bedrooms
    x[2] = number_of_bathrooms
    x[3] = number_of_floors
    x[5] = parking_space

    if location in location_mapping:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    if property_type in property_type_mapping:
        prop_index = data_columns.index(property_type)
        x[prop_index] = 1

    return np.array([x])

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    size_sqm = data.get("size_sqm")
    number_of_bedrooms = data.get("number_of_bedrooms")
    number_of_bathrooms = data.get("number_of_bathrooms")
    number_of_floors = data.get("number_of_floors")
    parking_space = data.get("parking_space")
    location = data.get("location")
    property_type = data.get("property_type")

    # Transform input data
    input_data_transformed = transform_data(size_sqm, number_of_bedrooms, number_of_bathrooms, number_of_floors, parking_space, location, property_type)
    
    # Predict using the model
    prediction = model.predict(input_data_transformed)
    response = {
        "prediction": round(prediction[0], 2)
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
