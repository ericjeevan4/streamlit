from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("careermodel.joblib")
label_encoders = joblib.load("careerencoders.joblib")

@app.route("/")
def home():
    return jsonify({"message": "Career Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Get inputs
        age = data.get("age")
        hobby = data.get("hobby")
        entrepreneurship = data.get("entrepreneurship")
        favorite_subject = data.get("favorite_subject")

        # Encode categorical inputs
        hobby_enc = label_encoders['Hobbies'].transform([hobby])[0]
        entrepreneurship_enc = label_encoders['Entrepreneurship'].transform([entrepreneurship])[0]
        fav_sub_enc = label_encoders['Favorite_Subject'].transform([favorite_subject])[0]

        # Prepare features
        features = np.array([[age, hobby_enc, entrepreneurship_enc, fav_sub_enc]])
        prediction = model.predict(features)[0]

        # Decode prediction
        predicted_career = label_encoders['Predicted_Career_Domain'].inverse_transform([prediction])[0]

        return jsonify({"predicted_career": predicted_career})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
