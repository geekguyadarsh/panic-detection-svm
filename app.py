from flask import Flask, request, jsonify
import numpy as np
# from kernel_svm import predictRes  # Import your ML function
import json
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
import joblib
from Predict import predictResult
# from flask_cors import CORS, cross_origin

app = Flask(__name__)
# CORS(app, support_credentials=True)

@app.route('/predict', methods=['POST'])
def predict():
    print("I GOT EXECUTEDDDD!!!!")
    try:
        data = request.json
        print(data)
        answers = np.array(data['answers'])
        print("hello :", answers)
    
        prediction = predictResult(answers)  
        print("@@@@", prediction)
        # return jsonify(int (prediction[0]))
        if (prediction[0]):
            detail = "Its a panic attack!" 
        else:
            detail = "Its not a panic attack!"

        return jsonify({"result": int (prediction[0]), "detail": str(detail)})
    
    except Exception as e:
        return jsonify({"message": "Error", "error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
    