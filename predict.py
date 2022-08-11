import pickle

import xgboost as xgb
import numpy as np
from flask import Flask
from flask import render_template, request

model_file = 'model_v0.bin'
print('loading model...')
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print('Loaded model: ', (dv, model))

# app = Flask(__name__)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    house = {
        "housing_median_age": 35.0,
        "total_rooms": 2061.0,
        "total_bedrooms": 371.0,
        "population": 1110.0,
        "households": 342.0,
        "median_income": 3.1944,
        "longitude": -121.41,
        "latitude": 38.53,
        "ocean_proximity": "inland",
    }
    house_features = [x for x in request.form.values()]
    print('customer features: ', house_features)
    pos = 0
    for key, value in house.items():
        if pos == len(house):
            break
        value = house_features[pos]
        # if value is contains letters or '_' then it is a string
        if not (value.isalpha() or '_' in value):
            value = float(value)
        if key == "median_income":
            value = value / 10000
        house[key] = value
        pos += 1

    print('A house: ', house)
    X = dv.transform([house])
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    predicted_house_price_log = model.predict(dtest)[0]
    predicted_house_price = np.expm1(predicted_house_price_log)
    prediction_text = 'The predicted house price is: ${}'.format(predicted_house_price)
    print(prediction_text)
    return render_template('index.html',prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
