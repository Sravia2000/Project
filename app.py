from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model_filename = 'final_gradient_boosting_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_gb_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

# Preprocessing scalers (these should match the scalers used during model training)
scalers = {
    'Tenure': StandardScaler(),
    'CityTier': StandardScaler(),
    'WarehouseToHome': StandardScaler(),
    'HourSpendOnApp': StandardScaler(),
    'NumberOfDeviceRegistered': StandardScaler(),
    'SatisfactionScore': StandardScaler(),
    'NumberOfAddress': StandardScaler(),
    'OrderAmountHikeFromlastYear': StandardScaler(),
    'CouponUsed': StandardScaler(),
    'OrderCount': StandardScaler(),
    'DaySinceLastOrder': StandardScaler(),
    'CashbackAmount': StandardScaler()
}

# Dummy fit the scalers to some data to avoid NotFittedError
# Replace with actual training data if available
dummy_data = np.random.rand(100, len(scalers))
for i, key in enumerate(scalers.keys()):
    scalers[key].fit(dummy_data[:, [i]])

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    tenure = float(request.form['tenure'])
    city_tier = float(request.form['cityTier'])
    warehouse_to_home = float(request.form['warehouseToHome'])
    hour_spent_on_app = float(request.form['hourSpendOnApp'])
    number_of_device_registered = float(request.form['numberOfDeviceRegistered'])
    satisfaction_score = float(request.form['satisfactionScore'])
    number_of_address = float(request.form['numberOfAddress'])
    order_amount_hike = float(request.form['orderAmountHikeFromlastYear'])
    coupon_used = float(request.form['couponUsed'])
    order_count = float(request.form['orderCount'])
    day_since_last_order = float(request.form['daySinceLastOrder'])
    cashback_amount = float(request.form['cashbackAmount'])
    gender = request.form['gender']
    logindevice = request.form['preferredLoginDevice']
    payment_mode = request.form['preferredPaymentMode']
    order_cat = request.form['preferedOrderCat']
    marital_status = request.form['maritalStatus']
    complain = float(request.form['complain'])

    # Encode categorical variables
    logindevice_mapping = {'Handphone': 0, 'Computer': 1}
    payment_mode_mapping = {'Cash on Delivery': 0, 'E Wallet': 1}
    order_cat_mapping = {'Electronics': 0, 'Fashion': 1, 'Grocery': 2, 'Others': 3}
    marital_status_mapping = {'Single': 0, 'Married': 1, 'Divorced': 2}

    logindevice_encoded = logindevice_mapping[logindevice]
    payment_mode_encoded = payment_mode_mapping[payment_mode]
    order_cat_encoded = order_cat_mapping[order_cat]
    marital_status_encoded = marital_status_mapping[marital_status]

    # Scale numeric features
    tenure_scaled = scalers['Tenure'].transform([[tenure]])[0][0]
    city_tier_scaled = scalers['CityTier'].transform([[city_tier]])[0][0]
    warehouse_to_home_scaled = scalers['WarehouseToHome'].transform([[warehouse_to_home]])[0][0]
    hour_spent_on_app_scaled = scalers['HourSpendOnApp'].transform([[hour_spent_on_app]])[0][0]
    number_of_device_registered_scaled = scalers['NumberOfDeviceRegistered'].transform([[number_of_device_registered]])[0][0]
    satisfaction_score_scaled = scalers['SatisfactionScore'].transform([[satisfaction_score]])[0][0]
    number_of_address_scaled = scalers['NumberOfAddress'].transform([[number_of_address]])[0][0]
    order_amount_hike_scaled = scalers['OrderAmountHikeFromlastYear'].transform([[order_amount_hike]])[0][0]
    coupon_used_scaled = scalers['CouponUsed'].transform([[coupon_used]])[0][0]
    order_count_scaled = scalers['OrderCount'].transform([[order_count]])[0][0]
    day_since_last_order_scaled = scalers['DaySinceLastOrder'].transform([[day_since_last_order]])[0][0]
    cashback_amount_scaled = scalers['CashbackAmount'].transform([[cashback_amount]])[0][0]

    # Combine all features into a single array for prediction
    features = [
        tenure_scaled, city_tier_scaled, warehouse_to_home_scaled, hour_spent_on_app_scaled,
        number_of_device_registered_scaled, satisfaction_score_scaled, number_of_address_scaled,
        order_amount_hike_scaled, coupon_used_scaled, order_count_scaled, day_since_last_order_scaled,
        cashback_amount_scaled, gender, logindevice_encoded, payment_mode_encoded,
        order_cat_encoded, complain, marital_status_encoded
    ]

    # Make prediction
    prediction = loaded_gb_model.predict([features])
    result = 'This Customer Will Churn' if prediction[0] == 1 else 'This Customer Will Not Churn'
    prediction_class = 'prediction-churn' if result == 'This Customer Will Churn' else 'prediction-no-churn'

    # Return prediction result with CSS class
    return render_template('index.html', prediction=result, prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True)