from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your ensemble model
model = pickle.load(open("churn_model.pkl", "rb"))

# 🔹 Reason & Recommendation Logic
def get_reason_and_recommendations(features, prediction):
    """
    features: dictionary of user input
    prediction: 0 or 1 (no churn / churn)
    """
    reasons = []
    recommendations = []

    # Check important triggers
    if features['International plan'] == 1:
        reasons.append("Customer has International Plan which increases churn risk.")
        recommendations.append("Offer discounts or revise International Plan charges.")

    if features['Customer service calls'] > 3:
        reasons.append(f"High customer service calls ({features['Customer service calls']}) indicate dissatisfaction.")
        recommendations.append("Improve support quality and resolve issues quickly.")

    if features['intl_plan_day_minutes'] > 200:
        reasons.append("High day minutes with International Plan indicates high billing risk.")
        recommendations.append("Provide loyalty discounts for heavy day usage customers.")

    if features['cust_serv_intl'] > 2:
        reasons.append("Frequent service calls with International Plan — potential dissatisfaction.")
        recommendations.append("Provide personalized assistance and incentives.")

    if features['total_calls_ratio'] > 0.5:
        reasons.append("High call ratio compared to account age indicates usage stress.")
        recommendations.append("Offer bundled packages to retain heavy callers.")

    if not reasons:
        reasons.append("No major churn indicators detected.")
        recommendations.append("Maintain regular engagement with customer.")

    return reasons, recommendations


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Get inputs from form
        account_length = int(request.form['account_length'])
        area_code = int(request.form['area_code'])
        intl_plan = 1 if request.form['intl_plan'].lower() == "yes" else 0
        vmail_plan = 1 if request.form['vmail_plan'].lower() == "yes" else 0
        vmail_messages = int(request.form['vmail_messages'])
        total_day_minutes = float(request.form['total_day_minutes'])
        total_day_calls = int(request.form['total_day_calls'])
        total_eve_minutes = float(request.form['total_eve_minutes'])
        total_eve_calls = int(request.form['total_eve_calls'])
        total_night_minutes = float(request.form['total_night_minutes'])
        total_night_calls = int(request.form['total_night_calls'])
        total_intl_minutes = float(request.form['total_intl_minutes'])
        total_intl_calls = int(request.form['total_intl_calls'])
        cust_serv_calls = int(request.form['cust_serv_calls'])

        # 🔹 Derived features
        total_minutes = total_day_minutes + total_eve_minutes + total_night_minutes
        total_calls = total_day_calls + total_eve_calls + total_night_calls
        intl_plan_day_minutes = intl_plan * total_day_minutes
        cust_serv_intl = cust_serv_calls * intl_plan
        total_calls_ratio = total_calls / (account_length + 1)

        # 🔹 Prepare features for model
        features_list = [
            account_length, area_code, intl_plan, vmail_plan, vmail_messages,
            total_day_minutes, total_day_calls, total_eve_minutes, total_eve_calls,
            total_night_minutes, total_night_calls, total_intl_minutes, total_intl_calls,
            cust_serv_calls, total_minutes, total_calls,
            intl_plan_day_minutes, cust_serv_intl, total_calls_ratio
        ]
        features_array = np.array([features_list])

        # 🔹 Prediction
        prediction = model.predict(features_array)[0]
        
        # 🔹 Soft probability approximation (votes/total_estimators)
        churn_percentage = model.estimators_[0].predict_proba(features_array)[:, 1][0]
        churn_percentage = round(churn_percentage * 100, 2)

        # 🔹 Get reasons & recommendations
        features_dict = {
            'International plan': intl_plan,
            'Customer service calls': cust_serv_calls,
            'intl_plan_day_minutes': intl_plan_day_minutes,
            'cust_serv_intl': cust_serv_intl,
            'total_calls_ratio': total_calls_ratio
        }
        reasons, recommendations = get_reason_and_recommendations(features_dict, prediction)

        return render_template('result.html',
                               prediction="Churn" if prediction == 1 else "No Churn",
                               probability=churn_percentage,
                               reasons=reasons,
                               recommendations=recommendations)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
