from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and feature scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/registrazione', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        lead_time = int(request.form['lead-time'])
        avg_price = int(request.form['avg-price'])
        special_requests = int(request.form['special-requests'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        market_segment_online = 1 if 'market-segment-online' in request.form else 0
        market_segment_offline = 1 if 'market-segment-offline' in request.form else 0

        # Preprocess the user input
        user_input = pd.DataFrame({
            'lead time': [lead_time],
            'average price': [avg_price],
            'special requests': [special_requests],
            'day': [day],
            'month': [month],
            'market segment type_Online': [market_segment_online],
            'market segment type_Offline': [market_segment_offline]
        })

        # Scale the features
        user_input_scaled = scaler.transform(user_input)

        # Make predictions
        prediction = model.predict(user_input_scaled)

        # Display prediction result
        if prediction == 0:
            result = 'Booking Not Cancelled'
        else:
            result = 'Booking Cancelled'

        # Return the result to the client
        return render_template('result.html', prediction=result)

    # If it's a GET request, render the same page (redirects to the home page)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
