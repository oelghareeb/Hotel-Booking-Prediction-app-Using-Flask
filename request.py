import requests

# Update the endpoint URL based on your Flask application
url = 'http://localhost:5000/predict_api'

# Modify the input data to match features relevant to hotel bookings
input_data = {
    'lead-time': 50,
    'avg-price': 150,
    'special-requests': 2,
    'day': 15,
    'month': 8,
    'market-segment-online': 1,
    'market-segment-offline': 0
}

# Send a POST request to the Flask API with the modified input data
r = requests.post(url, json=input_data)

# Print the response
print(r.json())
