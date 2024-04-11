from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
def load_model(filepath):
    with open(filepath + '.json', 'r') as f:
        model_config = json.load(f)
    model = model_from_json(model_config)
    model.load_weights(filepath + '.weights.h5')
    return model

model = load_model('model_config.json')  # Adjust the file path as needed

# Function to make predictions
def make_predictions(month, day, year, pred_days):
    # Preprocess user inputs
    df=pd.read_csv('C:/Users/Dhruva/OneDrive/Desktop/Final Year/code playground/BTC-USD.csv')
    closedf = df[['Date','Close']]
    del closedf['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
    time_stamp=15
    input_date = datetime(year, month, day)
    start_date = input_date.strftime('%Y-%m-%d')
    
    # Load the trained LSTM model and scaler
    model = load_model('model_config.json')
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Replace 'your_model_weights.h5' with the actual path to your model weights file
    
    
    # Preprocess the input date to match the format expected by the model
    input_date_index = df[df['Date'] == start_date].index.values[0]
    input_sequence = closedf[input_date_index:input_date_index+time_stamp].reshape(1, -1)
    
    # Use the model to predict Bitcoin close prices for the specified number of days
    lst_output = []
    for i in range(pred_days):
        if len(input_sequence[0]) > time_stamp:
            x_input = np.array(input_sequence[0][1:]).reshape(1, -1)
            x_input = x_input.reshape((1, 1))
            yhat = model.predict(x_input, verbose=0)
            input_sequence[0] = np.append(input_sequence[0][1:], yhat[0])
            lst_output.append(yhat[0][0])
        else:
            x_input = input_sequence.reshape((1,-1 1))
            yhat = model.predict(x_input, verbose=0)
            input_sequence[0] = np.append(input_sequence[0], yhat[0])
            lst_output.append(yhat[0][0])
    
    # Inverse transform the predicted values using the scaler
    predicted_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten().tolist()
    
    return predicted_values
    
# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the request
    request_data = request.get_json()
    month = int(request_data['month'])
    day = int(request_data['day'])
    year = int(request_data['year'])
    pred_days = int(request_data['pred_days'])
    
    # Make predictions
    predicted_values = make_predictions(month, day, year, pred_days)
    
    # Prepare response data
    response_data = {
        'predicted_values': predicted_values,
        'start_date': f"{year}-{month}-{day}",
        'pred_days': pred_days
    }
    
    # Return the response as JSON
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
