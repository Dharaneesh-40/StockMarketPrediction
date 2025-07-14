import numpy as np
from flask import Flask, request, jsonify
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

app = Flask(__name__)

# File paths to save model and scaler
model_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\stock_lstm_model.keras"
scaler_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\scaler.pkl"

# Helper function to calculate model accuracy
def calculate_accuracy(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    accuracy = 100 - mape
    return accuracy, mape

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Get stock symbol from the request
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        # Fetch historical stock data (5 years of data)
        data = yf.Ticker(symbol).history(period="5y")
        if data.empty:
            return jsonify({"error": f"No data available for {symbol}. Please check the symbol and try again."}), 404

        # Extract closing prices
        prices = data["Close"].values.reshape(-1, 1)

        # Normalize the data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Prepare sequences for LSTM model (60 days of past data for each prediction)
        X, y = [], []
        for i in range(60, len(prices_scaled)):
            X.append(prices_scaled[i - 60:i, 0])  # Past 60 days as features
            y.append(prices_scaled[i, 0])  # Next day's price as label

        X, y = np.array(X), np.array(y)

        # Split data into training and validation sets
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Reshape input for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(units=50),
            Dense(units=1)  # Output layer
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Validate the model
        y_pred_scaled = model.predict(X_val)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        # Calculate accuracy and MAPE
        accuracy, mape = calculate_accuracy(y_val_original, y_pred)

        # Save the model and scaler for future use
        model.save(model_path)  # Save the trained model
        joblib.dump(scaler, scaler_path)  # Save the scaler
        print("accuracy:",accuracy,"mape:",mape)
        # Return success message with training details
        return jsonify({
            "message": f"Training completed for symbol: {symbol}. Model and scaler saved successfully.",
            "accuracy": f"{accuracy:.2f}%",
            "mape": f"{mape:.2f}%"
        }), 200

    except Exception as e:
        # Handle exceptions and return error details
        return jsonify({"error": f"Error occurred during training: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
