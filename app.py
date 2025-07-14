import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
from flask import Flask, request, render_template, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# Define paths for model and scaler
model_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\stock_lstm_model.keras"
scaler_path = r"C:\Users\deepa\OneDrive\Desktop\StockMarketPredictionSystem\model\scaler.pkl"

# Load the model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo", interval="1d")
    
    if hist.empty:
        return jsonify({"error": f"No data found for symbol: {symbol}"}), 404

    info = stock.info
    data = {
        "name": info.get("shortName", "N/A"),
        "open": float(hist['Open'].iloc[-1]) if len(hist) > 0 else "N/A",
        "high": float(hist['High'].iloc[-1]) if len(hist) > 0 else "N/A",
        "low": float(hist['Low'].iloc[-1]) if len(hist) > 0 else "N/A",
        "close": float(hist['Close'].iloc[-1]) if len(hist) > 0 else "N/A",
        "volume": int(hist['Volume'].iloc[-1]) if len(hist) > 0 else "N/A",
        "today_low": float(hist['Low'].min()) if len(hist) > 0 else "N/A",
        "today_high": float(hist['High'].max()) if len(hist) > 0 else "N/A",
        "prev_close": float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "shares_outstanding": info.get("sharesOutstanding", "N/A"),
        "dates": hist.index.strftime('%Y-%m-%d').tolist(),
        "prices": hist['Close'].tolist()
    }

    return jsonify(data)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        # Fetch historical stock data (5 years of data)
        data = yf.Ticker(symbol).history(period="5y")
        if data.empty:
            return jsonify({"error": f"No data available for {symbol}. Please check the symbol and try again."}), 404
        
        prices = data["Close"].values.reshape(-1, 1)

        # Scaling the data (Normalization)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Prepare data for LSTM (sequence of 60 days as features)
        X, y = [], []
        for i in range(60, len(prices_scaled)):
            X.append(prices_scaled[i-60:i, 0])  # Use the past 60 days as features
            y.append(prices_scaled[i, 0])  # The next day's closing price as the label

        X, y = np.array(X), np.array(y)

        # Reshaping X to be 3D [samples, time steps, features] for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model
        model.fit(X, y, epochs=20, batch_size=32)

        # Save the trained model and scaler for future use
        model.save(model_path)  # Save the LSTM model
        joblib.dump(scaler, scaler_path)  # Save the scaler

        return jsonify({"message": f"Training completed for symbol: {symbol}. Model and scaler saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Error occurred during training: {e}"}), 500

@app.route('/api/predict', methods=['GET'])
def predict_stock():
    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo", interval="1d")
    
    if hist.empty:
        return jsonify({"error": f"No historical data found for symbol: {symbol}"}), 404

    prices = hist['Close'].values.reshape(-1, 1)
    if len(prices) < 60:
        return jsonify({"error": "Insufficient data (less than 60 days) for prediction"}), 404

    prices_scaled = scaler.transform(prices)
    last_60_days = prices_scaled[-60:].reshape((1, 60, 1))

    predictions_scaled = []
    input_sequence = last_60_days

    for _ in range(30):
        predicted_scaled = model.predict(input_sequence, verbose=0)
        predicted_scaled = np.reshape(predicted_scaled, (1, 1, 1))
        input_sequence = np.append(input_sequence[:, 1:, :], predicted_scaled, axis=1)
        predictions_scaled.append(predicted_scaled[0, 0])

    predicted_prices = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]

    return jsonify(dict(zip(dates, predicted_prices.tolist())))

if __name__ == '__main__':
    app.run(debug=True)
