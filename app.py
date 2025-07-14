import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
from flask import Flask, request, render_template, jsonify
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Define paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "model")
model_path = os.path.join(model_dir, "stock_lstm_model.keras")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load model and scaler if available
model = None
scaler = None
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol')
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo", interval="1d")

        if hist.empty:
            return jsonify({"error": f"No data found for symbol: {symbol}. Possible network issue or invalid symbol."}), 404

        info = stock.info

        data = {
            "name": info.get("shortName", "N/A"),
            "open": float(hist['Open'].iloc[-1]),
            "high": float(hist['High'].iloc[-1]),
            "low": float(hist['Low'].iloc[-1]),
            "close": float(hist['Close'].iloc[-1]),
            "volume": int(hist['Volume'].iloc[-1]),
            "today_low": float(hist['Low'].min()),
            "today_high": float(hist['High'].max()),
            "prev_close": float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "shares_outstanding": info.get("sharesOutstanding", "N/A"),
            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
            "prices": hist['Close'].tolist()
        }

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    global model, scaler

    try:
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        data = yf.Ticker(symbol).history(period="5y")
        if data.empty:
            return jsonify({"error": f"No data available for {symbol}. Possible network issue or invalid symbol."}), 404

        prices = data["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(60, len(prices_scaled)):
            X.append(prices_scaled[i-60:i, 0])
            y.append(prices_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(X, y, epochs=20, batch_size=32, verbose=1)

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        return jsonify({"message": f"Model trained and saved for {symbol}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['GET'])
def predict_stock():
    global model, scaler

    if model is None or scaler is None:
        return jsonify({"error": "Model is not trained yet. Please call /api/train first."}), 400

    symbol = request.args.get('symbol')
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo", interval="1d")

    if hist.empty:
        return jsonify({"error": f"No historical data found for symbol: {symbol}. Possible network issue or invalid symbol."}), 404

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

# Health check for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
