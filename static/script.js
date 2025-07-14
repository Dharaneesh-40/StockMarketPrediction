let actualChart, predictionChart;

async function fetchStockData() {
    const symbol = document.getElementById("stock-symbol").value;

    try {
        const response = await fetch(`/api/stock?symbol=${symbol}`);
        
        if (!response.ok) {
            // Check if response is not OK (status code not in the range 200-299)
            const errorData = await response.json();
            alert(errorData.error || "An error occurred while fetching the stock data.");
            return; // Exit the function after showing the error
        }

        const data = await response.json();
        displayStockInfo(data);
        plotActualGraph(data);
        fetchPredictionData(symbol);
    } catch (error) {
        console.error("Error fetching stock data:", error);
        alert("An error occurred while fetching the stock data. Please try again.");
    }
}


async function trainStockModel() {
    const symbol = document.getElementById("stock-symbol").value;

    // Show a loading message while training
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = "Training the model... Please wait.";

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol }) // Send stock symbol
        });

        const result = await response.json();
        if (response.ok) {
            statusDiv.innerHTML = "Model training completed successfully!";
            alert(result.message);
        } else {
            statusDiv.innerHTML = "Error occurred during training!";
            alert(result.error || "Model training failed."); // Error
        }
    } catch (error) {
        console.error("Error training model:", error);
        statusDiv.innerHTML = "Error occurred during training!";
        alert("An error occurred while training the model. Please try again.");
    }
}

function displayStockInfo(data) {
    const stockInfoDiv = document.getElementById("stock-data");
    stockInfoDiv.innerHTML = `
        <h2>${data.name || 'N/A'}</h2>
        <p>Open: ${data.open?.toFixed(2) || 'N/A'}</p>
        <p>High: ${data.high?.toFixed(2) || 'N/A'}</p>
        <p>Low: ${data.low?.toFixed(2) || 'N/A'}</p>
        <p>Close: ${data.close?.toFixed(2) || 'N/A'}</p>
        <p>Volume: ${data.volume || 'N/A'}</p>
        <p>Today's Low: ${data.today_low || 'N/A'}</p>
        <p>Today's High: ${data.today_high || 'N/A'}</p>
        <p>Previous Close: ${data.prev_close || 'N/A'}</p>
        <p>Market Cap: ${data.market_cap || 'N/A'}</p>
        <p>PE Ratio: ${data.pe_ratio || 'N/A'}</p>
        <p>Dividend Yield: ${data.dividend_yield || 'N/A'}</p>
        <p>Shares Outstanding: ${data.shares_outstanding || 'N/A'}</p>
    `;
}

function plotActualGraph(data) {
    const ctx = document.getElementById('actual-chart').getContext('2d');
    if (actualChart) actualChart.destroy();

    actualChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Actual Stock Prices',
                data: data.prices,
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Stock Price Data'
                }
            }
        }
    });
}

async function fetchPredictionData(symbol) {
    try {
        const response = await fetch(`/api/predict?symbol=${symbol}`);
        const data = await response.json();

        plotPredictionGraph(data);
    } catch (error) {
        console.error("Error fetching prediction data:", error);
        alert("An error occurred while fetching the prediction data. Please try again.");
    }
}

function plotPredictionGraph(predictedData) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    if (predictionChart) predictionChart.destroy();

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Object.keys(predictedData),
            datasets: [{
                label: 'Predicted Stock Prices',
                data: Object.values(predictedData),
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Predicted Stock Prices (Next 30 Days)'
                }
            }
        }
    });
}
