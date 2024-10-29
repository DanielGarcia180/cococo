from flask import Flask, request, jsonify
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/predict_arima', methods=['POST'])
def predict_arima():
    data = request.json['data']
    
    if len(data) < 3:
        return jsonify({'error': 'ARIMA necesita al menos 3 puntos de datos.'}), 400

    # Ajustar el modelo ARIMA
    model = ARIMA(data, order=(5, 1, 0))  # (p, d, q) ARIMA parameters
    model_fit = model.fit()

    # Predecir el siguiente valor
    pred = model_fit.forecast(steps=1)[0]
    
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True)
