"""
app.py — Flask REST API for Malicious URL Detection
Loads the trained scikit-learn model and serves predictions.

Setup:
    pip install flask flask-cors scikit-learn joblib pandas tldextract
    python train_model.py          # trains & saves model.pkl
    python app.py                  # starts server on http://localhost:5000

Endpoints:
    POST /predict   { "url": "https://..." }
    GET  /health
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import FEATURE_NAMES, extract_features
from train_model import build_feature_matrix, load_dataset, train_and_evaluate, save_artifacts

# App setup

app = Flask(__name__)
CORS(app)  # Allow React frontend on any port

model = None
feature_names = FEATURE_NAMES


def ensure_model():
    """Load model from disk, or train a quick demo model if none exists."""
    global model
    if model is not None:
        return

    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        print("Model loaded from model.pkl")
    else:
        print("No model.pkl found — training demo model now...")
        X, y = load_dataset(None)           # uses built-in demo data
        clf, name, _, feat_names = train_and_evaluate(X, y)
        save_artifacts(clf, feat_names, name)
        model = clf
        print("Demo model ready.")


# Routes

@app.route('/predict', methods=['POST'])
def predict():
    ensure_model()

    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'Request body must be JSON with a "url" field.'}), 400

    url = str(data['url']).strip()
    if not url:
        return jsonify({'error': 'URL cannot be empty.'}), 400

    try:
        features = extract_features(url)
        X = pd.DataFrame([features], columns=feature_names)

        prediction  = int(model.predict(X)[0])
        proba       = model.predict_proba(X)[0].tolist()   # [p_benign, p_malicious]

        label       = 'malicious' if prediction == 1 else 'benign'
        confidence  = round(max(proba) * 100, 2)
        malicious_p = round(proba[1] * 100, 2)
        benign_p    = round(proba[0] * 100, 2)

        # Build human-readable risk signals for the UI
        risk_signals = []
        if features['has_ip_address']:
            risk_signals.append('URL contains a raw IP address')
        if features['suspicious_tld']:
            risk_signals.append('Suspicious top-level domain detected')
        if features['has_suspicious_keyword']:
            risk_signals.append(f"Phishing keyword found in URL")
        if features['url_entropy'] > 4.5:
            risk_signals.append(f"High URL entropy ({features['url_entropy']:.2f}) — may be obfuscated")
        if features['num_subdomains'] > 3:
            risk_signals.append(f"Excessive subdomain depth ({features['num_subdomains']})")
        if features['has_at_symbol']:
            risk_signals.append('@ symbol detected — may be hiding true host')
        if features['num_hyphens'] > 4:
            risk_signals.append(f"Many hyphens ({features['num_hyphens']}) — common in spoofed domains")
        if not features['has_https']:
            risk_signals.append('Not using HTTPS')
        if features['num_redirects'] > 1:
            risk_signals.append('Multiple redirects detected')

        return jsonify({
            'url':               url,
            'prediction':        label,
            'confidence':        confidence,
            'malicious_prob':    malicious_p,
            'benign_prob':       benign_p,
            'risk_signals':      risk_signals,
            'features':          features,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    ensure_model()
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None,
        'num_features': len(feature_names),
    })


@app.route('/features', methods=['GET'])
def list_features():
    return jsonify({'features': feature_names})


# Run

if __name__ == '__main__':
    ensure_model()
    print("\nServer running at http://localhost:5001")
    app.run(debug=True, port=5001)
