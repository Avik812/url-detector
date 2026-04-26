"""
app.py — Flask REST API for Malicious URL Detection
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from features import extract_features

app          = Flask(__name__)
model        = None
feature_names = []
model_name   = ''


def load_model():
    global model, feature_names, model_name
    if not os.path.exists('model.pkl'):
        raise FileNotFoundError('model.pkl not found — run train_model.py first')
    if not os.path.exists('model_meta.json'):
        raise FileNotFoundError('model_meta.json not found — run train_model.py first')
    model = joblib.load('model.pkl')
    with open('model_meta.json') as f:
        meta = json.load(f)
    feature_names = meta['feature_names']
    model_name    = meta['model_name']
    print(f'Model loaded: {model_name} ({len(feature_names)} features)')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': model_name,
        'features': len(feature_names)
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({'error': 'Send JSON with a "url" key'}), 400

    url = str(data['url']).strip()
    if not url:
        return jsonify({'error': 'url is empty'}), 400

    try:
        feats = extract_features(url)
        X = pd.DataFrame([feats], columns=feature_names)

        proba = model.predict_proba(X)[0].tolist()
        mal_p = proba[1]
        ben_p = proba[0]

        # Threshold: 0.75 — model must be 75% confident to flag as malicious
        # This reduces false positives on legitimate sites
        label = 'malicious' if mal_p >= 0.75 else 'benign'

        confidence = round(max(proba) * 100, 1)

        # Human-readable risk signals
        signals = []
        if feats['has_ip']:
            signals.append('Raw IP address used instead of domain name')
        if feats['suspicious_tld']:
            signals.append('Suspicious top-level domain detected')
        if feats['has_suspicious_keyword']:
            signals.append('Phishing keyword detected in URL')
        if feats['url_entropy'] > 4.5:
            signals.append(f"High URL entropy ({feats['url_entropy']:.2f}) — possible obfuscation")
        if feats['many_subdomains']:
            signals.append(f"Excessive subdomain depth ({feats['n_subdomains']} subdomains)")
        if feats['has_at']:
            signals.append('@ symbol found — may be hiding the real destination')
        if feats['domain_n_hyphens'] > 3:
            signals.append(f"Many hyphens in domain ({feats['domain_n_hyphens']})")
        if not feats['is_https']:
            signals.append('Not using HTTPS')
        if feats['has_hex_chars']:
            signals.append('Hex-encoded characters detected — possible obfuscation')
        if feats['long_domain']:
            signals.append('Unusually long domain name')

        return jsonify({
            'url':            url,
            'prediction':     label,
            'confidence':     confidence,
            'malicious_prob': round(mal_p * 100, 1),
            'benign_prob':    round(ben_p * 100, 1),
            'risk_signals':   signals,
            'features':       feats,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    print('Server running at http://localhost:5001')
    app.run(debug=False, port=5001)