"""
train_model.py — Train Malicious URL Classifier

Usage:
    python train_model.py --data data/malicious_phish.csv
    python train_model.py   # uses demo data
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_features

# ---------------------------------------------------------------------------
# Demo URLs — used when no CSV is provided
# ---------------------------------------------------------------------------

DEMO = [
    ('https://google.com/search?q=python', 0),
    ('https://github.com/user/repo', 0),
    ('https://stackoverflow.com/questions/123', 0),
    ('https://en.wikipedia.org/wiki/Python', 0),
    ('https://apple.com/iphone', 0),
    ('https://amazon.com/dp/B08N5WRWNW', 0),
    ('https://youtube.com/watch?v=dQw4w9WgXcQ', 0),
    ('https://docs.python.org/3/library/re.html', 0),
    ('https://bbc.com/news/world', 0),
    ('https://reddit.com/r/python', 0),
    ('https://linkedin.com/jobs', 0),
    ('https://microsoft.com/en-us/windows', 0),
    ('https://nytimes.com/section/technology', 0),
    ('https://psu.instructure.com/', 0),
    ('https://mail.google.com/mail/u/0/', 0),
    ('http://192.168.1.1/login-paypal-secure.php', 1),
    ('http://secure-paypal-login.tk/verify?account=1', 1),
    ('http://amazon-account-update.xyz/signin', 1),
    ('http://login.ebay.com.phishing-site.cf/verify', 1),
    ('http://bankofamerica-secure.ml/account/locked', 1),
    ('http://paypal.com.alert-verify.pw/update', 1),
    ('http://free-prize-winner-claim.gq/enter?user=you', 1),
    ('http://apple-id-verify-account.top/signin', 1),
    ('http://secure.login.paypal-billing-update.xyz', 1),
    ('http://unusual-signin-activity.ml/verify', 1),
    ('http://wallet-bonus-free.click/claim?ref=abc', 1),
    ('http://account-suspended-amazon.cf/restore', 1),
    ('http://authenticate.user-bank-alert.pw/update', 1),
    ('http://ebay.com.password-reset-secure.tk/confirm', 1),
]


def build_X(urls):
    records = []
    for u in urls:
        try:
            records.append(extract_features(u))
        except Exception:
            records.append({k: 0 for k in FEATURE_NAMES})
    return pd.DataFrame(records, columns=FEATURE_NAMES)


def load_data(path=None):
    if path and os.path.exists(path):
        print(f'Loading {path} ...')
        df = pd.read_csv(path)
        label_col = 'label' if 'label' in df.columns else 'type'
        urls = df['url'].tolist()
        if df[label_col].dtype == object:
            y = (df[label_col] != 'benign').astype(int).values
        else:
            y = df[label_col].values.astype(int)
        X = build_X(urls)
        print(f'  {len(y):,} rows — {y.sum():,} malicious / {(y==0).sum():,} benign')
        return X, y
    else:
        print('No dataset — using built-in demo data')
        urls, labels = zip(*DEMO)
        X = build_X(list(urls))
        y = np.array(labels)
        print(f'  {len(y)} rows — {y.sum()} malicious / {(y==0).sum()} benign')
        return X, y


def get_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': Pipeline([
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(
                C=1.0, max_iter=1000,
                class_weight='balanced',
                random_state=42
            )),
        ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }


def run(path=None):
    X, y = load_data(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_clf, best_name, best_f1 = None, '', 0.0

    print('\n' + '='*65)
    print('  MODEL COMPARISON')
    print('='*65)

    for name, clf in get_models().items():
        print(f'\n  Training {name}...')
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv  = cross_val_score(clf, X, y, cv=5, scoring='f1').mean()
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        print(f'    Accuracy  : {acc*100:.2f}%')
        print(f'    ROC-AUC   : {auc*100:.2f}%')
        print(f'    Macro F1  : {f1*100:.2f}%')
        print(f'    CV F1     : {cv*100:.2f}%')
        print(classification_report(
            y_test, y_pred,
            target_names=['Benign', 'Malicious'], digits=3
        ))

        if f1 > best_f1:
            best_f1, best_clf, best_name = f1, clf, name

    print('='*65)
    print(f'  Best Model: {best_name} — F1 {best_f1*100:.2f}%')
    print('='*65)

    # Save model
    joblib.dump(best_clf, 'model.pkl')

    meta = {
        'model_name': best_name,
        'feature_names': X_train.columns.tolist()
    }
    with open('model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    raw = best_clf[-1] if hasattr(best_clf, '__getitem__') else best_clf
    if hasattr(raw, 'feature_importances_'):
        imp = dict(zip(X_train.columns, raw.feature_importances_))
        imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
        with open('feature_importance.json', 'w') as f:
            json.dump(imp, f, indent=2)
        print('\n  Top 10 Features:')
        for feat, score in list(imp.items())[:10]:
            bar = '█' * int(score * 80)
            print(f'    {feat:<28} {score:.4f}  {bar}')

    print('\n  Saved: model.pkl  model_meta.json  feature_importance.json')
    print('  Run: python app.py\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV with url and label/type columns')
    args = parser.parse_args()
    run(args.data)