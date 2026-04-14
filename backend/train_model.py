"""
train_model.py — Train & Evaluate the Malicious URL Classifier
Uses scikit-learn with NLP-extracted features. Compares RF, SVM, and LR.

Dataset format (CSV):
    url,label
    https://google.com,0
    http://192.168.1.1/login.php,1
    ...
  label: 0 = benign, 1 = malicious

Recommended datasets:
  - PhishTank (https://www.phishtank.com/developer_info.php)
  - ISCX URL dataset (https://www.unb.ca/cic/datasets/url-2016.html)
  - Kaggle malicious URL dataset

Usage:
    python train_model.py --data dataset.csv
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
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, extract_features

# ---------------------------------------------------------------------------
# Demo data — small hardcoded set for quick testing (replace with real CSV)
# ---------------------------------------------------------------------------

DEMO_URLS = [
    # Benign (label = 0)
    ('https://www.google.com', 0),
    ('https://github.com/user/repo', 0),
    ('https://stackoverflow.com/questions/123', 0),
    ('https://en.wikipedia.org/wiki/Python', 0),
    ('https://www.apple.com/iphone', 0),
    ('https://www.amazon.com/dp/B08N5WRWNW', 0),
    ('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 0),
    ('https://docs.python.org/3/library/re.html', 0),
    ('https://pytorch.org/docs/stable/index.html', 0),
    ('https://www.bbc.com/news/world', 0),
    ('https://www.nytimes.com/section/technology', 0),
    ('https://reddit.com/r/python', 0),
    ('https://www.linkedin.com/jobs', 0),
    ('https://mail.google.com/mail/u/0/', 0),
    ('https://www.microsoft.com/en-us/windows', 0),

    # Malicious / Phishing (label = 1)
    ('http://192.168.1.1/login-paypal-secure.php', 1),
    ('http://secure-paypal-login.tk/verify?account=1', 1),
    ('http://amazon-account-update.xyz/signin', 1),
    ('http://login.ebay.com.phishing-site.cf/verify', 1),
    ('http://bankofamerica-secure.ml/account/locked', 1),
    ('http://www.paypal.com.alert-verify.pw/update', 1),
    ('http://192.0.2.1/admin/login?redirect=bank', 1),
    ('http://free-prize-winner-claim.gq/enter?user=you', 1),
    ('http://apple-id-verify-account.top/signin', 1),
    ('http://microsofft.com-support-alert.link/fix', 1),
    ('http://secure.login.paypal-billing-update.xyz', 1),
    ('http://unusual-signin-activity.ml/verify', 1),
    ('http://wallet-bonus-free.click/claim?ref=abc', 1),
    ('http://account-suspended-amazon.cf/restore', 1),
    ('http://authenticate.user-bank-alert.pw/update', 1),
    ('http://ebay.com.password-reset-secure.tk/confirm', 1),
]


def build_feature_matrix(urls: list[str]) -> pd.DataFrame:
    records = [extract_features(u) for u in urls]
    return pd.DataFrame(records, columns=FEATURE_NAMES)


def load_dataset(path = None) -> tuple[pd.DataFrame, np.ndarray]:
    if path and os.path.exists(path):
        print(f"Loading dataset from {path}")
        df = pd.read_csv(path)
        # Support both 'label' and 'type' column names
        label_col = 'label' if 'label' in df.columns else 'type'
        urls  = df['url'].tolist()
        y     = df[label_col].values.astype(int)
        X     = build_feature_matrix(urls)
        print(f"  Loaded {len(df)} rows — {y.sum()} malicious, {(y==0).sum()} benign")
    else:
        print("No dataset file found. Using built-in demo data.")
        urls, labels = zip(*DEMO_URLS)
        X = build_feature_matrix(list(urls))
        y = np.array(labels)
        print(f"  Demo set: {len(y)} rows — {y.sum()} malicious, {(y==0).sum()} benign")
    return X, y


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm',    SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr',     LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X: pd.DataFrame, y: np.ndarray) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models   = get_models()
    results  = {}
    best_clf = None
    best_acc = 0.0
    best_name = ''

    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cv_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

        results[name] = {'accuracy': acc, 'auc': auc, 'cv_accuracy': cv_acc}

        print(f"\n  {name}")
        print(f"    Test Accuracy : {acc*100:.2f}%")
        print(f"    ROC-AUC       : {auc*100:.2f}%" if auc else "    ROC-AUC       : N/A")
        print(f"    5-Fold CV Acc : {cv_acc*100:.2f}%")
        print(f"\n  {classification_report(y_test, y_pred, target_names=['Benign','Malicious'])}")

        if acc > best_acc:
            best_acc  = acc
            best_clf  = clf
            best_name = name

    print("="*60)
    print(f"  Best Model: {best_name} — {best_acc*100:.2f}% accuracy")
    print("="*60)

    return best_clf, best_name, results, X_train.columns.tolist()


def save_artifacts(clf, feature_names: list[str], model_name: str) -> None:
    joblib.dump(clf, 'model.pkl')
    print("\n  Saved: model.pkl")

    # Feature importances (only for tree-based models)
    raw_clf = clf[-1] if hasattr(clf, '__getitem__') else clf
    if hasattr(raw_clf, 'feature_importances_'):
        importance = dict(zip(feature_names, raw_clf.feature_importances_.tolist()))
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        with open('feature_importance.json', 'w') as f:
            json.dump(importance_sorted, f, indent=2)
        print("  Saved: feature_importance.json")

        print("\n  Top 10 Features:")
        for feat, score in list(importance_sorted.items())[:10]:
            bar = '█' * int(score * 100)
            print(f"    {feat:<30} {score:.4f}  {bar}")

    meta = {'model_name': model_name, 'feature_names': feature_names}
    with open('model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("  Saved: model_meta.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Malicious URL Classifier')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV dataset (url, label columns)')
    args = parser.parse_args()

    X, y = load_dataset(args.data)

    clf, model_name, results, feat_names = train_and_evaluate(X, y)

    save_artifacts(clf, feat_names, model_name)

    print("\n  Training complete. Run `python app.py` to start the API server.")
