"""
features.py — NLP-based URL Feature Extraction
Extracts structured features from raw URLs for ML classification.
"""

import re
import math
from urllib.parse import urlparse

# Constants

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'secure', 'account', 'update', 'verify', 'bank',
    'paypal', 'password', 'confirm', 'ebay', 'amazon', 'support', 'billing',
    'alert', 'suspended', 'unusual', 'locked', 'unauthorized', 'authenticate',
    'wallet', 'bonus', 'free', 'prize', 'winner', 'click', 'urgent',
]

TRUSTED_TLDS = {'.com', '.org', '.net', '.edu', '.gov', '.io', '.co.uk'}
SUSPICIOUS_TLDS = {'.tk', '.xyz', '.ml', '.ga', '.cf', '.gq', '.pw',
                   '.top', '.click', '.link', '.online', '.site', '.work'}

IP_PATTERN = re.compile(
    r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])'
)

# Helpers

def shannon_entropy(s: str) -> float:
    """Shannon entropy — high entropy suggests random/obfuscated strings."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def get_tld(hostname: str) -> str:
    parts = hostname.split('.')
    return '.' + parts[-1] if parts else ''


def get_subdomain_count(hostname: str) -> int:
    """Count subdomains (everything left of the registered domain)."""
    parts = hostname.split('.')
    # e.g. secure.login.paypal.com → 2 subdomains
    return max(0, len(parts) - 2)


# Main extractor

def extract_features(url: str) -> dict:
    """
    Return a flat dict of numeric/boolean features for a given URL.
    All features are interpretable — no black-box embeddings.
    """
    # Normalise
    if not url.startswith(('http://', 'https://')):
        url_for_parse = 'http://' + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)
    hostname = parsed.hostname or ''
    path     = parsed.path or ''
    query    = parsed.query or ''
    full     = url  # use original for char-level features

    tld = get_tld(hostname)

    features = {
        # ── Length features ──────────────────────────────────────────────
        'url_length':        len(full),
        'domain_length':     len(hostname),
        'path_length':       len(path),
        'query_length':      len(query),

        # ── Structural features ──────────────────────────────────────────
        'num_subdomains':    get_subdomain_count(hostname),
        'has_ip_address':    int(bool(IP_PATTERN.search(hostname))),
        'has_https':         int(parsed.scheme == 'https'),
        'has_port':          int(bool(parsed.port)),
        'num_redirects':     full.count('//') - 1,          # extra // → redirect
        'has_at_symbol':     int('@' in full),               # used to obscure real host

        # ── Special character counts (NLP character-level) ───────────────
        'num_dots':          full.count('.'),
        'num_hyphens':       full.count('-'),
        'num_underscores':   full.count('_'),
        'num_slashes':       full.count('/'),
        'num_question':      full.count('?'),
        'num_ampersand':     full.count('&'),
        'num_equals':        full.count('='),
        'num_percent':       full.count('%'),                # URL encoding / obfuscation
        'num_tilde':         full.count('~'),

        # ── Character ratio features (NLP-style token analysis) ──────────
        'digit_ratio':       sum(c.isdigit() for c in full) / max(len(full), 1),
        'letter_ratio':      sum(c.isalpha() for c in full) / max(len(full), 1),
        'special_char_ratio': sum(not c.isalnum() and c not in '/:.' for c in full) / max(len(full), 1),

        # ── Entropy (information-theoretic) ─────────────────────────────
        'url_entropy':       round(shannon_entropy(full), 4),
        'domain_entropy':    round(shannon_entropy(hostname), 4),

        # ── NLP keyword features ─────────────────────────────────────────
        'suspicious_keyword_count': sum(kw in full.lower() for kw in SUSPICIOUS_KEYWORDS),
        'has_suspicious_keyword':   int(any(kw in full.lower() for kw in SUSPICIOUS_KEYWORDS)),

        # ── TLD features ─────────────────────────────────────────────────
        'suspicious_tld':    int(tld in SUSPICIOUS_TLDS),
        'trusted_tld':       int(tld in TRUSTED_TLDS),

        # ── Domain-level digit density ────────────────────────────────────
        'domain_digit_count': sum(c.isdigit() for c in hostname),
        'domain_hyphen_count': hostname.count('-'),
    }

    return features


# Feature names (ordered list for DataFrame columns)

FEATURE_NAMES = list(extract_features('http://example.com').keys())
