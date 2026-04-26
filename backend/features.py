"""
features.py — URL Feature Extraction for Malicious URL Detector

Key design decision: ALL URLs are normalized to https:// before feature
extraction. This makes training data and inference data fully consistent.
The is_https flag is preserved separately before normalization.
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'secure', 'account', 'update', 'verify', 'bank',
    'paypal', 'password', 'confirm', 'ebay', 'amazon', 'support', 'billing',
    'alert', 'suspended', 'unusual', 'locked', 'wallet', 'bonus', 'free',
    'prize', 'winner', 'urgent', 'authenticate', 'webscr', 'cmd',
]

SUSPICIOUS_TLDS = {
    '.tk', '.xyz', '.ml', '.ga', '.cf', '.gq', '.pw', '.top',
    '.click', '.link', '.online', '.site', '.work', '.ru', '.cn',
    '.info', '.biz', '.cc'
}

TRUSTED_TLDS = {
    '.com', '.org', '.net', '.edu', '.gov', '.io',
    '.uk', '.ca', '.au', '.de', '.fr', '.jp'
}

IP_RE = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')


def entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def normalize_url(url):
    """
    Strip any existing protocol and re-add https:// uniformly.
    This ensures training URLs (no prefix) and inference URLs (https://)
    produce identical feature values.
    Returns: (normalized_url, is_https_flag)
    """
    url = url.strip()
    # Detect original scheme
    if url.startswith('https://'):
        is_https = 1
        bare = url[8:]
    elif url.startswith('http://'):
        is_https = 0
        bare = url[7:]
    else:
        # No prefix — treat as https for normalization
        is_https = 1
        bare = url
    # Rebuild with https:// so all URLs have the same structure
    normalized = 'https://' + bare
    return normalized, bare, is_https


def extract_features(raw_url: str) -> dict:
    normalized, bare, is_https = normalize_url(raw_url)

    try:
        parsed = urlparse(normalized)
    except Exception:
        parsed = urlparse('https://invalid.com')

    hostname = (parsed.hostname or '').lower()
    path     = parsed.path or ''
    query    = parsed.query or ''

    # TLD and subdomains
    parts      = hostname.split('.')
    tld        = ('.' + parts[-1]) if len(parts) >= 2 else ''
    subdomains = [p for p in parts[:-2] if p]

    # Use bare URL (no protocol) for char-level features
    # so training (no prefix) and inference (stripped prefix) match
    u = bare

    n = max(len(u), 1)

    f = {}

    # --- Length ---
    f['url_len']    = len(u)
    f['domain_len'] = len(hostname)
    f['path_len']   = len(path)
    f['query_len']  = len(query)

    # --- Char counts on bare URL ---
    f['n_dots']       = u.count('.')
    f['n_hyphens']    = u.count('-')
    f['n_slashes']    = u.count('/')
    f['n_question']   = u.count('?')
    f['n_equals']     = u.count('=')
    f['n_ampersand']  = u.count('&')
    f['n_percent']    = u.count('%')
    f['n_underscore'] = u.count('_')
    f['n_at']         = u.count('@')
    f['n_digits']     = sum(c.isdigit() for c in u)

    # --- Ratios ---
    f['digit_ratio']  = round(f['n_digits'] / n, 4)
    f['letter_ratio'] = round(sum(c.isalpha() for c in u) / n, 4)

    # --- Entropy ---
    f['url_entropy']    = round(entropy(u), 4)
    f['domain_entropy'] = round(entropy(hostname), 4)

    # --- Domain ---
    f['domain_n_digits']  = sum(c.isdigit() for c in hostname)
    f['domain_n_hyphens'] = hostname.count('-')
    f['n_subdomains']     = len(subdomains)
    f['max_sub_len']      = max((len(s) for s in subdomains), default=0)
    f['domain_has_digit'] = int(any(c.isdigit() for c in hostname))

    # --- Structural ---
    f['is_https']       = is_https
    f['has_port']       = int(parsed.port is not None)
    f['has_ip']         = int(bool(IP_RE.match(hostname)))
    f['has_at']         = int('@' in u)
    f['n_path_tokens']  = len([p for p in path.split('/') if p])
    f['n_query_params'] = len([p for p in query.split('&') if '=' in p])

    # --- TLD ---
    f['suspicious_tld'] = int(tld in SUSPICIOUS_TLDS)
    f['trusted_tld']    = int(tld in TRUSTED_TLDS)

    # --- Keywords ---
    low = u.lower()
    f['n_suspicious_keywords'] = sum(kw in low for kw in SUSPICIOUS_KEYWORDS)
    f['has_suspicious_keyword'] = int(f['n_suspicious_keywords'] > 0)

    # --- Derived risk signals ---
    f['long_domain']    = int(len(hostname) > 30)
    f['many_subdomains'] = int(len(subdomains) >= 3)
    f['has_hex_chars']  = int(bool(re.search(r'%[0-9a-fA-F]{2}', u)))

    return f


FEATURE_NAMES = list(extract_features('https://example.com').keys())