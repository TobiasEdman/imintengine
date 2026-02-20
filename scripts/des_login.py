"""
scripts/des_login.py — Authenticate to DES/openEO

Supports multiple authentication methods:

1. Bearer token (from Web Editor):
    python scripts/des_login.py --token "your_access_token"

2. OIDC device flow (opens browser):
    python scripts/des_login.py --device

3. Test current authentication:
    python scripts/des_login.py --test
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPENEO_URL = "https://openeo.digitalearth.se"


def login_token(token: str):
    """Authenticate with an EGI OIDC access token (e.g. from Web Editor).

    DES requires authenticate_oidc_access_token with provider_id='egi',
    not a plain bearer token.
    """
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")

    # Verify
    collections = conn.list_collections()
    print(f"Authenticated! Found {len(collections)} collections.")

    # Save token to file for reuse
    token_path = os.path.join(os.path.dirname(__file__), "..", ".des_token")
    with open(token_path, "w") as f:
        f.write(token)
    print(f"Token saved to .des_token (gitignored)")
    return conn


def login_device():
    """Authenticate via OIDC device flow."""
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)
    print("Starting OIDC device flow (check your browser)...")
    conn.authenticate_oidc(provider_id="keycloak")
    print("Authenticated!")
    return conn


def test_connection():
    """Test if we have valid authentication."""
    import openeo

    conn = openeo.connect(OPENEO_URL)

    # Try saved token first
    token_path = os.path.join(os.path.dirname(__file__), "..", ".des_token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            token = f.read().strip()
        if token:
            conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")
            try:
                collections = conn.list_collections()
                print(f"Authenticated via saved token. {len(collections)} collections available.")
                for c in collections:
                    print(f"  - {c['id']}: {c.get('title', '')}")
                return conn
            except Exception as e:
                print(f"Saved token expired or invalid: {e}")

    # Try cached OIDC token
    try:
        conn.authenticate_oidc(provider_id="keycloak")
        collections = conn.list_collections()
        print(f"Authenticated via cached OIDC token. {len(collections)} collections available.")
        return conn
    except Exception:
        pass

    print("Not authenticated. Use one of:")
    print("  python scripts/des_login.py --token YOUR_TOKEN")
    print("  python scripts/des_login.py --device")
    return None


def main():
    parser = argparse.ArgumentParser(description="DES/openEO authentication")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--token", help="Bearer token from Web Editor")
    group.add_argument("--device", action="store_true", help="OIDC device flow")
    group.add_argument("--test", action="store_true", help="Test current auth")
    args = parser.parse_args()

    if args.token:
        login_token(args.token)
    elif args.device:
        login_device()
    elif args.test:
        test_connection()


if __name__ == "__main__":
    main()
