"""
scripts/des_login.py — Authenticate to DES/openEO

Supports multiple authentication methods:

1. OIDC device flow (recommended — stores refresh token for auto-renewal):
    python scripts/des_login.py --device

2. Basic Auth (tutorial-style — shared credentials):
    python scripts/des_login.py --basic --user testuser --password secretpassword

3. Access token (from Web Editor — expires in ~60 min, no auto-renewal):
    python scripts/des_login.py --token "your_access_token"

4. Test current authentication:
    python scripts/des_login.py --test
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPENEO_URL = "https://openeo.digitalearth.se"


def login_device():
    """Authenticate via OIDC device flow.

    This is the recommended method — it stores a refresh token that
    imint.fetch can use to automatically renew expired access tokens.
    No need to manually copy tokens from the Web Editor.

    Steps:
        1. A URL + code is printed to the terminal
        2. Open the URL in your browser and enter the code
        3. Log in with your EGI account
        4. The refresh token is stored automatically by openeo
    """
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)

    print("Starting OIDC device flow...")
    print("A browser window may open. If not, follow the URL printed below.")
    print()

    conn.authenticate_oidc_device(
        provider_id="egi",
        store_refresh_token=True,
    )

    # Verify
    collections = conn.list_collections()
    print()
    print(f"Authenticated! Found {len(collections)} collections.")
    print("Refresh token stored — fetch will auto-renew tokens from now on.")
    return conn


def login_basic(user: str, password: str):
    """Authenticate with HTTP Basic Auth (username + password).

    This is the method used in the DES community tutorials.
    Credentials can optionally be saved as environment variables
    DES_USER and DES_PASSWORD for use by imint.fetch.

    NOTE: Basic Auth uses shared credentials and is primarily intended
    for tutorials and development. Use --device for production.
    """
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_basic(username=user, password=password)

    # Verify
    collections = conn.list_collections()
    print(f"Authenticated via Basic Auth! Found {len(collections)} collections.")
    for c in collections:
        print(f"  - {c['id']}: {c.get('title', '')}")

    print()
    print("To use with imint.fetch, set environment variables:")
    print(f"  export DES_USER=\"{user}\"")
    print(f"  export DES_PASSWORD=\"{password}\"")
    print()
    print("TIP: Use --device instead for production (per-user OIDC, auto-renewing).")
    return conn


def login_token(token: str):
    """Authenticate with an EGI OIDC access token (e.g. from Web Editor).

    NOTE: Access tokens expire in ~60 min and cannot be auto-renewed.
    Use --device instead for persistent authentication.
    """
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_access_token(access_token=token, provider_id="egi")

    # Verify
    collections = conn.list_collections()
    print(f"Authenticated! Found {len(collections)} collections.")

    # Save token to file for reuse (short-lived)
    token_path = os.path.join(os.path.dirname(__file__), "..", ".des_token")
    with open(token_path, "w") as f:
        f.write(token)
    print(f"Token saved to .des_token (expires in ~60 min)")
    print("TIP: Use --device instead for persistent authentication.")
    return conn


def test_connection():
    """Test if we have valid authentication."""
    import openeo

    conn = openeo.connect(OPENEO_URL)

    # 1. Try refresh token (best — auto-renews)
    try:
        conn.authenticate_oidc_refresh_token(
            provider_id="egi",
        )
        collections = conn.list_collections()
        print(f"Authenticated via refresh token (auto-renewable).")
        print(f"  {len(collections)} collections available.")
        for c in collections:
            print(f"  - {c['id']}: {c.get('title', '')}")
        return conn
    except Exception:
        pass

    # 2. Try Basic Auth from environment variables
    des_user = os.environ.get("DES_USER")
    des_password = os.environ.get("DES_PASSWORD")
    if des_user and des_password:
        try:
            conn.authenticate_basic(username=des_user, password=des_password)
            collections = conn.list_collections()
            print(f"Authenticated via Basic Auth (DES_USER env var).")
            print(f"  {len(collections)} collections available.")
            for c in collections:
                print(f"  - {c['id']}: {c.get('title', '')}")
            return conn
        except Exception as e:
            print(f"  Basic Auth failed: {e}")

    # 3. Try saved access token
    token_path = os.path.join(os.path.dirname(__file__), "..", ".des_token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            token = f.read().strip()
        if token:
            try:
                conn.authenticate_oidc_access_token(
                    access_token=token, provider_id="egi"
                )
                collections = conn.list_collections()
                print(f"Authenticated via saved access token (short-lived).")
                print(f"  {len(collections)} collections available.")
                return conn
            except Exception as e:
                print(f"  Saved access token expired: {e}")

    print("Not authenticated. Run one of:")
    print("  python scripts/des_login.py --device    (recommended)")
    print("  python scripts/des_login.py --basic --user USER --password PASS")
    print("  python scripts/des_login.py --token YOUR_TOKEN")
    return None


def main():
    parser = argparse.ArgumentParser(description="DES/openEO authentication")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--device", action="store_true",
        help="OIDC device flow (recommended — stores refresh token)"
    )
    group.add_argument(
        "--basic", action="store_true",
        help="Basic Auth (tutorial-style, shared credentials)"
    )
    group.add_argument(
        "--token",
        help="Access token from Web Editor (short-lived, ~60 min)"
    )
    group.add_argument(
        "--test", action="store_true",
        help="Test current authentication"
    )

    parser.add_argument("--user", help="Username for --basic auth")
    parser.add_argument("--password", help="Password for --basic auth")

    args = parser.parse_args()

    if args.device:
        login_device()
    elif args.basic:
        if not args.user or not args.password:
            parser.error("--basic requires --user and --password")
        login_basic(args.user, args.password)
    elif args.token:
        login_token(args.token)
    elif args.test:
        test_connection()


if __name__ == "__main__":
    main()
