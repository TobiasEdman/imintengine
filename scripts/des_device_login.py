"""
scripts/des_device_login.py — OIDC device flow login with auto browser open.

Run this script, it will:
1. Open your browser to the EGI login page
2. Wait for you to log in
3. Store refresh token for auto-renewal

Usage:
    python scripts/des_device_login.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import webbrowser
from urllib.request import urlopen, Request
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPENEO_URL = "https://openeo.digitalearth.se"


def main():
    import openeo

    print(f"Connecting to {OPENEO_URL}...")
    conn = openeo.connect(OPENEO_URL)

    # Get OIDC provider info
    well_known_url = "https://aai-demo.egi.eu/auth/realms/egi/.well-known/openid-configuration"
    oidc_config = json.loads(urlopen(well_known_url).read())
    device_endpoint = oidc_config["device_authorization_endpoint"]
    token_endpoint = oidc_config["token_endpoint"]

    client_id = "des-openeo-api"

    # Step 1: Request device code
    data = urlencode({
        "client_id": client_id,
        "scope": "openid offline_access eduperson_entitlement voperson_id profile email",
    }).encode()
    req = Request(device_endpoint, data=data)
    resp = json.loads(urlopen(req).read())

    device_code = resp["device_code"]
    user_code = resp["user_code"]
    verification_url = resp.get("verification_uri_complete", resp["verification_uri"])
    interval = resp.get("interval", 5)

    print(f"\n{'='*60}")
    print(f"  Code: {user_code}")
    print(f"  URL:  {verification_url}")
    print(f"{'='*60}")
    print(f"\nOpening browser...")
    webbrowser.open(verification_url)
    print("Log in and approve access. Waiting...\n")

    # Step 2: Poll for token
    max_wait = 300
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(interval)
        try:
            token_data = urlencode({
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            }).encode()
            token_req = Request(token_endpoint, data=token_data)
            token_resp = json.loads(urlopen(token_req).read())

            if "access_token" in token_resp:
                access_token = token_resp["access_token"]
                refresh_token = token_resp.get("refresh_token")

                # Verify with DES
                conn.authenticate_oidc_access_token(
                    access_token=access_token, provider_id="egi"
                )
                collections = conn.list_collections()
                print(f"\nAuthenticated! {len(collections)} collections available.")

                # Store refresh token via openeo's built-in mechanism
                if refresh_token:
                    # Save refresh token using openeo's token storage
                    from openeo.rest.auth.config import RefreshTokenStore
                    store = RefreshTokenStore()
                    store.set_refresh_token(
                        issuer="https://aai-demo.egi.eu/auth/realms/egi",
                        client_id=client_id,
                        refresh_token=refresh_token,
                    )
                    print("Refresh token stored — auto-renewal will work from now on.")
                    print("You won't need to log in again (until refresh token expires).")

                # Also save access token to .des_token as fallback
                token_path = os.path.join(
                    os.path.dirname(__file__), "..", ".des_token"
                )
                with open(token_path, "w") as f:
                    f.write(access_token)
                print(f"Access token saved to .des_token (short-lived backup)")

                return

        except Exception as e:
            err = str(e)
            if "authorization_pending" in err or "slow_down" in err:
                elapsed = int(time.time() - start)
                remaining = max_wait - elapsed
                print(f"  Waiting for login... ({remaining}s remaining)", end="\r")
                continue
            elif "expired_token" in err:
                print("\nDevice code expired. Please run again.")
                sys.exit(1)
            else:
                # Might be an HTTP error with authorization_pending in body
                if hasattr(e, 'read'):
                    body = e.read().decode()
                    if "authorization_pending" in body or "slow_down" in body:
                        continue
                    print(f"\nUnexpected error: {body}")
                else:
                    elapsed = int(time.time() - start)
                    remaining = max_wait - elapsed
                    print(f"  Waiting for login... ({remaining}s remaining)", end="\r")
                    continue

    print("\nTimeout waiting for login. Please try again.")
    sys.exit(1)


if __name__ == "__main__":
    main()
