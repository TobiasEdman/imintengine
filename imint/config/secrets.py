"""Centralised secrets loader with fail-fast validation.

Purpose
-------
ImintEngine historically read credentials from four separate files:
    .env                       — DES password, CDSE client id/secret
    .cdse_credentials          — CDSE refresh token cache
    .des_token.bak             — DES token backup
    .skg_endpoints             — Skogsstyrelsen endpoint URLs
    config/environments/*.env  — per-environment overrides

This module consolidates loading into a single, testable entry point that:

1. **Delegates env-file parsing** to ``imint.config.env.load_env()`` so the
   existing ``IMINT_ENV`` mechanism keeps working.
2. **Validates that required credentials are present** at startup and fails
   loudly with a helpful error (``SecretMissingError``) listing the missing
   keys and where they should come from.
3. **Redacts secret values in exceptions** — never prints a real token or
   password in a traceback.
4. **Treats ``config/environments/<env>.env`` as the source of truth** for
   local development; the loose root-level ``.env``, ``.cdse_credentials``,
   ``.des_token.bak``, and ``.skg_endpoints`` files are legacy and are NOT
   read by this module. Migration guidance at the bottom of this file.

Usage
-----

    from imint.config.secrets import load_secrets, Secrets

    # Fail-fast at module import time
    secrets: Secrets = load_secrets()
    resp = requests.get(url, auth=(secrets.DES_USER, secrets.DES_PASSWORD))

Rationale
---------
Per ``agentic_workflow/docs/rollout_plan.md`` Wave 1.2. The plaintext
credentials themselves must still be rotated by the human operator — this
module only addresses the *scatter* and *silent fallback* problems, not
the credentials themselves.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, Optional

from imint.config.env import get_env, load_env


class SecretMissingError(RuntimeError):
    """Raised when one or more required secrets are absent from the env.

    The exception message lists which keys were missing and where they
    should come from. Secret *values* are never included.
    """


# ── Required-secret manifest ────────────────────────────────────────────────
#
# Each entry is (env_var_name, human-readable source). Add entries here as
# new integrations land. Only add a secret as REQUIRED if the module genuinely
# cannot operate without it.
REQUIRED_SECRETS: tuple[tuple[str, str], ...] = (
    ("DES_USER",           "config/environments/<env>.env  or  shell env"),
    ("DES_PASSWORD",       "config/environments/<env>.env  or  shell env"),
    ("CDSE_CLIENT_ID",     "config/environments/<env>.env  or  shell env"),
    ("CDSE_CLIENT_SECRET", "config/environments/<env>.env  or  shell env"),
)

# Optional secrets — loaded if present, None otherwise. Document them here so
# consumers know what to look for.
OPTIONAL_SECRETS: tuple[tuple[str, str], ...] = (
    ("DES_TOKEN",               "set by DES login flow; cached in memory"),
    ("SKG_ENDPOINT_SJOKORT",    "set in <env>.env if S-57 fetch is needed"),
    ("SKG_ENDPOINT_AVVERKNING", "set in <env>.env if SKS fetch is needed"),
    ("COLONYOS_API_KEY",        "required only for ColonyOS submissions"),
    ("COLONYOS_PROJECT",        "required only for ColonyOS submissions"),
)

# Pattern used to redact anything that *looks* like a credential value in
# exception messages. Matches long opaque strings. Never perfect; paired with
# a "never stringify the Secrets dataclass" convention.
_REDACT_RE = re.compile(r"[A-Za-z0-9_\-+/=]{12,}")


@dataclass
class Secrets:
    """Typed container for loaded secrets.

    Required fields (must be non-empty when returned by ``load_secrets``):
        DES_USER, DES_PASSWORD, CDSE_CLIENT_ID, CDSE_CLIENT_SECRET

    Optional fields default to ``None``.

    Do NOT ``print`` or ``repr`` a ``Secrets`` instance in logs —
    ``__repr__`` is overridden to redact all values, but leaks via f-strings
    on individual fields are still possible. Use attribute access, not
    serialisation.
    """

    DES_USER: str
    DES_PASSWORD: str
    CDSE_CLIENT_ID: str
    CDSE_CLIENT_SECRET: str

    # Optional — default None
    DES_TOKEN: Optional[str] = None
    SKG_ENDPOINT_SJOKORT: Optional[str] = None
    SKG_ENDPOINT_AVVERKNING: Optional[str] = None
    COLONYOS_API_KEY: Optional[str] = None
    COLONYOS_PROJECT: Optional[str] = None

    # Metadata — not a secret
    env_name: str = field(default="unknown")

    def __repr__(self) -> str:  # noqa: D401 — brief is fine
        """Redacted repr. Never prints real credential values."""
        populated = [f.name for f in fields(self) if getattr(self, f.name)]
        return f"Secrets(env={self.env_name!r}, populated={populated})"

    # Convenience — used by HTTP clients. Keeps the Bearer-header convention
    # out of consumer code so we can swap auth schemes in one place later.
    def des_bearer(self) -> Optional[str]:
        """Return ``Bearer <token>`` if DES_TOKEN is set, else None."""
        return f"Bearer {self.DES_TOKEN}" if self.DES_TOKEN else None


# ── Loader ──────────────────────────────────────────────────────────────────

def _redact(s: str) -> str:
    """Replace anything that looks like a credential with ``***`` in-place."""
    return _REDACT_RE.sub("***", s)


def _collect(env: Dict[str, str]) -> Secrets:
    """Build a Secrets instance from a dict-like environment."""
    missing: list[tuple[str, str]] = []
    required: Dict[str, str] = {}
    for key, source in REQUIRED_SECRETS:
        val = env.get(key, "").strip()
        if not val:
            missing.append((key, source))
        else:
            required[key] = val

    if missing:
        bullets = "\n".join(f"  - {k}  (expected source: {src})" for k, src in missing)
        raise SecretMissingError(
            "ImintEngine cannot start — the following required secrets are missing:\n"
            f"{bullets}\n\n"
            "To fix this for local development, copy "
            "config/environments/template.env to config/environments/dev.env "
            "and fill in the values, then run with IMINT_ENV=dev. "
            "For production, set the values as real environment variables or "
            "mount them as Kubernetes Secrets.\n"
        )

    optional: Dict[str, Optional[str]] = {}
    for key, _source in OPTIONAL_SECRETS:
        val = env.get(key, "").strip()
        optional[key] = val if val else None

    return Secrets(
        env_name=get_env() or "unknown",
        **required,
        **optional,
    )


def load_secrets(env_name: Optional[str] = None) -> Secrets:
    """Load + validate secrets. Returns a populated ``Secrets`` instance.

    Args:
        env_name: ``dev`` / ``test`` / ``prod``. If None, uses ``IMINT_ENV``
            or defaults to ``'dev'``.

    Raises:
        SecretMissingError: If any required secret is missing. Exception
            message enumerates missing keys and their expected sources.

    Notes:
        This function is idempotent — calling it multiple times in the same
        process re-reads from ``os.environ`` but does not re-parse the env
        file (that's handled by ``load_env`` which caches).

        Secret values are never logged or printed by this module.
    """
    load_env(env_name)
    return _collect(dict(os.environ))


# ── Migration guidance (read this comment, do not delete) ──────────────────
#
# LEGACY FILES still present in the repo root as of 2026-04-24:
#
#     .env                -> migrate values to config/environments/dev.env
#     .cdse_credentials   -> contains CDSE refresh token cache, managed by
#                            imint.fetch code; NOT consumed by this module
#     .des_token.bak      -> stale DES token backup; delete after rotation
#     .skg_endpoints      -> migrate SKG_ENDPOINT_* values to <env>.env
#     .env.colonyos       -> migrate COLONYOS_* values to <env>.env
#
# To complete the consolidation:
#   1. Rotate any secrets that ever shipped to git.
#   2. Copy each value from the legacy file into config/environments/<env>.env.
#   3. Delete the legacy file.
#   4. Verify .gitignore still covers config/environments/*.env (it should).
#
# This module intentionally does NOT read the legacy files. New code should
# call load_secrets() and nothing else.
