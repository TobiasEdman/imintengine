"""Lightweight environment configuration loader.

Loads environment-specific ``.env`` files from ``config/environments/``
based on the ``IMINT_ENV`` variable (``dev`` | ``test`` | ``prod``).

Priority (highest wins):
    1. CLI arguments (argparse in scripts)
    2. Shell environment variables already set
    3. Values from ``config/environments/{IMINT_ENV}.env``
    4. TrainingConfig dataclass defaults

Usage::

    from imint.config.env import load_env, get_env

    load_env()              # auto-detects from IMINT_ENV or defaults to 'dev'
    load_env('prod')        # force production environment
    env = get_env()         # returns 'dev', 'test', or 'prod'
"""

from __future__ import annotations

import os
from pathlib import Path

_VALID_ENVS = ("dev", "test", "prod")
_loaded = False


def load_env(env_name: str | None = None) -> str:
    """Load an environment config file into ``os.environ``.

    Args:
        env_name: Environment name.  If *None*, reads ``IMINT_ENV``
            from the current environment, defaulting to ``'dev'``.

    Returns:
        The resolved environment name.

    Raises:
        ValueError: If *env_name* is not one of ``dev``, ``test``, ``prod``.
        FileNotFoundError: If the ``.env`` file does not exist.
    """
    global _loaded

    if env_name is None:
        env_name = os.environ.get("IMINT_ENV", "dev")

    env_name = env_name.lower().strip()
    if env_name not in _VALID_ENVS:
        raise ValueError(
            f"Invalid IMINT_ENV={env_name!r}. Must be one of {_VALID_ENVS}"
        )

    # Find the env file — check project root and common locations
    env_file = _find_env_file(env_name)
    if env_file is None:
        # No file found — just set IMINT_ENV and return
        os.environ.setdefault("IMINT_ENV", env_name)
        _loaded = True
        return env_name

    # Parse and load (don't overwrite existing vars)
    _load_dotenv(env_file, override=False)
    os.environ["IMINT_ENV"] = env_name
    _loaded = True
    return env_name


def get_env() -> str:
    """Return the current environment name (``dev``, ``test``, or ``prod``)."""
    return os.environ.get("IMINT_ENV", "dev")


def is_loaded() -> bool:
    """Return True if :func:`load_env` has been called."""
    return _loaded


# ── Internal helpers ──────────────────────────────────────────────────


def _find_env_file(env_name: str) -> Path | None:
    """Locate ``config/environments/{env_name}.env``."""
    # Walk up from this file to find the project root
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "config" / "environments" / f"{env_name}.env"
        if candidate.exists():
            return candidate
        # Stop at .git boundary
        if (ancestor / ".git").exists():
            break
    return None


def _load_dotenv(path: Path, *, override: bool = False) -> None:
    """Parse a .env file and inject into ``os.environ``.

    Uses ``python-dotenv`` if available, otherwise falls back to a
    simple built-in parser (supports ``KEY=VALUE`` and comments).
    """
    try:
        from dotenv import load_dotenv as _dotenv_load
        _dotenv_load(path, override=override)
        return
    except ImportError:
        pass

    # Fallback: simple .env parser
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value
