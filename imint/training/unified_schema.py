"""
imint/training/unified_schema.py — thin shim over des-contracts.

**As of des-contracts v0.1.0 (agentic_workflow rollout plan W3.6, 2026-04)**,
the canonical definition of the 23-class schema lives in the `des-contracts`
package. This module re-exports the identical public API so existing
consumers inside ImintEngine don't need to change their imports.

Migration plan:
  - ImintEngine v0.2.0 (now): this shim. No deprecation warning yet.
  - ImintEngine v1.0.0 (later): shim removed; callers must import from
    `des_contracts` directly.

If you are writing new code, import from `des_contracts`:

    from des_contracts.schema import UNIFIED_CLASSES, SJV_TO_UNIFIED
    from des_contracts.schema.ops import merge_all, nmd19_to_unified

For backward compatibility, every previously-exposed symbol below is
still available from this module.
"""
from __future__ import annotations

# Re-export everything the old module exposed.
from des_contracts.schema import (  # noqa: F401
    HARVEST_CLASS,
    NUM_UNIFIED_CLASSES,
    SJV_TO_UNIFIED,
    UNIFIED_CLASS_NAMES,
    UNIFIED_CLASSES,
    UNIFIED_COLOR_LIST,
    UNIFIED_COLORS,
    class_name,
)
from des_contracts.schema import NMD_TO_UNIFIED as _NMD_TO_UNIFIED_DICT
from des_contracts.schema import SJV_DEFAULT as _SJV_DEFAULT  # noqa: F401

# numpy-backed ops. This module used to always import numpy (it defined
# _NMD19_TO_UNIFIED as a numpy array at module scope). We preserve that
# behaviour by importing .ops up front — identical semantics.
from des_contracts.schema.ops import (  # noqa: F401
    get_class_weights,
    merge_all,
    merge_nmd_sjv,
    nmd19_to_unified,
)

# Legacy alias preserved for backward-compat. The pre-W3 module exposed
# this as a module-scope numpy array at `_NMD19_TO_UNIFIED`. If any code
# reached into that internal, it still works.
import numpy as _np

_NMD19_TO_UNIFIED = _np.zeros(20, dtype=_np.uint8)
for _src, _dst in _NMD_TO_UNIFIED_DICT.items():
    _NMD19_TO_UNIFIED[_src] = _dst


def export_schema_json(path: str | None = None) -> dict:
    """Export unified schema as JSON for dashboards and viz scripts.

    Kept here (not moved to des-contracts) because it's JS/HTML/bash-facing —
    a dashboard artefact, not a contract. Returns dict; optionally writes
    to file. Eliminates hardcoded class names/colors in HTML/JS/bash.

    Args:
        path: Optional file path to write JSON. If None, just returns dict.

    Returns:
        Schema dict with class_names, colors_rgb, colors_css, num_classes.
    """
    import json

    schema = {
        "num_classes": NUM_UNIFIED_CLASSES,
        "class_names": UNIFIED_CLASS_NAMES,
        "colors_rgb": [list(UNIFIED_COLORS[i]) for i in range(NUM_UNIFIED_CLASSES)],
        "colors_css": {
            UNIFIED_CLASS_NAMES[i]: (
                f"rgb({UNIFIED_COLORS[i][0]},"
                f"{UNIFIED_COLORS[i][1]},"
                f"{UNIFIED_COLORS[i][2]})"
            )
            for i in range(NUM_UNIFIED_CLASSES)
        },
    }

    if path:
        with open(path, "w") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

    return schema
