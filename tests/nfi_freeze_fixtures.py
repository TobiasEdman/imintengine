"""Complete synthetic manifest structure; never substitutes for runtime verification."""
from copy import deepcopy
from pathlib import Path
import sys
from imint.eval.fieldtruth import NFI_CELLS, NFI_KEY, NFI_PROTOCOL, NFI_SOURCE_FILES, sha256_file


def complete_manifest(manifest):
    root = Path(__file__).resolve().parents[1]
    manifest.update(identity=NFI_KEY, protocol=deepcopy(NFI_PROTOCOL),
                    git_sha="a" * 40, runtime_image="registry/test@sha256:" + "b" * 64,
                    source_sha256={n: sha256_file(root / n) for n in NFI_SOURCE_FILES})
    cells = manifest.get("cells", {})
    template = deepcopy(next(iter(cells.values()), {}))
    manifest["cells"] = {cell: deepcopy(cells.get(cell, template)) for cell in sorted(NFI_CELLS)}
    manifest.setdefault("baselines", {"NMD2023": {}})
    manifest["preparation_runtime"] = {
        "source": {"git_sha": manifest["git_sha"], "payload_sha256": "c" * 64},
        "image": {"ref": manifest["runtime_image"]},
        "runtime_manifest": {"path": "/fixture/runtime.json", "sha256": "d" * 64},
        "environments": {name: {"python": {"path": sys.executable}} for name in ("model", "scoring")},
    }
    return manifest
