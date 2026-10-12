#!/usr/bin/env python3
"""Explain every unmatched detection of a compare_to_reference.py result (dossier tables).

Re-runs the detector WITHOUT the 0.7 cut on every patch that contains the
point, in both tilings, and reports the best score within 10 px:
  "ai2" side:  AI2's materialised input with the reference-window tiling;
  "port" side: the port input with the AOI's own tiling (run C), or with
               ``--run-b`` cut to AI2's window and masked to AI2's coverage.

    python scripts/marine/s1_port/diag_unmatched.py cmp_run_c.json --record <record> \\
        --port-input <cache>/s1_vessel_inputs/2026-06-28_<key>.tif --out diag_run_c.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmp", type=rec.Path, help="compare_to_reference.py output")
    rec.add_record_arg(ap)
    ap.add_argument("--port-input", type=rec.Path, required=True)
    ap.add_argument("--run-b", action="store_true")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import torch
    from pyproj import Transformer

    from imint.analyzers import ai2_vessels_s1 as s1

    ai2_stack, (c0, r0, _, _) = rec.read_ai2_stack(a.record)
    port, (pc0, pr0), _ = rec.read_port_input(a.port_input)
    if a.run_b:
        sub = port[:, r0 - pr0:r0 - pr0 + ai2_stack.shape[1], c0 - pc0:c0 - pc0 + ai2_stack.shape[2]]
        port, (pc0, pr0) = np.where(ai2_stack > 0, sub, 0).astype(np.uint16), (c0, r0)
    sides = {"ai2": (ai2_stack, (c0, r0)), "port": (port, (pc0, pr0))}
    model = s1.load_detector(s1.DEFAULT_DETECT_CKPT)
    to_utm = Transformer.from_crs(4326, 32632, always_xy=True).transform

    def patch(arr: np.ndarray, pc: int, pr: int) -> torch.Tensor:
        out = np.zeros((2, s1.PATCH_SIZE, s1.PATCH_SIZE), np.float32)
        sub = arr[:, pr:pr + s1.PATCH_SIZE, pc:pc + s1.PATCH_SIZE]
        out[:, :sub.shape[1], :sub.shape[2]] = s1.normalize_detector(sub)
        return torch.from_numpy(out[None])

    def best_scores(side: str, col_abs: int, row_abs: int) -> list[dict]:
        stack, (o_c, o_r) = sides[side]
        lc, lr = col_abs - o_c, row_abs - o_r
        res = []
        for pc, pr in s1.patch_origins(*stack.shape[1:]):
            if not (pc <= lc < pc + s1.PATCH_SIZE and pr <= lr < pr + s1.PATCH_SIZE):
                continue
            with torch.inference_mode():
                det = s1.detector_forward(model, patch(stack[:2], pc, pr), patch(stack[2:], pc, pr))[0]
            b, sc = det["boxes"].numpy(), det["scores"].numpy()
            near = np.hypot((b[:, 0] + b[:, 2]) / 2 + pc - lc, (b[:, 1] + b[:, 3]) / 2 + pr - lr) <= 10
            res.append({"patch_local": [pc, pr],
                        "edge_dist_px": int(min(lc - pc, pc + s1.PATCH_SIZE - lc,
                                                lr - pr, pr + s1.PATCH_SIZE - lr)),
                        "best_score": float(sc[near].max()) if near.any() else None})
        return res

    cmp = json.loads(a.cmp.read_text())
    report = []
    for kind, label in (("unmatched_ref", "ai2-only"), ("unmatched_mine", "port-only")):
        for u in cmp[kind]:
            x, y = to_utm(u["lon"], u["lat"])
            col, row = int(round(x / 10)), int(round(y / -10))
            h, w = port.shape[1:]
            report.append({"kind": label, "lon": u["lon"], "lat": u["lat"], "score": u["score"],
                           "nearest_other_m": u.get("nearest_mine_m", u.get("nearest_ref_m")),
                           "aoi_edge_dist_px": int(min(col - pc0, pc0 + w - col,
                                                       row - pr0, pr0 + h - row)),
                           "ai2_patches": best_scores("ai2", col, row),
                           "port_patches": best_scores("port", col, row)})
    rec.write_json(a.out, report)


if __name__ == "__main__":
    main()
