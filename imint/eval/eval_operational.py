"""Phase 4 — operational / deployment readiness.

Pure performance + robustness checks that don't require labels:

    inference_latency_p50_p95: Per-tile wall-time at batch=1 and
                               batch=8 on the target inference device.
    gpu_memory_peak:           torch.cuda.max_memory_allocated during
                               a representative forward pass.
    throughput_batch_8:        Tiles/min sustained over 100 tiles.
    aux_drop_robustness:       For each aux channel, zero it and
                               measure the relative IoU drop. Cheap
                               proxy for "how much do we depend on
                               this signal?".
    nodata_robustness:         Randomly mask 15% of input pixels and
                               measure IoU drop. Tests robustness to
                               cloud/shadow remnants the SCL filter
                               missed.

All of these are quick — phase 4 should finish in ≤30 minutes on a
single H100.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import EvalResult


def run(
    ensemble_checkpoint: Path,
    tiles_dir: str,
    sample_tiles: list[str],                 # ~100 tiles for averaging
    *,
    output_dir: Path | None = None,
    device: str = "cuda",
    batch_sizes: tuple[int, ...] = (1, 4, 8),
) -> dict[str, EvalResult]:
    """Run the operational suite.

    TODO: torch.cuda.Event for accurate latency; torch.cuda.reset_peak_
    memory_stats / max_memory_allocated for GPU peak; pre-load all
    sample tiles into a list-of-tensors so IO isn't part of the timing.
    """
    raise NotImplementedError
