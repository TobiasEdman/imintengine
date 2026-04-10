#!/usr/bin/env python3
"""scripts/stream_training_log.py — Stream pixel training pod logs → training_log.json.

Polls `kubectl logs` for the train-pixel-v1 pod and parses epoch lines
into the training_log.json format consumed by training_dashboard.html.

Usage::

    python scripts/stream_training_log.py \\
        --namespace prithvi-training-default \\
        --job train-pixel-v1 \\
        --out training_log.json \\
        --interval 15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Resolve kubectl — may not be on PATH when run as a background process
_KUBECTL = shutil.which("kubectl") or "/usr/local/bin/kubectl"


# ── Log line pattern ──────────────────────────────────────────────────────
# "  Epoch   3/ 35  loss=1.2345  mAcc=0.1234  val_loss=1.1  val_mAcc=0.15  val_OA=0.7  lr=3.00e-04  42s"
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)"
    r".*?loss=([\d.]+)"
    r"(?:.*?val_loss=([\d.]+))?"
    r"(?:.*?val_mAcc=([\d.]+))?"
    r"(?:.*?val_OA=([\d.]+))?"
    r"(?:.*?mAcc=([\d.]+))?"
    r"(?:.*?lr=([\d.e+-]+))?"
)
_STAGE_RE = re.compile(r"Stage\s+(\d+)")
_BEST_RE  = re.compile(r"New best mAcc=([\d.]+)")


def _get_pod_name(namespace: str, job: str) -> str | None:
    try:
        out = subprocess.check_output(
            [_KUBECTL, "get", "pods", "-n", namespace,
             "-l", f"job-name={job}", "-o", "jsonpath={.items[0].metadata.name}"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _fetch_logs(namespace: str, pod: str, since: str = "1h") -> str:
    try:
        return subprocess.check_output(
            [_KUBECTL, "logs", pod, "-n", namespace, f"--since={since}"],
            stderr=subprocess.DEVNULL,
        ).decode(errors="replace")
    except Exception:
        return ""


def _parse_logs(raw: str) -> dict:
    epochs: list[dict] = []
    best_val_acc = 0.0
    current_stage = 1

    for line in raw.splitlines():
        # Stage transition
        sm = _STAGE_RE.search(line)
        if sm:
            current_stage = int(sm.group(1))

        # New best marker
        bm = _BEST_RE.search(line)
        if bm:
            best_val_acc = float(bm.group(1))

        # Epoch line
        em = _EPOCH_RE.search(line)
        if em:
            ep_num   = int(em.group(1))
            ep_total = int(em.group(2))
            train_loss = float(em.group(3))
            val_loss   = float(em.group(4)) if em.group(4) else None
            val_acc    = float(em.group(5)) if em.group(5) else None
            val_oa     = float(em.group(6)) if em.group(6) else None
            train_acc  = float(em.group(7)) if em.group(7) else None
            lr         = float(em.group(8)) if em.group(8) else None

            entry: dict = {
                "epoch": ep_num,
                "total_epochs": ep_total,
                "stage": current_stage,
                "train_loss": train_loss,
            }
            if train_acc is not None:
                entry["train_mAcc"] = train_acc
            if val_loss is not None:
                entry["val_loss"] = val_loss
            if val_acc is not None:
                entry["val_mAcc"] = val_acc
            if val_oa is not None:
                entry["val_OA"] = val_oa
            if lr is not None:
                entry["lr"] = lr

            # Deduplicate by epoch number (keep latest)
            epochs = [e for e in epochs if e["epoch"] != ep_num]
            epochs.append(entry)

    # Sort by epoch
    epochs.sort(key=lambda e: e["epoch"])

    # Determine status
    status = "running"
    if epochs:
        last = epochs[-1]
        if last["epoch"] >= last["total_epochs"]:
            status = "completed"

    return {
        "status": status,
        "model": "PrithviPixelClassifier",
        "best_val_mAcc": best_val_acc,
        "epochs": epochs,
        "num_epochs": epochs[-1]["total_epochs"] if epochs else 35,
        "current_epoch": epochs[-1]["epoch"] if epochs else 0,
    }


def _write_log(out_path: Path, data: dict) -> None:
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default="prithvi-training-default")
    p.add_argument("--job", default="train-pixel-v1")
    p.add_argument("--out", default="training_log.json")
    p.add_argument("--interval", type=int, default=15,
                   help="Poll interval in seconds (default: 15)")
    p.add_argument("--once", action="store_true",
                   help="Fetch once and exit")
    args = p.parse_args()

    out_path = Path(args.out)
    print(f"[stream] polling job={args.job} ns={args.namespace} → {out_path}", flush=True)

    while True:
        pod = _get_pod_name(args.namespace, args.job)
        if not pod:
            print(f"[stream] pod not found for job {args.job}, retrying…", flush=True)
        else:
            raw = _fetch_logs(args.namespace, pod)
            data = _parse_logs(raw)
            _write_log(out_path, data)
            ep = data.get("current_epoch", 0)
            tot = data.get("num_epochs", "?")
            st = data.get("status", "?")
            best = data.get("best_val_mAcc", 0)
            print(
                f"[stream] epoch {ep}/{tot}  status={st}  best_mAcc={best:.4f}"
                f"  → {out_path}",
                flush=True,
            )
            if st == "completed" or args.once:
                break

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
