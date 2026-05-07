"""Steg 3 av Lilla Karlsö-pipelinen: docker-run C2RCC per SAFE.

Loopar genom alla SAFE-arkiv i config.SAFE_CACHE och kör
docker/c2rcc-snap/run.sh för varje. BEAM-DIMAP-output landar i
config.DIMAP_OUT.

Skip-existing: om <date>.dim redan finns hoppas SAFE över.

Användning:
    python -m demos.lilla_karlso_birds.run_c2rcc
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demos.lilla_karlso_birds import config  # noqa: E402


def aoi_wkt() -> str:
    """WGS84-polygon för SNAP Subset-operatorn (CCW, sluten)."""
    b = config.BBOX_WGS84
    return (
        f"POLYGON(("
        f"{b['west']} {b['south']},"
        f"{b['east']} {b['south']},"
        f"{b['east']} {b['north']},"
        f"{b['west']} {b['north']},"
        f"{b['west']} {b['south']}"
        f"))"
    )


def run_c2rcc(safe_path: Path, output_dim: Path) -> dict:
    """Kör docker/c2rcc-snap/run.sh på en SAFE-arkiv."""
    if output_dim.exists():
        return {"status": "skipped", "output": str(output_dim)}

    output_dim.parent.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "docker" / "c2rcc-snap" / "run.sh"
    assert script.is_file(), f"missing {script}"

    b = config.BBOX_WGS84
    cmd = [
        "bash", str(script),
        str(safe_path), str(output_dim),
        "--west", str(b["west"]),
        "--south", str(b["south"]),
        "--east", str(b["east"]),
        "--north", str(b["north"]),
    ]
    t0 = time.time()
    env = {"C2RCC_IMAGE": config.DOCKER_IMAGE}
    import os
    env = {**os.environ, **env}

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=3600,
        )
        elapsed = round(time.time() - t0, 1)
        if result.returncode != 0:
            return {
                "status": "error", "elapsed_s": elapsed,
                "returncode": result.returncode,
                "stderr": result.stderr[-1000:],
            }
        return {
            "status": "ok", "elapsed_s": elapsed,
            "output": str(output_dim),
            "data_dir": str(output_dim.with_suffix(".data")),
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": round(time.time() - t0, 1)}


def main() -> int:
    plan_path = config.SAFE_CACHE.parent / "plan.json"
    if not plan_path.is_file():
        print(f"ERR: ingen plan.json — kör fetch_safes.py först")
        return 1
    plan = json.loads(plan_path.read_text())

    config.DIMAP_OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== C2RCC ({len(plan['dates'])} datum) ===")
    print(f"image:      {config.DOCKER_IMAGE}")
    print(f"safe-cache: {config.SAFE_CACHE}")
    print(f"dimap-out:  {config.DIMAP_OUT}")

    results = []
    for i, date in enumerate(plan["dates"], 1):
        # Hitta SAFE-katalogen för datumet
        safes = list(config.SAFE_CACHE.glob(f"S2*_MSIL1C_{date.replace('-', '')}T*.SAFE"))
        if not safes:
            print(f"\n[{i}/{len(plan['dates'])}] {date} — SAFE saknas")
            results.append({"date": date, "status": "no_safe"})
            continue
        safe = safes[0]
        out = config.DIMAP_OUT / f"{date}.dim"

        print(f"\n[{i}/{len(plan['dates'])}] {date} → {out.name}")
        print(f"    SAFE: {safe.name}")
        rec = run_c2rcc(safe, out)
        rec["date"] = date
        rec["safe"] = safe.name
        results.append(rec)
        if rec["status"] == "ok":
            print(f"    ✓ {rec['elapsed_s']}s")
        elif rec["status"] == "skipped":
            print(f"    [skip] {out.name} finns redan")
        else:
            print(f"    ✗ {rec['status']}: {rec.get('stderr', '')[:200]}")

    summary_path = config.DIMAP_OUT / "c2rcc_summary.json"
    summary_path.write_text(json.dumps({
        "image": config.DOCKER_IMAGE,
        "results": results,
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "n_failed": sum(1 for r in results if r["status"] not in ("ok", "skipped")),
    }, indent=2, ensure_ascii=False))
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
