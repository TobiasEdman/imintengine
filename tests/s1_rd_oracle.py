"""Independent zero-Doppler range-Doppler geocoder for S1 GRD geolocation-grid points.

Test oracle, ported from the 2026-10-20 offset diagnosis (``rd.py``: validated to
0.015 m on the 840 grid points of four products at the stated heights). It
shares nothing with the pipeline's geocoding: orbit state vectors (cubic
Hermite, ECEF) + the grid point's azimuth time and slant-range time -> the point
at ellipsoidal height h with zero Doppler, ``|P - S| = R``, on the right-looking
side.
"""
from __future__ import annotations

import datetime as dt
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

C = 299792458.0
_A_WGS, _B_WGS = 6378137.0, 6356752.314245


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC)


class RangeDopplerOracle:
    """Orbit + geolocation grid of one product annotation (plain or ``.gz``)."""

    def __init__(self, path: Path):
        from scipy.interpolate import CubicHermiteSpline

        if str(path).endswith(".gz"):
            with gzip.open(path) as f:
                raw = f.read()
        else:
            raw = Path(path).read_bytes()
        self.xml = raw
        a = ET.fromstring(raw)
        self.t0 = _t(a.findtext("imageAnnotation/imageInformation/productFirstLineUtcTime"))
        self.pass_ = a.findtext("generalAnnotation/productInformation/pass")
        orb = a.findall("generalAnnotation/orbitList/orbit")
        ts = np.array([(_t(o.findtext("time")) - self.t0).total_seconds() for o in orb])
        pos = np.array([[float(o.findtext(f"position/{k}")) for k in "xyz"] for o in orb])
        vel = np.array([[float(o.findtext(f"velocity/{k}")) for k in "xyz"] for o in orb])
        i = np.argsort(ts)
        self._pos = CubicHermiteSpline(ts[i], pos[i], vel[i], axis=0)
        self._vel = self._pos.derivative()
        g = a.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
        self.grid = {k: np.array([float(p.findtext(k)) for p in g]) for k in
                     ("line", "pixel", "latitude", "longitude", "height", "incidenceAngle",
                      "slantRangeTime")}
        self.grid["az_s"] = np.array([(_t(p.findtext("azimuthTime")) - self.t0).total_seconds()
                                      for p in g])

    def geolocate_grid(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) of every grid point re-solved at ellipsoidal height ``h`` (m)."""
        from pyproj import Transformer

        to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
        to_llh = Transformer.from_crs(4978, 4979, always_xy=True)
        g = self.grid
        h = np.broadcast_to(np.asarray(h, float), g["az_s"].shape)
        start = np.array(to_ecef.transform(g["longitude"], g["latitude"], g["height"])).T
        sats, vels = self._pos(g["az_s"]), self._vel(g["az_s"])
        out = np.empty_like(start)
        for k in range(h.size):
            s, v, p = sats[k], vels[k], start[k]
            r = g["slantRangeTime"][k] * C / 2
            a2, b2 = (_A_WGS + h[k]) ** 2, (_B_WGS + h[k]) ** 2
            for _ in range(30):
                d = p - s
                f = np.array([d @ d - r ** 2, d @ v,
                              (p[0] ** 2 + p[1] ** 2) / a2 + p[2] ** 2 / b2 - 1])
                j = np.array([2 * d, v, [2 * p[0] / a2, 2 * p[1] / a2, 2 * p[2] / b2]])
                step = np.linalg.solve(j, f)
                p = p - step
                if np.abs(step).max() < 1e-5:
                    break
            out[k] = p
        lon, lat, _ = to_llh.transform(out[:, 0], out[:, 1], out[:, 2])
        return np.asarray(lon), np.asarray(lat)
