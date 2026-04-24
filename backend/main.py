"""
CERN CMS Open Data – Dimuon Event API
Serves real LHC collision events from Dimuon_DoubleMu.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── CSV loading & pre-processing ──────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "Dimuon_DoubleMu.csv"

df = pd.read_csv(CSV_PATH)

# Build a unique event_id from Run + Event
df["event_id"] = df["Run"].astype(str) + "-" + df["Event"].astype(str)


def classify_mass(m: float) -> str:
    if 85 <= m <= 97:
        return "Z Boson Candidate"
    if 2.9 <= m <= 3.3:
        return "J/ψ Meson"
    if 9.2 <= m <= 9.6:
        return "Υ (Upsilon) Meson"
    return "Dimuon Event"


df["event_type"] = df["M"].apply(classify_mass)

# Pre-compute counts once
total_events = len(df)
z_boson_count = int((df["event_type"] == "Z Boson Candidate").sum())
jpsi_count = int((df["event_type"] == "J/ψ Meson").sum())
upsilon_count = int((df["event_type"] == "Υ (Upsilon) Meson").sum())
mass_min = float(df["M"].min())
mass_max = float(df["M"].max())

# Index for O(1) lookup by event_id
event_index: dict[str, int] = {eid: idx for idx, eid in enumerate(df["event_id"])}

# ─── helpers ───────────────────────────────────────────────────────────────────


def _normalize(px: float, py: float, pz: float) -> dict:
    mag = math.sqrt(px * px + py * py + pz * pz)
    if mag == 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "x": round(px / mag, 6),
        "y": round(py / mag, 6),
        "z": round(pz / mag, 6),
    }


def _row_to_event(row: pd.Series) -> dict:
    q1, q2 = int(row["Q1"]), int(row["Q2"])
    return {
        "event_id": row["event_id"],
        "invariant_mass": round(float(row["M"]), 3),
        "energy_tev": "8.0 TeV",
        "event_type": row["event_type"],
        "particles": [
            {
                "type": "muon",
                "charge": q1,
                "momentum": {
                    "px": round(float(row["px1"]), 4),
                    "py": round(float(row["py1"]), 4),
                    "pz": round(float(row["pz1"]), 4),
                },
                "energy": round(float(row["E1"]), 4),
                "eta": round(float(row["eta1"]), 4),
                "phi": round(float(row["phi1"]), 4),
                "pt": round(float(row["pt1"]), 4),
                "color": "#ff006e" if q1 < 0 else "#00d4ff",
            },
            {
                "type": "muon",
                "charge": q2,
                "momentum": {
                    "px": round(float(row["px2"]), 4),
                    "py": round(float(row["py2"]), 4),
                    "pz": round(float(row["pz2"]), 4),
                },
                "energy": round(float(row["E2"]), 4),
                "eta": round(float(row["eta2"]), 4),
                "phi": round(float(row["phi2"]), 4),
                "pt": round(float(row["pt2"]), 4),
                "color": "#ff006e" if q2 < 0 else "#00d4ff",
            },
        ],
        "track_directions": [
            _normalize(float(row["px1"]), float(row["py1"]), float(row["pz1"])),
            _normalize(float(row["px2"]), float(row["py2"]), float(row["pz2"])),
        ],
    }


# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="CERN CMS Dimuon Event API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/events")
def get_events():
    """Return 10 random collision events."""
    sample = df.sample(n=min(10, total_events))
    return [_row_to_event(row) for _, row in sample.iterrows()]


@app.get("/event/{event_id}")
def get_event(event_id: str):
    """Return a single event by its Run-Event ID."""
    idx = event_index.get(event_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return _row_to_event(df.iloc[idx])


@app.get("/stats")
def get_stats():
    """Return aggregate dataset statistics."""
    return {
        "total_events": total_events,
        "z_boson_candidates": z_boson_count,
        "jpsi_candidates": jpsi_count,
        "upsilon_candidates": upsilon_count,
        "mass_range": {
            "min": round(mass_min, 3),
            "max": round(mass_max, 3),
        },
    }
