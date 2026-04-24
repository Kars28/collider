"""
CERN CMS Open Data – Dimuon Event API
Serves real LHC collision events from Dimuon_DoubleMu.csv
"""

from __future__ import annotations

import math
import random
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
        return "J/\u03c8 Meson"
    if 9.2 <= m <= 9.6:
        return "\u03a5(1S) Meson"
    if 9.9 <= m <= 10.1:
        return "\u03a5(2S) Meson"
    if 10.3 <= m <= 10.4:
        return "\u03a5(3S) Meson"
    if 0.99 <= m <= 1.05:
        return "\u03c6 Meson"
    if 0.6 <= m <= 0.9:
        return "\u03c1/\u03c9 Meson"
    if m < 2.0:
        return "Low Mass Dimuon"
    if m > 100:
        return "High Mass Dimuon"
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

# Pre-filtered subsets for /events/filter
_FILTER_MAP: dict[str, pd.DataFrame] = {
    "z_boson":    df[(df["M"] >= 85) & (df["M"] <= 97)],
    "jpsi":       df[(df["M"] >= 2.9) & (df["M"] <= 3.3)],
    "upsilon_1s": df[(df["M"] >= 9.2) & (df["M"] <= 9.6)],
    "upsilon_2s": df[(df["M"] >= 9.9) & (df["M"] <= 10.1)],
    "upsilon_3s": df[(df["M"] >= 10.3) & (df["M"] <= 10.4)],
    "phi":        df[(df["M"] >= 0.99) & (df["M"] <= 1.05)],
    "rho_omega":  df[(df["M"] >= 0.6) & (df["M"] <= 0.9)],
    "low_mass":   df[df["M"] < 2.0],
    "high_mass":  df[df["M"] > 100],
    "dimuon":     df[df["M"] > 20],
    "any":        df,
}

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


_SPECIFIC_CATEGORIES = [
    "z_boson", "jpsi", "upsilon_1s", "upsilon_2s", "upsilon_3s",
    "phi", "rho_omega", "high_mass", "low_mass",
]


@app.get("/events/filter")
def get_filtered_events(event_type: str = "any", limit: int = 10):
    """Return a random sample of events filtered by type."""
    resolved = event_type

    if event_type == "any":
        # Pick a random specific category
        resolved = random.choice(_SPECIFIC_CATEGORIES)
        subset = _FILTER_MAP.get(resolved)
    else:
        subset = _FILTER_MAP.get(event_type)

    if subset is None:
        valid = ", ".join(_FILTER_MAP.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event_type '{event_type}'. Use: {valid}",
        )
    if subset.empty:
        return []
    sample = subset.sample(n=min(limit, len(subset)))
    events = [_row_to_event(row) for _, row in sample.iterrows()]
    for ev in events:
        ev["resolved_type"] = resolved
    return events


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

# ─── particle info ─────────────────────────────────────────────────────────────

_PARTICLE_INFO: dict[str, dict] = {
    "z_boson": {
        "name": "Z Boson",
        "mass": "91.2 GeV",
        "symbol": "Z\u2070",
        "what_is_it": "The neutral carrier of the weak nuclear force \u2014 one of the four fundamental forces of nature.",
        "what_happened": "A quark from one proton annihilated with an antiquark from the other, briefly creating a Z boson. It instantly decayed into the two muons flying outward.",
        "discovery": "Discovered at CERN\u2019s SPS collider in 1983 by Carlo Rubbia and Simon van der Meer.",
        "nobel_prize": "Nobel Prize in Physics, 1984",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The Z boson lives for only 3\u00d710\u207b\u00b2\u2075 seconds, yet its mass is known to 0.002% precision \u2014 one of the most precisely measured particles ever discovered.",
        "color": "#00d4ff",
    },
    "jpsi": {
        "name": "J/\u03c8 Meson",
        "mass": "3.097 GeV",
        "symbol": "J/\u03c8",
        "what_is_it": "A bound state of a charm quark and its antimatter partner, the charm antiquark.",
        "what_happened": "Two protons collided and their constituent quarks and gluons produced a charm-anticharm pair that bound together into the J/\u03c8 meson, which then decayed into the two muons you see.",
        "discovery": "Discovered simultaneously in 1974 by Burton Richter at SLAC and Samuel Ting at Brookhaven \u2014 known as the November Revolution in particle physics.",
        "nobel_prize": "Nobel Prize in Physics, 1976",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The simultaneous independent discovery of the J/\u03c8 at two different labs on the same day \u2014 November 11, 1974 \u2014 shocked the physics world and confirmed the existence of the charm quark.",
        "color": "#ff006e",
    },
    "upsilon_1s": {
        "name": "\u03a5(1S) Upsilon Meson",
        "mass": "9.46 GeV",
        "symbol": "\u03a5(1S)",
        "what_is_it": "The ground state of the bottomonium family \u2014 a bound state of a bottom quark and bottom antiquark.",
        "what_happened": "A bottom quark and bottom antiquark were produced in the collision and bound together into the Upsilon meson, which decayed into the two muons you see.",
        "discovery": "Discovered at Fermilab in 1977 by Leon Lederman and colleagues using the E288 experiment.",
        "nobel_prize": "Leon Lederman received the Nobel Prize in Physics in 1988 (for the neutrino beam method and discovery of the muon neutrino).",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The Upsilon was the first evidence for the bottom (beauty) quark \u2014 the fifth quark to be discovered, heavier than a proton by nearly 10 times.",
        "color": "#aaff00",
    },
    "upsilon_2s": {
        "name": "\u03a5(2S) Upsilon Meson",
        "mass": "10.023 GeV",
        "symbol": "\u03a5(2S)",
        "what_is_it": "The first excited state of the bottomonium family \u2014 a bottom quark-antiquark pair in a higher energy configuration.",
        "what_happened": "A bottom-antibottom pair formed in an excited energy state, creating the \u03a5(2S). Like an atom jumping to a higher electron orbit, this is the same particles but with more internal energy.",
        "discovery": "Discovered shortly after the \u03a5(1S) at Fermilab in 1977.",
        "nobel_prize": "Part of the bottomonium family discoveries.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The three Upsilon states (1S, 2S, 3S) are like the energy levels of a hydrogen atom \u2014 but made of bottom quarks instead of electrons and protons.",
        "color": "#aaff00",
    },
    "upsilon_3s": {
        "name": "\u03a5(3S) Upsilon Meson",
        "mass": "10.355 GeV",
        "symbol": "\u03a5(3S)",
        "what_is_it": "The second excited state of the bottomonium system \u2014 the highest clearly resolved Upsilon state.",
        "what_happened": "A bottom-antibottom pair formed in the second excited energy configuration and decayed into the two muons you see.",
        "discovery": "Discovered at Fermilab in 1977 alongside the other Upsilon states.",
        "nobel_prize": "Part of the bottomonium family discoveries.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "Above the \u03a5(3S) energy, bottom quark pairs can escape each other \u2014 the meson dissolves. This boundary is called the open beauty threshold.",
        "color": "#aaff00",
    },
    "phi": {
        "name": "\u03c6 Phi Meson",
        "mass": "1.019 GeV",
        "symbol": "\u03c6",
        "what_is_it": "A bound state of a strange quark and strange antiquark \u2014 the lightest hidden-strangeness meson.",
        "what_happened": "A strange-antistrange quark pair produced in the collision formed the phi meson, which decayed into the two muons detected.",
        "discovery": "Discovered in 1962-1963 at Brookhaven National Laboratory.",
        "nobel_prize": "Part of the quark model discoveries that led to the 1969 Nobel Prize for Murray Gell-Mann.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The phi meson is one of the narrowest resonances in particle physics \u2014 its precise mass peak is used to calibrate detector energy scales at CMS.",
        "color": "#ffffff",
    },
    "rho_omega": {
        "name": "\u03c1/\u03c9 Meson Region",
        "mass": "0.775 / 0.782 GeV",
        "symbol": "\u03c1\u2070/\u03c9",
        "what_is_it": "Two overlapping light mesons made of up and down quarks \u2014 the rho and omega mesons are nearly identical in mass.",
        "what_happened": "Light quark-antiquark pairs produced in the collision formed these short-lived mesons, which decayed into the detected muon pair.",
        "discovery": "Both discovered in 1961 using early particle accelerators.",
        "nobel_prize": "Part of the broader meson spectrum discoveries.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The rho meson has the shortest lifetime of any meson \u2014 just 4\u00d710\u207b\u00b2\u2074 seconds. It decays before it can even travel across a single proton.",
        "color": "#ffffff",
    },
    "high_mass": {
        "name": "High Mass Dimuon",
        "mass": ">100 GeV",
        "symbol": "\u03bc\u207a\u03bc\u207b",
        "what_is_it": "A high-energy muon pair beyond the Z boson peak \u2014 potential signal region for new physics beyond the Standard Model.",
        "what_happened": "Two extremely energetic muons were produced in this collision. At these masses, physicists search for hypothetical new particles like Z-prime bosons or Kaluza-Klein gravitons.",
        "discovery": "No new particles discovered yet in this region \u2014 the search continues at higher LHC energies.",
        "nobel_prize": "This is an active search region \u2014 the next Nobel Prize in particle physics may come from discoveries here.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "Every high-mass dimuon event is carefully examined for signs of new physics. So far, all observations are consistent with Standard Model predictions \u2014 but physicists keep looking.",
        "color": "#ff006e",
    },
    "low_mass": {
        "name": "Low Mass Dimuon",
        "mass": "<2 GeV",
        "symbol": "\u03bc\u207a\u03bc\u207b",
        "what_is_it": "Low energy muon pairs from the lightest quark-antiquark bound states and QED processes.",
        "what_happened": "Light quarks or direct photon processes produced this low-energy muon pair. This region contains the pion, eta, rho, and omega meson contributions.",
        "discovery": "This mass region was among the first studied at early particle accelerators in the 1960s.",
        "nobel_prize": "Multiple Nobel Prizes connected to discoveries in this mass region.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "The low mass dimuon spectrum is like a fingerprint of QCD \u2014 the theory of the strong nuclear force. Every peak and valley encodes information about quark interactions.",
        "color": "#ffffff",
    },
    "any": {
        "name": "Random LHC Event",
        "mass": "varies",
        "symbol": "pp",
        "what_is_it": "A random proton-proton collision event from the CMS detector at the Large Hadron Collider.",
        "what_happened": "Two protons travelling at 99.9999991% the speed of light collided. Their constituent quarks and gluons interacted, producing the particles you see.",
        "discovery": "Every collision at the LHC is a unique experiment \u2014 billions happen every second during operation.",
        "nobel_prize": "The LHC program led to the 2013 Nobel Prize for the discovery of the Higgs boson.",
        "experiment": "CMS Detector, CERN LHC \u2014 Run 2012B \u2014 8 TeV proton-proton collisions",
        "did_you_know": "During peak operation, the LHC produces 600 million proton-proton collisions per second. Only 1 in 100,000 is interesting enough to save \u2014 the rest are discarded in real time.",
        "color": "#00d4ff",
    },
}


@app.get("/particle-info")
def get_particle_info():
    """Return physics explanations for each particle type."""
    return _PARTICLE_INFO
