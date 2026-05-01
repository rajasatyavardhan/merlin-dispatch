# core/emergency.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Emergency Module — generates and profiles emergency events

import random
import numpy as np
from core.grid import (
    SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL,
    SEVERITY_NAMES, WEATHER_CLEAR, WEATHER_CLOUDY,
    WEATHER_STORM, WEATHER_BLIZZARD, TERRAIN_WATER
)

# ── Emergency Types ──────────────────────────────────────────────
# Each type has a different severity distribution.
# This reflects real EMS (Emergency Medical Services) call patterns.

EMERGENCY_CARDIAC   = 0   # Cardiac arrest — highest severity
EMERGENCY_TRAUMA    = 1   # Vehicle accident, fall, injury
EMERGENCY_MEDICAL   = 2   # Stroke, diabetic emergency, seizure
EMERGENCY_RESPIRATORY = 3 # Breathing difficulty, asthma
EMERGENCY_MINOR     = 4   # Minor injury, non-life-threatening

EMERGENCY_NAMES = {
    EMERGENCY_CARDIAC:     "Cardiac Arrest",
    EMERGENCY_TRAUMA:      "Trauma",
    EMERGENCY_MEDICAL:     "Medical Emergency",
    EMERGENCY_RESPIRATORY: "Respiratory",
    EMERGENCY_MINOR:       "Minor Injury"
}

# ── Emergency Type Probabilities ─────────────────────────────────
# How likely each type is — based on real Canadian EMS call data.
# Must sum to 1.0
EMERGENCY_TYPE_WEIGHTS = [
    0.08,   # Cardiac — rare but critical
    0.22,   # Trauma — common in rural areas
    0.30,   # Medical — most common overall
    0.20,   # Respiratory — common
    0.20,   # Minor — common but low priority
]

# ── Emergency Class ──────────────────────────────────────────────
class Emergency:
    """
    Represents a single emergency event in MERLIN's simulation.

    Each emergency has:
    - A location (cell)
    - A type (cardiac, trauma, etc.)
    - A severity (low → critical)
    - A timestamp (tick)
    - Patient attributes

    This object flows through the entire system:
    generation → AI → dispatch → outcome
    """

    def __init__(self, cell, emergency_type, severity,
                 tick, patient_age, call_delay_ticks):

        # ── Location ─────────────────────────────────────────────
        self.cell = cell
        self.col  = cell.col
        self.row  = cell.row

        # ── Emergency profile ────────────────────────────────────
        self.emergency_type = emergency_type
        self.severity       = severity

        # ── Time ─────────────────────────────────────────────────
        self.tick = tick
        self.call_delay_ticks = call_delay_ticks
        self.call_tick = tick + call_delay_ticks

        # ── Patient ──────────────────────────────────────────────
        self.patient_age = patient_age

        # ── Dispatch results (filled later) ──────────────────────
        self.dispatched_vehicle = None
        self.dispatch_tick      = None
        self.arrival_tick       = None
        self.outcome            = None

    @property
    def response_time_ticks(self):
        """
        Total response time (in ticks/minutes).
        Only available after dispatch.
        """
        if self.arrival_tick is None or self.call_tick is None:
            return None
        return self.arrival_tick - self.call_tick

    @property
    def severity_name(self):
        return SEVERITY_NAMES[self.severity]

    @property
    def type_name(self):
        return EMERGENCY_NAMES[self.emergency_type]

    def __repr__(self):
        return (
            f"Emergency | {self.type_name} | "
            f"Severity: {self.severity_name} | "
            f"Cell: ({self.col},{self.row}) | "
            f"Tick: {self.tick}"
        )

# ── Severity Assignment ──────────────────────────────────────────
def assign_severity(emergency_type, patient_age, cell, season):
    """
    Assigns a severity level using weighted probabilities.

    Factors:
    - Emergency type
    - Patient age
    - Terrain constraints
    - Season effects

    Returns: severity (0=Low → 3=Critical)
    """

    # ── Base weights by emergency type ──────────────────────────
    base_weights = {
        EMERGENCY_CARDIAC:     [0.00, 0.05, 0.25, 0.70],
        EMERGENCY_TRAUMA:      [0.10, 0.30, 0.40, 0.20],
        EMERGENCY_MEDICAL:     [0.15, 0.40, 0.35, 0.10],
        EMERGENCY_RESPIRATORY: [0.10, 0.35, 0.40, 0.15],
        EMERGENCY_MINOR:       [0.60, 0.30, 0.08, 0.02],
    }

    weights = list(base_weights[emergency_type])

    # ── Age effect ──────────────────────────────────────────────
    if patient_age >= 65:
        weights[0] *= 0.5
        weights[1] *= 0.7
        weights[2] *= 1.3
        weights[3] *= 1.8

    elif patient_age < 18:
        weights[2] *= 1.2
        weights[3] *= 1.4

    # ── Terrain effect ──────────────────────────────────────────
    if cell.terrain == TERRAIN_WATER:
        weights[0] *= 0.3
        weights[1] *= 0.6
        weights[2] *= 1.4
        weights[3] *= 2.0

    # ── Seasonal effect ─────────────────────────────────────────
    if season == 3:  # Winter
        if emergency_type == EMERGENCY_TRAUMA:
            weights[2] *= 1.3
            weights[3] *= 1.5

    # ── Normalize weights ───────────────────────────────────────
    total = sum(weights)
    weights = [w / total for w in weights]

    # ── Sample severity ─────────────────────────────────────────
    severity = random.choices(
        [SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL],
        weights=weights,
        k=1
    )[0]

    return severity

# ── Season Helper ────────────────────────────────────────────────
def get_season(tick):
    """
    Returns season based on tick.
    0 = Spring, 1 = Summer, 2 = Autumn, 3 = Winter
    """
    ticks_per_season = 525600 // 4
    return (tick // ticks_per_season) % 4


def get_time_of_day(tick):
    """Returns hour (0–23)"""
    return (tick % 1440) // 60


# ── Emergency Generator ──────────────────────────────────────────
def generate_emergency_event(cell, tick):
    """
    Converts a triggered cell into a full Emergency object.
    """

    # ── Emergency type ───────────────────────────────────────────
    emergency_type = random.choices(
        [0, 1, 2, 3, 4],
        weights=EMERGENCY_TYPE_WEIGHTS,
        k=1
    )[0]

    # ── Patient age ──────────────────────────────────────────────
    patient_age = int(np.clip(np.random.normal(52, 18), 1, 95))

    # ── Season ───────────────────────────────────────────────────
    season = get_season(tick)

    # ── Severity ─────────────────────────────────────────────────
    severity = assign_severity(emergency_type, patient_age, cell, season)

    # ── Call delay ───────────────────────────────────────────────
    if cell.population_density > 0.5:
        call_delay = max(1, int(np.random.normal(3, 1)))
    elif cell.population_density > 0.1:
        call_delay = max(1, int(np.random.normal(8, 3)))
    else:
        call_delay = max(1, int(np.random.normal(20, 8)))

    return Emergency(
        cell=cell,
        emergency_type=emergency_type,
        severity=severity,
        tick=tick,
        patient_age=patient_age,
        call_delay_ticks=call_delay
    )