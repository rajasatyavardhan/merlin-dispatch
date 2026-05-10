# safety/validator.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Safety Validator — rule-based constraint checking
# Ensures no dispatch decision violates hard operational rules

from core.grid import (
    SEVERITY_CRITICAL, SEVERITY_HIGH,
    TERRAIN_WATER, WEATHER_STORM, WEATHER_BLIZZARD
)
from core.assets import VehicleType

# ── Validation Result Codes ──────────────────────────────────────
# Every validation check returns one of these codes.
# This makes debugging dispatch decisions transparent —
# you always know WHY a vehicle was approved or rejected.

VALID               = "valid"
INVALID_WEATHER     = "invalid_weather"
INVALID_TERRAIN     = "invalid_terrain"
INVALID_NO_LANDING  = "invalid_no_landing_zone"
INVALID_UNAVAILABLE = "invalid_vehicle_unavailable"
INVALID_RANGE       = "invalid_out_of_range"
FORCE_HELICOPTER    = "force_helicopter"
FORCE_GROUND        = "force_ground"

# ── Hard Rule Thresholds ─────────────────────────────────────────
# These are the non-negotiable operational limits.
# No optimisation can override these — they are absolute.

# If distance exceeds this, ground ambulance is too slow
# for Critical severity — helicopter must be considered
CRITICAL_DISTANCE_THRESHOLD_KM = 30.0

# If distance is under this, ambulance is always faster
# even for helicopter-eligible emergencies
AMBULANCE_ALWAYS_FASTER_KM = 5.0

# Maximum helicopter range before refuel
HELICOPTER_MAX_RANGE_KM = 400.0

# ── Individual Rule Checks ───────────────────────────────────────

def check_helicopter_weather(cell):
    """
    Rule 1 — Weather Safety
    
    Helicopter cannot fly in Storm or Blizzard.
    This encodes Transport Canada operational minimums.
    This rule CANNOT be overridden regardless of severity.
    
    Returns: VALID or INVALID_WEATHER
    """
    if cell.weather in (WEATHER_STORM, WEATHER_BLIZZARD):
        return INVALID_WEATHER
    return VALID


def check_helicopter_terrain(cell):
    """
    Rule 2 — Landing Zone and Terrain
    
    Helicopter cannot land on water cells.
    Helicopter cannot land where no landing zone exists.
    
    Returns: VALID or INVALID_TERRAIN or INVALID_NO_LANDING
    """
    if cell.terrain == TERRAIN_WATER:
        return INVALID_TERRAIN
    if not cell.has_landing_zone:
        return INVALID_NO_LANDING
    return VALID


def check_helicopter_range(distance_km):
    """
    Rule 3 — Range Limit
    
    Helicopter cannot travel beyond its operational range.
    400km is the ORNGE (Ontario Air Ambulance) EC135 range.
    
    Returns: VALID or INVALID_RANGE
    """
    if distance_km > HELICOPTER_MAX_RANGE_KM:
        return INVALID_RANGE
    return VALID


def check_critical_override(severity, distance_km,
                             helicopter_available):
    """
    Rule 4 — Critical Emergency Override
    
    If severity is CRITICAL and distance exceeds the
    threshold AND helicopter is available → FORCE helicopter.
    
    This is the most important rule in MERLIN.
    A Critical patient beyond 30km MUST get a helicopter
    if one is available — the optimizer cannot choose
    ambulance to save cost in this scenario.
    
    This rule directly addresses the misallocation
    crisis MERLIN is designed to solve.
    
    Returns: FORCE_HELICOPTER or VALID (no override needed)
    """
    if (severity == SEVERITY_CRITICAL and
            distance_km > CRITICAL_DISTANCE_THRESHOLD_KM and
            helicopter_available):
        return FORCE_HELICOPTER
    return VALID


def check_proximity_override(distance_km):
    """
    Rule 5 — Proximity Rule
    
    If the emergency is very close (under 5km),
    the ambulance arrives faster than the helicopter
    even accounting for spinup time difference.
    
    Helicopter spinup = 10 min + 5km/250kmh × 60 = 11.2 min
    Ambulance spinup  = 2 min  + 5km/80kmh  × 60 = 5.75 min
    
    Ground ambulance is always faster under 5km.
    Sending helicopter wastes money and blocks it
    for the next potentially critical call.
    
    Returns: FORCE_GROUND or VALID (no override needed)
    """
    if distance_km <= AMBULANCE_ALWAYS_FASTER_KM:
        return FORCE_GROUND
    return VALID

# ── Master Validation Function ───────────────────────────────────
class ValidationResult:
    """
    Contains the complete result of a dispatch validation.
    
    Tells the dispatch engine:
    1. Can helicopter be dispatched? (and why not if not)
    2. Can ambulance be dispatched? (and why not if not)
    3. Is there a forced decision overriding the optimizer?
    4. What was the reasoning? (for logging and paper results)
    """

    def __init__(self):
        self.helicopter_valid   = False
        self.ambulance_valid    = False
        self.forced_decision    = None  # None / "helicopter" / "ambulance"
        self.helicopter_reason  = None
        self.ambulance_reason   = None
        self.force_reason       = None

    def __repr__(self):
        return (
            f"ValidationResult | "
            f"Heli: {self.helicopter_valid} "
            f"({self.helicopter_reason}) | "
            f"Amb: {self.ambulance_valid} | "
            f"Forced: {self.forced_decision}"
        )


def validate_dispatch(emergency, fleet, grid):
    """
    Master validation function — runs all rules for one emergency.
    
    Called by the dispatch engine before every decision.
    Returns a ValidationResult that the engine acts on.
    
    Rule priority (highest to lowest):
    1. Weather safety       (absolute — cannot override)
    2. Terrain/landing zone (absolute — cannot override)
    3. Range limit          (absolute — cannot override)
    4. Critical override    (forces helicopter if available)
    5. Proximity override   (forces ground if very close)
    6. Availability         (is vehicle actually free?)
    
    The dispatch engine only runs its optimisation
    if no forced decision exists. Otherwise it follows
    the safety validator's ruling.
    """
    result = ValidationResult()

    # ── Get emergency attributes ─────────────────────────────────
    cell     = emergency.cell
    severity = emergency.severity

    # ── Calculate distances ──────────────────────────────────────
    # Distance from nearest helicopter base to emergency
    heli_distance = float('inf')
    if fleet.helicopters:
        h = fleet.helicopters[0]   # primary helicopter
        heli_distance = grid.distance_between_cells(
            h.base_col, h.base_row,
            cell.col, cell.row
        )

    # Distance from nearest available ambulance to emergency
    amb_distance = float('inf')
    available_ambs = fleet.available_ambulances(cell)
    if available_ambs:
        nearest_amb = fleet.nearest_vehicle(
            available_ambs, grid, cell.col, cell.row
        )
        if nearest_amb:
            amb_distance = grid.distance_between_cells(
                nearest_amb.base_col, nearest_amb.base_row,
                cell.col, cell.row
            )

    # ── Check helicopter validity ────────────────────────────────
    weather_check  = check_helicopter_weather(cell)
    terrain_check  = check_helicopter_terrain(cell)
    range_check    = check_helicopter_range(heli_distance)
    heli_available = len(fleet.available_helicopters(cell)) > 0

    if weather_check != VALID:
        result.helicopter_valid  = False
        result.helicopter_reason = weather_check
    elif terrain_check != VALID:
        result.helicopter_valid  = False
        result.helicopter_reason = terrain_check
    elif range_check != VALID:
        result.helicopter_valid  = False
        result.helicopter_reason = range_check
    elif not heli_available:
        result.helicopter_valid  = False
        result.helicopter_reason = INVALID_UNAVAILABLE
    else:
        result.helicopter_valid  = True
        result.helicopter_reason = VALID

    # ── Check ambulance validity ─────────────────────────────────
    # Ambulance is valid if any ambulance can reach this cell
    result.ambulance_valid = len(available_ambs) > 0

    # ── Check forced decisions ───────────────────────────────────
    # Proximity check — ground always faster under 5km
    proximity = check_proximity_override(
        min(heli_distance, amb_distance)
    )
    if proximity == FORCE_GROUND:
        result.forced_decision = "ambulance"
        result.force_reason    = FORCE_GROUND
        return result   # Return immediately — no need to check further

    # Critical override — must send helicopter for critical + far
    critical = check_critical_override(
        severity, heli_distance, result.helicopter_valid
    )
    if critical == FORCE_HELICOPTER:
        result.forced_decision = "helicopter"
        result.force_reason    = FORCE_HELICOPTER

    return result