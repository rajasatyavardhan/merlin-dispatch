# dispatch/engine.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Dispatch Engine — the central decision maker
# Combines safety rules + optimization to select the best vehicle

from core.grid import (
    SEVERITY_LOW, SEVERITY_MEDIUM,
    SEVERITY_HIGH, SEVERITY_CRITICAL
)
from core.assets import VehicleType
from safety.validator import (
    validate_dispatch, VALID,
    FORCE_HELICOPTER, FORCE_GROUND
)

# ── Dispatch Decision Constants ──────────────────────────────────
DISPATCH_HELICOPTER = "helicopter"
DISPATCH_AMBULANCE  = "ambulance"
DISPATCH_NONE       = "none"   # No vehicle available

# ── Cost-Time Trade-off Weight ───────────────────────────────────
# How much do we weight time savings vs cost savings?
# 
# Alpha = 1.0 means: every minute saved is worth $1
# Alpha = 10.0 means: every minute saved is worth $10
#
# In medical emergencies, time directly maps to survival.
# We set alpha high — saving minutes is worth extra cost.
# This value can be tuned for different priorities.
# In your paper, you can test multiple alpha values
# and show how results change — that is a sensitivity analysis.
ALPHA_TIME_WEIGHT = 8.0   # $8 value per minute saved

# ── Cost-Time Optimizer ──────────────────────────────────────────
def calculate_dispatch_score(vehicle, emergency, grid):
    """
    Calculates a single score for dispatching this vehicle
    to this emergency.
    
    MERLIN's optimization objective:
    Minimize: (cost) + alpha × (response_time_minutes)
    
    Lower score = better dispatch choice.
    
    This is a simplified Linear Programming (LP) objective —
    the mathematical heart of MERLIN's optimization layer.
    
    In your paper this is described as:
    "We formulate dispatch selection as a cost-time
    minimization problem with weighted objectives..."
    
    Parameters:
      vehicle   → the vehicle being evaluated
      emergency → the emergency it would respond to
      grid      → for distance calculations
    
    Returns: float score (lower = better)
    """
    cell = emergency.cell

    # ── Distance from vehicle to emergency ───────────────────────
    distance_km = grid.distance_between_cells(
        vehicle.base_col, vehicle.base_row,
        cell.col, cell.row
    )

    # ── Response time in minutes (T_travel + T_spinup) ───────────
    response_time = vehicle.travel_time_ticks(distance_km)

    # ── Severity multiplier ──────────────────────────────────────
    # For higher severity, time matters MORE.
    # We amplify the time component based on severity.
    severity_multiplier = {
        SEVERITY_LOW:      0.5,   # time less critical
        SEVERITY_MEDIUM:   1.0,   # neutral
        SEVERITY_HIGH:     2.0,   # time twice as important
        SEVERITY_CRITICAL: 4.0,   # time four times as important
    }
    multiplier = severity_multiplier[emergency.severity]

    # ── The objective function ───────────────────────────────────
    # Minimize: cost + alpha × time × severity_multiplier
    score = (
        vehicle.cost_per_call +
        ALPHA_TIME_WEIGHT * response_time * multiplier
    )

    return score, distance_km, response_time


def select_best_vehicle(vehicles, emergency, grid):
    """
    From a list of available vehicles, selects the one
    with the lowest cost-time score.
    
    This is the optimization step — not just nearest,
    not just cheapest, but the mathematically optimal
    balance of cost and time given severity.
    
    Returns: (best_vehicle, score, distance_km, response_time)
    """
    if not vehicles:
        return None, float('inf'), 0, 0

    best_vehicle      = None
    best_score        = float('inf')
    best_distance     = 0
    best_response     = 0

    for vehicle in vehicles:
        score, dist, resp = calculate_dispatch_score(
            vehicle, emergency, grid
        )
        if score < best_score:
            best_score    = score
            best_vehicle  = vehicle
            best_distance = dist
            best_response = resp

    return best_vehicle, best_score, best_distance, best_response

# ── Dispatch Log Entry ───────────────────────────────────────────
class DispatchDecision:
    """
    Records everything about one dispatch decision.
    
    This is what gets logged to the results dataset.
    Every DispatchDecision becomes one row in the
    simulation output CSV — the data your paper
    analyses and your ML model trains on.
    """

    def __init__(self, emergency, vehicle,
                 decision_type, distance_km,
                 response_time_ticks, score,
                 validation_result, decision_reason):

        self.emergency         = emergency
        self.vehicle           = vehicle
        self.decision_type     = decision_type      # helicopter / ambulance
        self.distance_km       = distance_km
        self.response_time_ticks = response_time_ticks
        self.score             = score
        self.validation_result = validation_result
        self.decision_reason   = decision_reason    # why this decision

        # ── Features for ML training ─────────────────────────────
        # These are the X features your XGBoost model learns from
        self.features = {
            'severity':              emergency.severity,
            'emergency_type':        emergency.emergency_type,
            'patient_age':           emergency.patient_age,
            'population_density':    emergency.cell.population_density,
            'weather':               emergency.cell.weather,
            'terrain':               emergency.cell.terrain,
            'has_landing_zone':      int(emergency.cell.has_landing_zone),
            'distance_to_hospital':  emergency.cell.distance_to_hospital_km,
            'distance_to_vehicle':   distance_km,
            'response_time':         response_time_ticks,
            'season':                (emergency.tick // 131400) % 4,
            'hour_of_day':           (emergency.tick % 1440) // 60,
        }

        # ── Label for ML training ─────────────────────────────────
        # This is Y — what the model learns to predict
        self.label = 1 if decision_type == DISPATCH_HELICOPTER else 0

    def __repr__(self):
        return (
            f"Decision | {self.decision_type.upper()} | "
            f"Severity: {self.emergency.severity_name} | "
            f"Distance: {self.distance_km:.1f}km | "
            f"Response: {self.response_time_ticks}min | "
            f"Reason: {self.decision_reason}"
        )
    

    # ── Master Dispatch Engine ───────────────────────────────────────
class DispatchEngine:
    """
    MERLIN's central decision maker.
    
    Receives an emergency and fleet state.
    Returns the optimal dispatch decision.
    
    Decision process:
    1. Run safety validator — get hard constraints
    2. If forced decision exists → follow it
    3. If not forced → run cost-time optimizer
    4. Select best available vehicle
    5. Dispatch vehicle and log decision
    
    This engine is the core of MERLIN's novelty —
    the first system to combine rule-based safety
    validation with cost-time optimization for
    hybrid air-ground rural dispatch.
    """

    def __init__(self, grid, fleet):
        self.grid  = grid
        self.fleet = fleet
        self.decisions = []   # log of all decisions made

    def decide(self, emergency):
        """
        Main decision function. Called once per emergency.
        Returns a DispatchDecision object.
        """

        # ── Step 1: Run safety validator ─────────────────────────
        validation = validate_dispatch(emergency, self.fleet, self.grid)

        # ── Step 2: Handle no-vehicle case ───────────────────────
        if not validation.helicopter_valid and not validation.ambulance_valid:
            decision = DispatchDecision(
                emergency         = emergency,
                vehicle           = None,
                decision_type     = DISPATCH_NONE,
                distance_km       = 0,
                response_time_ticks = 0,
                score             = 0,
                validation_result = validation,
                decision_reason   = "no_vehicle_available"
            )
            self.decisions.append(decision)
            return decision

        # ── Step 3: Follow forced decision if exists ─────────────
        if validation.forced_decision == "helicopter":
            vehicle, score, dist, resp = select_best_vehicle(
                self.fleet.available_helicopters(emergency.cell),
                emergency, self.grid
            )
            decision_type   = DISPATCH_HELICOPTER
            decision_reason = validation.force_reason

        elif validation.forced_decision == "ambulance":
            vehicle, score, dist, resp = select_best_vehicle(
                self.fleet.available_ambulances(emergency.cell),
                emergency, self.grid
            )
            decision_type   = DISPATCH_AMBULANCE
            decision_reason = validation.force_reason

        else:
            # ── Step 4: No forced decision — run optimizer ────────
            # Score ALL available vehicles and pick best
            candidate_vehicles = []

            if validation.helicopter_valid:
                candidate_vehicles += self.fleet.available_helicopters(
                    emergency.cell
                )

            if validation.ambulance_valid:
                candidate_vehicles += self.fleet.available_ambulances(
                    emergency.cell
                )

            vehicle, score, dist, resp = select_best_vehicle(
                candidate_vehicles, emergency, self.grid
            )

            decision_type   = (
                DISPATCH_HELICOPTER
                if vehicle.vehicle_type == VehicleType.HELICOPTER
                else DISPATCH_AMBULANCE
            )
            decision_reason = "optimizer"

        # ── Step 5: Dispatch the vehicle ─────────────────────────
        if vehicle:
    vehicle.dispatch(emergency, emergency.call_tick, dist)
    emergency.dispatched_vehicle = decision_type
    emergency.dispatch_tick      = emergency.call_tick

    # ── Simulate arrival and return ───────────────────────────
    # Vehicle arrives at estimated_arrival tick
    # then immediately returns to base (ready for next call)
    # Runner.py will handle proper tick-by-tick tracking later.
    # For now: complete dispatch and free vehicle immediately.
    arrival = emergency.call_tick + dist
    vehicle.complete_dispatch(arrival)
    emergency.arrival_tick = arrival
    vehicle.return_to_base()

        # ── Step 6: Log and return decision ──────────────────────
        decision = DispatchDecision(
            emergency         = emergency,
            vehicle           = vehicle,
            decision_type     = decision_type,
            distance_km       = dist,
            response_time_ticks = resp,
            score             = score,
            validation_result = validation,
            decision_reason   = decision_reason
        )

        self.decisions.append(decision)
        return decision

    def summary(self):
        """
        Prints dispatch statistics after simulation run.
        These numbers become your paper's results section.
        """
        if not self.decisions:
            print("No decisions made yet.")
            return

        total     = len(self.decisions)
        heli      = sum(1 for d in self.decisions
                        if d.decision_type == DISPATCH_HELICOPTER)
        amb       = sum(1 for d in self.decisions
                        if d.decision_type == DISPATCH_AMBULANCE)
        none_disp = sum(1 for d in self.decisions
                        if d.decision_type == DISPATCH_NONE)

        avg_resp = (
            sum(d.response_time_ticks for d in self.decisions
                if d.response_time_ticks > 0) /
            max(1, total - none_disp)
        )

        forced_heli  = sum(1 for d in self.decisions
                           if d.decision_reason == FORCE_HELICOPTER)
        forced_ground = sum(1 for d in self.decisions
                            if d.decision_reason == FORCE_GROUND)
        optimized    = sum(1 for d in self.decisions
                           if d.decision_reason == "optimizer")

        print("\nMERLIN Dispatch Summary")
        print("─" * 50)
        print(f"Total emergencies:     {total}")
        print(f"Helicopter dispatches: {heli} ({heli/total*100:.1f}%)")
        print(f"Ambulance dispatches:  {amb} ({amb/total*100:.1f}%)")
        print(f"No vehicle available:  {none_disp}")
        print(f"Avg response time:     {avg_resp:.1f} minutes")
        print(f"")
        print(f"Decision reasons:")
        print(f"  Forced helicopter:   {forced_heli}")
        print(f"  Forced ground:       {forced_ground}")
        print(f"  Optimizer decided:   {optimized}")
        print("─" * 50)