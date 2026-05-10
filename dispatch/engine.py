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
DISPATCH_NONE       = "none"

# ── Cost-Time Trade-off Weight ───────────────────────────────────
# How much is one minute of response time worth in dollars?
# Higher alpha = system prioritises speed over cost.
# For Critical emergencies, time directly maps to survival.
# Alpha = 8.0 means each minute saved is valued at $8.
# This value can be tuned — testing different alphas
# is a sensitivity analysis in your paper.
ALPHA_TIME_WEIGHT = 8.0


# ── Cost-Time Score Calculator ───────────────────────────────────
def calculate_dispatch_score(vehicle, emergency, grid):
    """
    Calculates a single score for dispatching this vehicle
    to this emergency.

    MERLIN's optimization objective:
    Minimize: cost + alpha × response_time × severity_multiplier

    Lower score = better dispatch choice.

    This is MERLIN's Linear Programming (LP) objective —
    the mathematical core of the optimization layer.

    Returns: (score, distance_km, response_time_ticks)
    """

    cell = emergency.cell

    # Distance from vehicle base to emergency cell
    distance_km = grid.distance_between_cells(
        vehicle.base_col, vehicle.base_row,
        cell.col, cell.row
    )

    # Response time in minutes including spinup
    response_time = vehicle.travel_time_ticks(distance_km)

    # Severity multiplier — time matters more for critical cases
    severity_multiplier = {
        SEVERITY_LOW:      0.5,
        SEVERITY_MEDIUM:   1.0,
        SEVERITY_HIGH:     2.0,
        SEVERITY_CRITICAL: 4.0,
    }
    multiplier = severity_multiplier[emergency.severity]

    # The objective function
    score = (
        vehicle.cost_per_call +
        ALPHA_TIME_WEIGHT * response_time * multiplier
    )

    return score, distance_km, response_time


# ── Best Vehicle Selector ────────────────────────────────────────
def select_best_vehicle(vehicles, emergency, grid):
    """
    From a list of available vehicles, selects the one
    with the lowest cost-time score.

    For Low and Medium severity:
      Prefers ambulance — helicopter reserved for critical cases.
      Only uses helicopter if no ambulance is available.

    For High and Critical severity:
      Scores all vehicles and picks mathematically optimal.

    Returns: (best_vehicle, score, distance_km, response_time)
    """

    if not vehicles:
        return None, float('inf'), 0, 0

    # Severity-based vehicle preference filter
    # Low / Medium → prefer ambulance to preserve helicopter
    if emergency.severity in (SEVERITY_LOW, SEVERITY_MEDIUM):
        ambulances = [
            v for v in vehicles
            if v.vehicle_type == VehicleType.AMBULANCE
        ]
        if ambulances:
            vehicles = ambulances
        # If only helicopters available → allow helicopter
        # This is an edge case — all ambulances busy

    best_vehicle  = None
    best_score    = float('inf')
    best_distance = 0
    best_response = 0

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


# ── Dispatch Decision Record ─────────────────────────────────────
class DispatchDecision:
    """
    Records everything about one dispatch decision.

    Every DispatchDecision becomes one row in the
    simulation output dataset — the data your paper
    analyses and your ML model trains on.

    Contains:
    - The emergency details
    - The vehicle chosen
    - Why the decision was made
    - The features (X) for ML training
    - The label (Y) for ML training
    """

    def __init__(self, emergency, vehicle,
                 decision_type, distance_km,
                 response_time_ticks, score,
                 validation_result, decision_reason):

        self.emergency            = emergency
        self.vehicle              = vehicle
        self.decision_type        = decision_type
        self.distance_km          = distance_km
        self.response_time_ticks  = response_time_ticks
        self.score                = score
        self.validation_result    = validation_result
        self.decision_reason      = decision_reason

        # Features — the X inputs your ML model learns from
        self.features = {
            'severity':             emergency.severity,
            'emergency_type':       emergency.emergency_type,
            'patient_age':          emergency.patient_age,
            'population_density':   emergency.cell.population_density,
            'weather':              emergency.cell.weather,
            'terrain':              emergency.cell.terrain,
            'has_landing_zone':     int(emergency.cell.has_landing_zone),
            'distance_to_hospital': emergency.cell.distance_to_hospital_km,
            'distance_to_vehicle':  distance_km,
            'response_time':        response_time_ticks,
            'season':               (emergency.tick // 131400) % 4,
            'hour_of_day':          (emergency.tick % 1440) // 60,
        }

        # Label — Y value the model learns to predict
        # 1 = helicopter dispatched
        # 0 = ambulance dispatched
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

    Decision process every emergency:
    1. Run safety validator — check all hard constraints
    2. If forced decision exists → follow it (no optimizer)
    3. If no forced decision → run cost-time optimizer
    4. Select best available vehicle
    5. Dispatch vehicle, simulate arrival, return to base
    6. Log decision for paper results and ML training

    This is the core of MERLIN's novelty — the first system
    combining rule-based safety validation with cost-time
    optimization for hybrid air-ground rural dispatch.
    """

    def __init__(self, grid, fleet):
        self.grid      = grid
        self.fleet     = fleet
        self.decisions = []

    def decide(self, emergency):
        """
        Main decision function — called once per emergency.
        Returns a DispatchDecision object.
        """

        # ── Step 1: Run safety validator ─────────────────────────
        validation = validate_dispatch(
            emergency, self.fleet, self.grid
        )

        # ── Step 2: Handle no vehicle available ──────────────────
        if (not validation.helicopter_valid and
                not validation.ambulance_valid):

            decision = DispatchDecision(
                emergency           = emergency,
                vehicle             = None,
                decision_type       = DISPATCH_NONE,
                distance_km         = 0,
                response_time_ticks = 0,
                score               = 0,
                validation_result   = validation,
                decision_reason     = "no_vehicle_available"
            )
            self.decisions.append(decision)
            return decision

        # ── Step 3: Follow forced decision if exists ─────────────
        if validation.forced_decision == "helicopter":
            available = self.fleet.available_helicopters(
                emergency.cell
            )
            vehicle, score, dist, resp = select_best_vehicle(
                available, emergency, self.grid
            )
            decision_type   = DISPATCH_HELICOPTER
            decision_reason = validation.force_reason

        elif validation.forced_decision == "ambulance":
            available = self.fleet.available_ambulances(
                emergency.cell
            )
            vehicle, score, dist, resp = select_best_vehicle(
                available, emergency, self.grid
            )
            decision_type   = DISPATCH_AMBULANCE
            decision_reason = validation.force_reason

        else:
            # ── Step 4: No forced decision → run optimizer ────────
            candidate_vehicles = []

            if validation.helicopter_valid:
                candidate_vehicles += (
                    self.fleet.available_helicopters(emergency.cell)
                )

            if validation.ambulance_valid:
                candidate_vehicles += (
                    self.fleet.available_ambulances(emergency.cell)
                )

            vehicle, score, dist, resp = select_best_vehicle(
                candidate_vehicles, emergency, self.grid
            )

            decision_type = (
                DISPATCH_HELICOPTER
                if vehicle and
                   vehicle.vehicle_type == VehicleType.HELICOPTER
                else DISPATCH_AMBULANCE
            )
            decision_reason = "optimizer"

        # ── Step 5: Dispatch vehicle and simulate return ──────────
        if vehicle:
            # Dispatch — vehicle status → DISPATCHED
            vehicle.dispatch(
                emergency, emergency.call_tick, dist
            )

            # Write dispatch info into emergency object
            emergency.dispatched_vehicle = decision_type
            emergency.dispatch_tick      = emergency.call_tick

            # Simulate arrival
            arrival_tick         = emergency.call_tick + resp
            emergency.arrival_tick = arrival_tick

            # Complete dispatch — vehicle status → ON_SCENE
            vehicle.complete_dispatch(arrival_tick)

            # Return to base — vehicle status → AVAILABLE
            # Vehicles return immediately in this simulation.
            # Stated as a limitation in the paper.
            vehicle.return_to_base()

        # ── Step 6: Log decision ──────────────────────────────────
        decision = DispatchDecision(
            emergency           = emergency,
            vehicle             = vehicle,
            decision_type       = decision_type,
            distance_km         = dist,
            response_time_ticks = resp,
            score               = score,
            validation_result   = validation,
            decision_reason     = decision_reason
        )

        self.decisions.append(decision)
        return decision

    def summary(self):
        """
        Prints dispatch statistics.
        These numbers become your paper's results section.
        """
        if not self.decisions:
            print("No decisions made yet.")
            return

        total     = len(self.decisions)
        heli      = sum(
            1 for d in self.decisions
            if d.decision_type == DISPATCH_HELICOPTER
        )
        amb       = sum(
            1 for d in self.decisions
            if d.decision_type == DISPATCH_AMBULANCE
        )
        none_disp = sum(
            1 for d in self.decisions
            if d.decision_type == DISPATCH_NONE
        )

        responded = total - none_disp
        avg_resp  = (
            sum(
                d.response_time_ticks for d in self.decisions
                if d.response_time_ticks > 0
            ) / max(1, responded)
        )

        forced_heli  = sum(
            1 for d in self.decisions
            if d.decision_reason == FORCE_HELICOPTER
        )
        forced_ground = sum(
            1 for d in self.decisions
            if d.decision_reason == FORCE_GROUND
        )
        optimized = sum(
            1 for d in self.decisions
            if d.decision_reason == "optimizer"
        )

        print("\nMERLIN Dispatch Summary")
        print("─" * 50)
        print(f"Total emergencies:     {total}")
        print(f"Helicopter dispatches: {heli} "
              f"({heli/total*100:.1f}%)")
        print(f"Ambulance dispatches:  {amb} "
              f"({amb/total*100:.1f}%)")
        print(f"No vehicle available:  {none_disp}")
        print(f"Avg response time:     {avg_resp:.1f} minutes")
        print(f"")
        print(f"Decision reasons:")
        print(f"  Forced helicopter:   {forced_heli}")
        print(f"  Forced ground:       {forced_ground}")
        print(f"  Optimizer decided:   {optimized}")
        print("─" * 50)