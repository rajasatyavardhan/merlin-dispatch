# simulation/runner.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Simulation Runner — runs full year simulation and baseline comparisons
#
# This file produces the results for your research paper.
# Three baselines are compared against MERLIN:
#
# Baseline 1 — Random Dispatch
#   Randomly sends helicopter or ambulance regardless of anything
#   This is the worst possible dispatcher
#   If MERLIN can't beat this, something is deeply wrong
#
# Baseline 2 — Nearest Vehicle
#   Always sends the closest available vehicle
#   Simple rule used in basic dispatch software
#   No consideration of severity, cost, or weather
#
# Baseline 3 — Severity Rule
#   If severity is High or Critical → helicopter
#   Otherwise → ambulance
#   This is how real human dispatchers roughly operate
#   MERLIN must beat this to claim novelty
#
# MERLIN — full system
#   Safety validation + cost-time optimization
#   The system we are building

import random
import numpy as np
import pandas as pd
from collections import defaultdict

from core.grid import (
    TERRAIN_WATER,
    WEATHER_CLEAR,
    WEATHER_CLOUDY,
    build_rural_ontario_map,
    generate_emergencies,
    SEVERITY_HIGH, SEVERITY_CRITICAL,
    SEVERITY_NAMES, WEATHER_STORM, WEATHER_BLIZZARD
)
from core.assets import (
    build_rural_ontario_fleet,
    VehicleType
)
from core.emergency import generate_emergency_event
from dispatch.engine import (
    DispatchEngine,
    DISPATCH_HELICOPTER,
    DISPATCH_AMBULANCE,
    DISPATCH_NONE
)

# ── Simulation Configuration ─────────────────────────────────────
SIM_TICKS         = 525600   # One full year (60 × 24 × 365)
SIM_TICKS_SHORT   = 50000    # Short run for testing
RANDOM_SEED       = 42       # Fixed seed for reproducibility

# ── Survival Probability Model ───────────────────────────────────
# Based on clinical literature:
# Cardiac arrest survival decreases ~10% per 10 minutes
# without ALS (Advanced Life Support) intervention.
# We use this to calculate a survival proxy metric.
SURVIVAL_BASE = {
    0: 0.95,   # Low severity — high baseline survival
    1: 0.85,   # Medium
    2: 0.65,   # High
    3: 0.40,   # Critical — lower baseline without fast response
}
SURVIVAL_DECAY_PER_MINUTE = 0.008  # 0.8% per minute


def update_weather(grid, tick):
    """
    Updates weather across all grid cells.
    Called every 1440 ticks (once per simulated day).

    Seasonal storm probability:
      Spring: 10%  Summer: 5%
      Autumn: 15%  Winter: 35%

    Based on Environment Canada northern Ontario
    weather pattern data.

    This creates real weather variation in the dataset
    so the ML model can learn weather's effect on
    emergency severity and dispatch outcomes.
    """
    season = (tick // 131400) % 4

    storm_probability = {
        0: 0.10,   # Spring
        1: 0.05,   # Summer
        2: 0.15,   # Autumn
        3: 0.35,   # Winter
    }[season]

    for col in range(grid.width):
        for row in range(grid.height):
            cell = grid.get_cell(col, row)

            # Water cells always have worse weather
            # (open water = more exposed conditions)
            local_prob = storm_probability
            if cell.terrain == TERRAIN_WATER:
                local_prob = min(1.0, storm_probability * 1.3)

            r = random.random()
            if r < local_prob * 0.3:
                cell.weather = WEATHER_BLIZZARD
            elif r < local_prob:
                cell.weather = WEATHER_STORM
            elif r < local_prob + 0.25:
                cell.weather = WEATHER_CLOUDY
            else:
                cell.weather = WEATHER_CLEAR

# ── Baseline Dispatchers ─────────────────────────────────────────

def baseline_random(emergency, fleet, grid):
    """
    Baseline 1 — Random Dispatch.

    Randomly chooses helicopter or ambulance with equal
    probability, ignoring all other factors.

    This is the worst possible dispatcher and the floor
    that MERLIN must exceed. If MERLIN cannot beat random
    dispatch, the system has no value.

    Returns: (decision_type, vehicle, distance, response_time)
    """
    available_helis = fleet.available_helicopters(emergency.cell)
    available_ambs  = fleet.available_ambulances(emergency.cell)

    if not available_helis and not available_ambs:
        return DISPATCH_NONE, None, 0, 0

    # Flip a coin
    if available_helis and available_ambs:
        use_helicopter = random.random() < 0.5
    elif available_helis:
        use_helicopter = True
    else:
        use_helicopter = False

    if use_helicopter:
        vehicle = available_helis[0]
        decision = DISPATCH_HELICOPTER
    else:
        vehicle = fleet.nearest_vehicle(
            available_ambs, grid,
            emergency.cell.col, emergency.cell.row
        )
        decision = DISPATCH_AMBULANCE

    if vehicle is None:
        return DISPATCH_NONE, None, 0, 0

    dist = grid.distance_between_cells(
        vehicle.base_col, vehicle.base_row,
        emergency.cell.col, emergency.cell.row
    )
    resp = vehicle.travel_time_ticks(dist)

    vehicle.dispatch(emergency, emergency.call_tick, dist)
    vehicle.complete_dispatch(emergency.call_tick + resp)
    vehicle.return_to_base()

    return decision, vehicle, dist, resp


def baseline_nearest(emergency, fleet, grid):
    """
    Baseline 2 — Nearest Vehicle.

    Always sends the closest available vehicle regardless
    of type, severity, cost, or weather prediction.

    This represents basic dispatch software that
    optimises only for proximity — no intelligence.

    Returns: (decision_type, vehicle, distance, response_time)
    """
    available_helis = fleet.available_helicopters(emergency.cell)
    available_ambs  = fleet.available_ambulances(emergency.cell)
    all_available   = available_helis + available_ambs

    if not all_available:
        return DISPATCH_NONE, None, 0, 0

    # Find nearest regardless of type
    vehicle = fleet.nearest_vehicle(
        all_available, grid,
        emergency.cell.col, emergency.cell.row
    )

    if vehicle is None:
        return DISPATCH_NONE, None, 0, 0

    decision = (
        DISPATCH_HELICOPTER
        if vehicle.vehicle_type == VehicleType.HELICOPTER
        else DISPATCH_AMBULANCE
    )

    dist = grid.distance_between_cells(
        vehicle.base_col, vehicle.base_row,
        emergency.cell.col, emergency.cell.row
    )
    resp = vehicle.travel_time_ticks(dist)

    vehicle.dispatch(emergency, emergency.call_tick, dist)
    vehicle.complete_dispatch(emergency.call_tick + resp)
    vehicle.return_to_base()

    return decision, vehicle, dist, resp


def baseline_severity_rule(emergency, fleet, grid):
    """
    Baseline 3 — Severity Rule (most important baseline).

    If severity is High or Critical → send helicopter.
    Otherwise → send ambulance.

    This is how real human dispatchers roughly operate.
    MERLIN must outperform this baseline to justify
    its existence as an AI system.

    Returns: (decision_type, vehicle, distance, response_time)
    """
    available_helis = fleet.available_helicopters(emergency.cell)
    available_ambs  = fleet.available_ambulances(emergency.cell)

    # Decide based on severity threshold
    want_helicopter = (
        emergency.severity in (SEVERITY_HIGH, SEVERITY_CRITICAL)
        and len(available_helis) > 0
    )

    if want_helicopter:
        vehicle  = available_helis[0]
        decision = DISPATCH_HELICOPTER
    elif available_ambs:
        vehicle = fleet.nearest_vehicle(
            available_ambs, grid,
            emergency.cell.col, emergency.cell.row
        )
        decision = DISPATCH_AMBULANCE
    elif available_helis:
        # Fallback — severity says ambulance but none available
        vehicle  = available_helis[0]
        decision = DISPATCH_HELICOPTER
    else:
        return DISPATCH_NONE, None, 0, 0

    if vehicle is None:
        return DISPATCH_NONE, None, 0, 0

    dist = grid.distance_between_cells(
        vehicle.base_col, vehicle.base_row,
        emergency.cell.col, emergency.cell.row
    )
    resp = vehicle.travel_time_ticks(dist)

    vehicle.dispatch(emergency, emergency.call_tick, dist)
    vehicle.complete_dispatch(emergency.call_tick + resp)
    vehicle.return_to_base()

    return decision, vehicle, dist, resp

# ── Metrics Calculator ───────────────────────────────────────────
def calculate_survival_probability(severity, response_time_minutes):
    """
    Calculates estimated survival probability based on
    severity level and response time.

    Based on clinical literature:
    - Cardiac arrest: ~10% survival decrease per 10 min
    - General principle: faster response = better outcome

    This is a proxy metric — not a clinical claim.
    Stated explicitly as a proxy in the paper.

    Returns: float between 0.0 and 1.0
    """
    base     = SURVIVAL_BASE[severity]
    decay    = SURVIVAL_DECAY_PER_MINUTE * response_time_minutes
    survival = max(0.0, base - decay)
    return round(survival, 4)


def calculate_metrics(results_list):
    """
    Calculates all evaluation metrics from a list of
    simulation result dictionaries.

    These metrics become the columns in your paper's
    results table — Table 1 in your paper.

    Metrics calculated:
    1. Mean response time (minutes)
    2. Median response time
    3. Helicopter utilization rate (%)
    4. Mean survival probability (proxy)
    5. Total operational cost ($)
    6. Cost per emergency ($)
    7. Critical case helicopter rate (%)
    8. No-vehicle rate (% emergencies with no response)

    Returns: dict of metric name → value
    """
    if not results_list:
        return {}

    total = len(results_list)

    # Filter responded emergencies
    responded = [r for r in results_list
                 if r['decision'] != DISPATCH_NONE]

    if not responded:
        return {'total_emergencies': total, 'no_vehicle_rate': 1.0}

    # Response times
    response_times = [r['response_time'] for r in responded]
    mean_resp  = np.mean(response_times)
    median_resp = np.median(response_times)

    # Helicopter rate
    heli_dispatches = sum(
        1 for r in responded
        if r['decision'] == DISPATCH_HELICOPTER
    )
    heli_rate = heli_dispatches / total

    # Critical case helicopter rate
    critical_cases = [
        r for r in responded
        if r['severity'] == SEVERITY_CRITICAL
    ]
    if critical_cases:
        critical_heli = sum(
            1 for r in critical_cases
            if r['decision'] == DISPATCH_HELICOPTER
        )
        critical_heli_rate = critical_heli / len(critical_cases)
    else:
        critical_heli_rate = 0.0

    # Survival probability
    survival_probs = [
        calculate_survival_probability(
            r['severity'], r['response_time']
        )
        for r in responded
    ]
    mean_survival = np.mean(survival_probs)

    # Costs
    total_cost = sum(r['cost'] for r in responded)
    cost_per_emergency = total_cost / len(responded)

    # No vehicle rate
    no_vehicle = sum(
        1 for r in results_list
        if r['decision'] == DISPATCH_NONE
    )
    no_vehicle_rate = no_vehicle / total

    return {
        'total_emergencies':      total,
        'mean_response_time':     round(mean_resp, 2),
        'median_response_time':   round(median_resp, 2),
        'helicopter_rate':        round(heli_rate * 100, 1),
        'critical_heli_rate':     round(critical_heli_rate * 100, 1),
        'mean_survival_prob':     round(mean_survival * 100, 2),
        'total_cost':             round(total_cost, 0),
        'cost_per_emergency':     round(cost_per_emergency, 0),
        'no_vehicle_rate':        round(no_vehicle_rate * 100, 2),
    }

# ── Single System Runner ─────────────────────────────────────────
def run_simulation(system_name, num_ticks=SIM_TICKS_SHORT,
                   seed=RANDOM_SEED, verbose=True):
    """
    Runs one complete simulation for one dispatch system.

    Generates all emergencies using a fixed random seed —
    this ensures all four systems (3 baselines + MERLIN)
    face EXACTLY the same emergencies.

    This is critical for fair comparison in your paper.
    All systems are evaluated on identical scenarios.

    Returns: (results_list, metrics_dict)
    """
    # Fixed seed ensures reproducibility
    random.seed(seed)
    np.random.seed(seed)

    grid  = build_rural_ontario_map()
    fleet = build_rural_ontario_fleet()

    # For MERLIN — use the full engine
    if system_name == "MERLIN":
        engine = DispatchEngine(grid, fleet)

    results = []

    if verbose:
        print(f"Running {system_name} — {num_ticks:,} ticks...")

    for tick in range(num_ticks):
        # Update weather daily
        if tick % 1440 == 0:
            update_weather(grid, tick)
            
        # Generate emergencies this tick
        triggered_cells = generate_emergencies(grid)

        for cell in triggered_cells:
            # Create emergency event
            emergency = generate_emergency_event(cell, tick)

            # Dispatch based on system type
            if system_name == "MERLIN":
                decision_obj = engine.decide(emergency)
                decision  = decision_obj.decision_type
                dist      = decision_obj.distance_km
                resp      = decision_obj.response_time_ticks
                cost      = (
                    decision_obj.vehicle.cost_per_call
                    if decision_obj.vehicle else 0
                )

            elif system_name == "Random":
                decision, vehicle, dist, resp = baseline_random(
                    emergency, fleet, grid
                )
                cost = vehicle.cost_per_call if vehicle else 0

            elif system_name == "Nearest":
                decision, vehicle, dist, resp = baseline_nearest(
                    emergency, fleet, grid
                )
                cost = vehicle.cost_per_call if vehicle else 0

            elif system_name == "SeverityRule":
                decision, vehicle, dist, resp = (
                    baseline_severity_rule(emergency, fleet, grid)
                )
                cost = vehicle.cost_per_call if vehicle else 0

            else:
                raise ValueError(f"Unknown system: {system_name}")
            # Record result — now saves ALL features
            results.append({
                'tick':                 tick,
                'system':               system_name,
                'severity':             emergency.severity,
                'emergency_type':       emergency.emergency_type,
                'patient_age':          emergency.patient_age,
                'decision':             decision,
                'distance_km':          dist,
                'response_time':        resp,
                'cost':                 cost,
                'col':                  cell.col,
                'row':                  cell.row,
                
                # ── All ML features now saved ──────────────────────
                'population_density':   cell.population_density,
                'weather':              cell.weather,
                'terrain':              cell.terrain,
                'has_landing_zone':     int(cell.has_landing_zone),
                'distance_to_hospital': cell.distance_to_hospital_km,
                'distance_to_vehicle':  dist,
                'season':               (tick // 131400) % 4,
                'hour_of_day':          (tick % 1440) // 60,
                })
    metrics = calculate_metrics(results)

    if verbose:
        print(f"  Emergencies: {metrics.get('total_emergencies', 0)}")
        print(
            f"  Mean response: "
            f"{metrics.get('mean_response_time', 0):.1f} min"
        )
        print(
            f"  Survival prob: "
            f"{metrics.get('mean_survival_prob', 0):.1f}%"
        )
        print(
            f"  Total cost: "
            f"${metrics.get('total_cost', 0):,.0f}"
        )

    return results, metrics

# ── Full Comparison Runner ───────────────────────────────────────
def run_full_comparison(num_ticks=SIM_TICKS_SHORT,
                        seed=RANDOM_SEED):
    """
    Runs all four systems on identical emergencies
    and produces the comparison table for your paper.

    THIS IS YOUR PAPER'S TABLE 1.

    All systems use the same random seed — meaning they
    face exactly the same emergencies in the same order.
    This is the only fair way to compare dispatch systems.

    Returns: DataFrame with all metrics side by side
    """
    print("=" * 60)
    print("MERLIN — Full System Comparison")
    print(f"Simulation length: {num_ticks:,} ticks "
          f"({num_ticks/525600*100:.1f}% of one year)")
    print(f"Random seed: {seed} (fixed for reproducibility)")
    print("=" * 60)
    print()

    systems = ["Random", "Nearest", "SeverityRule", "MERLIN"]
    all_metrics = {}
    all_results = {}

    for system in systems:
        results, metrics = run_simulation(
            system, num_ticks=num_ticks,
            seed=seed, verbose=True
        )
        all_metrics[system] = metrics
        all_results[system] = results
        print()

    # ── Build comparison table ───────────────────────────────────
    print("=" * 60)
    print("RESULTS TABLE — Paper Table 1")
    print("=" * 60)

    metrics_to_show = [
        ('total_emergencies',   'Total Emergencies',        ''),
        ('mean_response_time',  'Mean Response Time',        'min'),
        ('median_response_time','Median Response Time',      'min'),
        ('mean_survival_prob',  'Mean Survival Probability', '%'),
        ('helicopter_rate',     'Helicopter Dispatch Rate',  '%'),
        ('critical_heli_rate',  'Critical → Helicopter',    '%'),
        ('total_cost',          'Total Operational Cost',    '$'),
        ('cost_per_emergency',  'Cost Per Emergency',        '$'),
        ('no_vehicle_rate',     'No Vehicle Rate',           '%'),
    ]

    # Header
    header = f"{'Metric':<30}"
    for s in systems:
        header += f"{s:>15}"
    print(header)
    print("-" * (30 + 15 * len(systems)))

    # Rows
    for key, label, unit in metrics_to_show:
        row = f"{label:<30}"
        for system in systems:
            val = all_metrics[system].get(key, 0)
            if unit == '$':
                row += f"{'${:,.0f}'.format(val):>15}"
            else:
                row += f"{'{:.1f}{}'.format(val, unit):>15}"
        print(row)

    print("=" * 60)

    # ── Save results to CSV ──────────────────────────────────────
    all_rows = []
    for system, results in all_results.items():
        all_rows.extend(results)

    df = pd.DataFrame(all_rows)
    df.to_csv('data/synthetic/simulation_results.csv', index=False)
    print()
    print("Results saved to: data/synthetic/simulation_results.csv")
    print("Use this CSV for ML model training and paper figures.")

    return all_metrics, df

