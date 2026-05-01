# core/assets.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Assets Module — helicopters and ground ambulances

import numpy as np
from enum import Enum
from core.grid import (
    TERRAIN_WATER, TERRAIN_FOREST,
    WEATHER_CLEAR, WEATHER_CLOUDY,
    WEATHER_STORM, WEATHER_BLIZZARD
)

# ── Vehicle Status ───────────────────────────────────────────────
# Every vehicle is always in exactly one of these states.
class VehicleStatus(Enum):
    AVAILABLE   = "available"    # At base, ready to dispatch
    DISPATCHED  = "dispatched"   # En route to emergency
    ON_SCENE    = "on_scene"     # At patient location
    RETURNING   = "returning"    # Returning to base
    UNAVAILABLE = "unavailable"  # Maintenance, crew rest, etc.

# ── Vehicle Types ────────────────────────────────────────────────
class VehicleType(Enum):
    HELICOPTER = "helicopter"
    AMBULANCE  = "ambulance"

# ── Operational Constants ────────────────────────────────────────
# Helicopter (HEMS — Helicopter Emergency Medical Services)
HELICOPTER_SPEED_KMH     = 250   # Cruise speed in km/h
HELICOPTER_COST_PER_CALL = 10000 # Average cost per dispatch in CAD
HELICOPTER_SPINUP_TICKS  = 10    # Minutes to prepare before departure
HELICOPTER_RANGE_KM      = 400   # Maximum range before refuel

# Ground Ambulance (EMS — Emergency Medical Services)
AMBULANCE_SPEED_KMH      = 80    # Average rural road speed in km/h
AMBULANCE_COST_PER_CALL  = 600   # Average cost per dispatch in CAD
AMBULANCE_SPINUP_TICKS   = 2     # Minutes to prepare before departure

# ── Weather Limits for Helicopter ───────────────────────────────
# These are Transport Canada operational minimums.
# Helicopter is grounded in Storm or Blizzard.
HELICOPTER_FLYABLE_WEATHER = {
    WEATHER_CLEAR:    True,
    WEATHER_CLOUDY:   True,
    WEATHER_STORM:    False,
    WEATHER_BLIZZARD: False
}

# ── Base Vehicle Class ───────────────────────────────────────────
class Vehicle:
    """
    Base class for all MERLIN emergency vehicles.
    
    Both helicopters and ground ambulances inherit from this.
    Tracks location, status, mission history, and costs.
    
    In MERLIN's simulation, each vehicle is one object
    that moves through states as the simulation runs:
    AVAILABLE → DISPATCHED → ON_SCENE → RETURNING → AVAILABLE
    """

    def __init__(self, vehicle_id, vehicle_type,
                 base_col, base_row,
                 speed_kmh, cost_per_call, spinup_ticks):

        # ── Identity ─────────────────────────────────────────────
        self.vehicle_id   = vehicle_id
        self.vehicle_type = vehicle_type

        # ── Base location (where vehicle returns after each call) 
        self.base_col = base_col
        self.base_row = base_row

        # ── Current location (updates during simulation) ─────────
        self.current_col = base_col
        self.current_row = base_row

        # ── Operational parameters ───────────────────────────────
        self.speed_kmh     = speed_kmh
        self.cost_per_call = cost_per_call
        self.spinup_ticks  = spinup_ticks

        # ── Status ───────────────────────────────────────────────
        self.status = VehicleStatus.AVAILABLE

        # ── Current mission (filled when dispatched) ─────────────
        self.current_emergency  = None
        self.dispatch_tick      = None
        self.estimated_arrival  = None

        # ── Statistics (accumulated over simulation) ─────────────
        self.total_dispatches   = 0
        self.total_cost         = 0.0
        self.total_response_time_ticks = 0

    def is_available(self):
        """Returns True if vehicle can be dispatched right now."""
        return self.status == VehicleStatus.AVAILABLE

    def travel_time_ticks(self, distance_km):
        """
        Calculates travel time in ticks (minutes) for a given
        distance in kilometres.
        
        Formula: time = distance / speed × 60 (convert hours to minutes)
        Plus spinup time before departure.
        
        Example:
          Helicopter: 80km / 250km/h × 60 = 19.2 min + 10 spinup = 29 ticks
          Ambulance:  80km / 80km/h  × 60 = 60.0 min + 2  spinup = 62 ticks
        """
        travel_minutes = (distance_km / self.speed_kmh) * 60
        return int(travel_minutes) + self.spinup_ticks

    def dispatch(self, emergency, current_tick, distance_km):
        """
        Dispatches this vehicle to an emergency.
        Updates status, records mission details, accumulates cost.
        """
        self.status            = VehicleStatus.DISPATCHED
        self.current_emergency = emergency
        self.dispatch_tick     = current_tick
        self.estimated_arrival = current_tick + self.travel_time_ticks(distance_km)

        # Update statistics
        self.total_dispatches += 1
        self.total_cost       += self.cost_per_call

    def complete_dispatch(self, arrival_tick):
        """
        Called when vehicle arrives at emergency scene.
        Updates status and response time statistics.
        """
        self.status = VehicleStatus.ON_SCENE
        if self.dispatch_tick is not None:
            self.total_response_time_ticks += (
                arrival_tick - self.dispatch_tick
            )

    def return_to_base(self):
        """
        Called when vehicle finishes at scene and returns to base.
        Resets to AVAILABLE for next dispatch.
        """
        self.status            = VehicleStatus.RETURNING
        self.current_col       = self.base_col
        self.current_row       = self.base_row
        self.current_emergency = None
        self.dispatch_tick     = None
        self.estimated_arrival = None

        # After returning, immediately available again
        self.status = VehicleStatus.AVAILABLE

    @property
    def average_response_time(self):
        """Average response time in minutes across all dispatches."""
        if self.total_dispatches == 0:
            return 0.0
        return self.total_response_time_ticks / self.total_dispatches

    def __repr__(self):
        return (
            f"{self.vehicle_type.value.capitalize()} "
            f"[{self.vehicle_id}] | "
            f"Status: {self.status.value} | "
            f"Base: ({self.base_col},{self.base_row}) | "
            f"Dispatches: {self.total_dispatches}"
        )


# ── Helicopter Class ─────────────────────────────────────────────
class Helicopter(Vehicle):
    """
    HEMS (Helicopter Emergency Medical Services) vehicle.
    
    Faster and more capable than ground ambulance but:
    - Cannot fly in storm or blizzard weather
    - Needs a landing zone at patient location
    - Costs ~16× more per call than ambulance
    - Limited to 1-2 units per region
    
    In Ontario, this represents an ORNGE helicopter
    or a STARS (Shock Trauma Air Rescue Service) unit.
    """

    def __init__(self, vehicle_id, base_col, base_row):
        super().__init__(
            vehicle_id   = vehicle_id,
            vehicle_type = VehicleType.HELICOPTER,
            base_col     = base_col,
            base_row     = base_row,
            speed_kmh    = HELICOPTER_SPEED_KMH,
            cost_per_call = HELICOPTER_COST_PER_CALL,
            spinup_ticks = HELICOPTER_SPINUP_TICKS
        )

    def can_fly(self, weather):
        """
        Returns True if current weather allows flight.
        Checks Transport Canada operational minimums.
        """
        return HELICOPTER_FLYABLE_WEATHER.get(weather, False)

    def can_land(self, cell):
        """
        Returns True if helicopter can land near this cell.
        Checks both terrain and landing zone availability.
        """
        if not cell.has_landing_zone:
            return False
        if cell.terrain == TERRAIN_WATER:
            return False
        return True

    def is_dispatchable(self, cell):
        """
        Master check — can this helicopter respond to
        an emergency in this cell right now?
        
        All three conditions must be true:
        1. Vehicle is available (not on another call)
        2. Weather allows flight
        3. Can land at destination
        """
        return (
            self.is_available() and
            self.can_fly(cell.weather) and
            self.can_land(cell)
        )


# ── Ambulance Class ──────────────────────────────────────────────
class Ambulance(Vehicle):
    """
    Ground EMS (Emergency Medical Services) ambulance.
    
    Available in most conditions, multiple units per region,
    lower cost — but significantly slower on rural distances.
    
    Crew level affects what care can be delivered on scene:
    - ALS (Advanced Life Support): full paramedic crew
    - BLS (Basic Life Support): emergency medical technician crew
    """

    def __init__(self, vehicle_id, base_col, base_row,
                 crew_level="ALS"):
        super().__init__(
            vehicle_id    = vehicle_id,
            vehicle_type  = VehicleType.AMBULANCE,
            base_col      = base_col,
            base_row      = base_row,
            speed_kmh     = AMBULANCE_SPEED_KMH,
            cost_per_call = AMBULANCE_COST_PER_CALL,
            spinup_ticks  = AMBULANCE_SPINUP_TICKS
        )
        # ALS = Advanced Life Support (full paramedic)
        # BLS = Basic Life Support (EMT level)
        self.crew_level = crew_level

    def is_dispatchable(self, cell):
        """
        Ground ambulance can respond if:
        1. Vehicle is available
        2. Cell is not water-only terrain
           (no road access = ambulance physically cannot reach)
        """
        if not self.is_available():
            return False
        if cell.terrain == TERRAIN_WATER:
            return False
        return True

# ── Fleet Class ──────────────────────────────────────────────────
class Fleet:
    """
    Manages all vehicles in MERLIN's simulation.
    
    Provides methods to:
    - Find available vehicles for a given emergency
    - Track overall fleet statistics
    - Reset fleet between simulation runs
    
    In rural Ontario this fleet represents:
    - 1 ORNGE helicopter (based at regional centre)
    - 3-4 ground ambulances (distributed across towns)
    """

    def __init__(self):
        self.helicopters = []
        self.ambulances  = []

    def add_helicopter(self, helicopter):
        self.helicopters.append(helicopter)

    def add_ambulance(self, ambulance):
        self.ambulances.append(ambulance)

    def all_vehicles(self):
        """Returns all vehicles in the fleet."""
        return self.helicopters + self.ambulances

    def available_helicopters(self, cell):
        """
        Returns list of helicopters that can be dispatched
        to a specific cell right now.
        Checks availability, weather, and landing zone.
        """
        return [
            h for h in self.helicopters
            if h.is_dispatchable(cell)
        ]

    def available_ambulances(self, cell):
        """
        Returns list of ambulances that can be dispatched
        to a specific cell right now.
        """
        return [
            a for a in self.ambulances
            if a.is_dispatchable(cell)
        ]

    def nearest_vehicle(self, vehicles, grid, target_col, target_row):
        """
        From a list of available vehicles, returns the one
        whose base is closest to the target cell.
        
        This is Baseline 2 in our paper — the nearest
        vehicle rule that MERLIN is compared against.
        """
        if not vehicles:
            return None

        nearest = None
        min_dist = float('inf')

        for v in vehicles:
            dist = grid.distance_between_cells(
                v.current_col, v.current_row,
                target_col, target_row
            )
            if dist < min_dist:
                min_dist = dist
                nearest  = v

        return nearest

    def reset_all(self):
        """
        Resets all vehicles to base — used between
        simulation runs to start fresh.
        """
        for v in self.all_vehicles():
            v.return_to_base()
            v.total_dispatches  = 0
            v.total_cost        = 0.0
            v.total_response_time_ticks = 0

    def summary(self):
        """Prints fleet statistics after a simulation run."""
        print("\nFleet Summary")
        print("─" * 50)
        for v in self.all_vehicles():
            print(
                f"{v} | "
                f"Avg response: {v.average_response_time:.1f} min | "
                f"Total cost: ${v.total_cost:,.0f}"
            )
        total_cost = sum(v.total_cost for v in self.all_vehicles())
        total_dispatches = sum(
            v.total_dispatches for v in self.all_vehicles()
        )
        print("─" * 50)
        print(f"Total dispatches: {total_dispatches}")
        print(f"Total cost:       ${total_cost:,.0f}")

    def __repr__(self):
        return (
            f"Fleet | "
            f"{len(self.helicopters)} helicopter(s) | "
            f"{len(self.ambulances)} ambulance(s)"
        )


# ── Rural Ontario Fleet Builder ──────────────────────────────────
def build_rural_ontario_fleet():
    """
    Builds a realistic fleet for rural Ontario simulation.
    
    Based on ORNGE operational data:
    - 1 helicopter based at regional centre (grid centre)
    - 3 ambulances distributed across the three towns
    
    This intentionally models resource scarcity —
    the core problem MERLIN is solving.
    """
    fleet = Fleet()

    # ── ORNGE Helicopter ─────────────────────────────────────────
    fleet.add_helicopter(Helicopter('ORNGE-1', base_col=5, base_row=5))

    # ── Ground Ambulances ────────────────────────────────────────
    fleet.add_ambulance(Ambulance('AMB-1', base_col=2, base_row=2, crew_level='ALS'))
    fleet.add_ambulance(Ambulance('AMB-2', base_col=7, base_row=3, crew_level='ALS'))
    fleet.add_ambulance(Ambulance('AMB-3', base_col=4, base_row=7, crew_level='BLS'))

    return fleet