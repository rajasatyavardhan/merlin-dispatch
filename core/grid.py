# core/grid.py
# MERLIN — Medical Emergency Routing and Live Intelligence Network
# Grid Module — builds and manages the rural region simulation

import numpy as np
import random


# ── Weather States ────────────────────────────────────────────────
# Represents environmental conditions in each cell.
# These directly affect whether helicopter dispatch is possible.

WEATHER_CLEAR   = 0   # Ideal flying conditions
WEATHER_CLOUDY  = 1   # Acceptable but with caution
WEATHER_STORM   = 2   # Unsafe — no helicopter allowed
WEATHER_BLIZZARD = 3  # Extremely unsafe — no helicopter

# Human-readable names for debugging / logging
WEATHER_NAMES = {
    WEATHER_CLEAR:    "Clear",
    WEATHER_CLOUDY:   "Cloudy",
    WEATHER_STORM:    "Storm",
    WEATHER_BLIZZARD: "Blizzard"
}


# ── Terrain Types ────────────────────────────────────────────────
# Defines physical geography of each cell.
# Affects landing feasibility and routing decisions.

TERRAIN_FLAT    = 0   # Easy landing and road access
TERRAIN_HILLY   = 1   # Moderate difficulty
TERRAIN_FOREST  = 2   # Limited landing zones
TERRAIN_WATER   = 3   # No road access — helicopter only

TERRAIN_NAMES = {
    TERRAIN_FLAT:   "Flat",
    TERRAIN_HILLY:  "Hilly",
    TERRAIN_FOREST: "Forest",
    TERRAIN_WATER:  "Water"
}


# ── Severity Levels ──────────────────────────────────────────────
# Used across entire MERLIN system — classification target later

SEVERITY_LOW      = 0
SEVERITY_MEDIUM   = 1
SEVERITY_HIGH     = 2
SEVERITY_CRITICAL = 3

SEVERITY_NAMES = {
    SEVERITY_LOW:      "Low",
    SEVERITY_MEDIUM:   "Medium",
    SEVERITY_HIGH:     "High",
    SEVERITY_CRITICAL: "Critical"
}


# ── Simulation Constants ─────────────────────────────────────────
# Defines time scaling and base emergency probability

BASE_EMERGENCY_RATE = 0.001   # Probability per tick at density=1.0
TICKS_PER_HOUR      = 60      # 1 tick = 1 minute
TICKS_PER_DAY       = 1440    # 24 hours
TICKS_PER_YEAR      = 525600  # Full simulation year


# ── Cell Class ───────────────────────────────────────────────────
class Cell:
    """
    Represents a single square in the simulation grid.

    Each cell models:
    - Population density
    - Terrain type
    - Weather conditions
    - Distance to hospital

    This is the fundamental unit where emergencies occur.
    """

    def __init__(self, col, row, population_density=0.0,
                 terrain=TERRAIN_FLAT, has_landing_zone=True,
                 distance_to_hospital_km=50.0):

        # ── Grid Position ────────────────────────────────────────
        self.col = col
        self.row = row

        # ── Population ───────────────────────────────────────────
        # Determines likelihood of emergencies
        self.population_density = population_density

        # Bernoulli probability (p)
        self.emergency_probability = population_density * BASE_EMERGENCY_RATE

        # ── Geography ────────────────────────────────────────────
        self.terrain = terrain
        self.has_landing_zone = has_landing_zone
        self.distance_to_hospital_km = distance_to_hospital_km

        # ── Weather ──────────────────────────────────────────────
        self.weather = WEATHER_CLEAR  # Default state

    def check_emergency(self):
        """
        Performs a Bernoulli trial:
        Returns True if an emergency occurs this tick.
        """
        return random.random() < self.emergency_probability

    def can_helicopter_land(self):
        """
        Determines if helicopter landing is physically possible.
        """
        if not self.has_landing_zone:
            return False
        if self.terrain == TERRAIN_WATER:
            return False
        return True

    def is_helicopter_weather_safe(self):
        """
        Determines if weather allows helicopter dispatch.
        """
        return self.weather in (WEATHER_CLEAR, WEATHER_CLOUDY)

    def __repr__(self):
        """
        Debug-friendly representation of the cell.
        """
        return (
            f"Cell({self.col},{self.row}) | "
            f"density={self.population_density:.2f} | "
            f"{WEATHER_NAMES[self.weather]} | "
            f"{TERRAIN_NAMES[self.terrain]}"
        )


# ── Grid Class ───────────────────────────────────────────────────
class Grid:
    """
    Represents the full simulation environment.

    A 2D grid of Cell objects covering a rural region.
    """

    def __init__(self, width=10, height=10):

        # ── Dimensions ──────────────────────────────────────────
        self.width  = width
        self.height = height

        # ── Grid Construction ───────────────────────────────────
        self.cells = [
            [Cell(col, row) for row in range(height)]
            for col in range(width)
        ]

        # ── Hospital Location ───────────────────────────────────
        # Positioned at grid center
        self.hospital_col = width // 2
        self.hospital_row = height // 2

    def get_cell(self, col, row):
        """
        Safely retrieve a cell.
        """
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise IndexError(f"Cell ({col},{row}) is outside grid")
        return self.cells[col][row]

    def set_population(self, col, row, density):
        """
        Assign population density and update probability.
        """
        cell = self.get_cell(col, row)
        cell.population_density = density
        cell.emergency_probability = density * BASE_EMERGENCY_RATE

    def distance_between_cells(self, col1, row1, col2, row2):
        """
        Euclidean distance between two cells.
        1 grid unit = 10 km.
        """
        grid_distance = np.sqrt((col2 - col1)**2 + (row2 - row1)**2)
        return grid_distance * 10.0

    def update_hospital_distances(self):
        """
        Compute distance from each cell to hospital.
        """
        for col in range(self.width):
            for row in range(self.height):
                dist = self.distance_between_cells(
                    col, row,
                    self.hospital_col, self.hospital_row
                )
                self.cells[col][row].distance_to_hospital_km = dist

    def total_cells(self):
        return self.width * self.height

    def populated_cells(self):
        """
        Returns all cells with non-zero population.
        """
        return [
            self.cells[col][row]
            for col in range(self.width)
            for row in range(self.height)
            if self.cells[col][row].population_density > 0
        ]

    def __repr__(self):
        return (
            f"Grid({self.width}×{self.height}) | "
            f"{self.total_cells()} cells | "
            f"Hospital at ({self.hospital_col},{self.hospital_row})"
        )


# ── Map Builder ──────────────────────────────────────────────────
def build_rural_ontario_map():
    """
    Constructs a realistic rural Ontario simulation map.

    Includes:
    - Town clusters
    - Highway corridor
    - Lakes (water)
    - Forest regions
    - Hilly terrain
    """

    grid = Grid(width=10, height=10)

    # ── Towns ──────────────────────────────────────────────
    towns = [
        (2, 2, 0.75),
        (7, 3, 0.65),
        (4, 7, 0.80),
    ]

    for col, row, density in towns:
        grid.set_population(col, row, density)

        # Surrounding lower-density outskirts
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
            nc, nr = col+dc, row+dr
            if 0 <= nc < 10 and 0 <= nr < 10:
                grid.set_population(nc, nr, density * 0.3)

    # ── Highway ────────────────────────────────────────────
    for col in range(10):
        cell = grid.get_cell(col, 5)
        cell.population_density = 0.15
        cell.emergency_probability = 0.15 * BASE_EMERGENCY_RATE * 2.0

    # ── Lakes ──────────────────────────────────────────────
    water_cells = [(6, 6), (6, 7), (7, 6), (7, 7)]
    for col, row in water_cells:
        cell = grid.get_cell(col, row)
        cell.terrain = TERRAIN_WATER
        cell.has_landing_zone = False
        cell.population_density = 0.0
        cell.emergency_probability = 0.0

    # ── Forest ─────────────────────────────────────────────
    forest_cells = [(0,7),(0,8),(1,8),(0,9),(1,9)]
    for col, row in forest_cells:
        cell = grid.get_cell(col, row)
        cell.terrain = TERRAIN_FOREST
        cell.has_landing_zone = False

    # ── Hilly region ───────────────────────────────────────
    for col in range(8, 10):
        for row in range(0, 3):
            cell = grid.get_cell(col, row)
            cell.terrain = TERRAIN_HILLY

    # Final distance calculation
    grid.update_hospital_distances()

    return grid


# ── Emergency Generation ─────────────────────────────────────────
def generate_emergencies(grid):
    """
    Runs one simulation tick.
    Returns all cells where emergencies occur.
    """
    emergencies = []
    for col in range(grid.width):
        for row in range(grid.height):
            cell = grid.get_cell(col, row)
            if cell.check_emergency():
                emergencies.append(cell)
    return emergencies


# ── Validation Function ──────────────────────────────────────────
def validate_simulation(grid, num_ticks=10000):
    """
    Validates Bernoulli correctness statistically.
    """

    print(f"\nValidating simulation over {num_ticks:,} ticks...")
    print("-" * 50)

    all_passed = True

    for cell in grid.populated_cells():

        # Simulate actual outcomes
        actual = sum(
            1 for _ in range(num_ticks)
            if random.random() < cell.emergency_probability
        )

        # Expected value
        expected = cell.emergency_probability * num_ticks

        # Adjust tolerance for small probabilities
        if expected < 5:
            lower = 0
            upper = expected * 3.0
        else:
            lower = expected * 0.70
            upper = expected * 1.30

        passed = lower <= actual <= upper

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | Cell({cell.col},{cell.row}) | "
            f"expected≈{expected:.1f} | actual={actual}"
        )

    print("-" * 50)
    print("Validation:", "ALL PASSED" if all_passed else "SOME FAILED")

    return all_passed