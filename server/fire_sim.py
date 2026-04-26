"""
Fire and smoke simulation for Pyre.

Cellular automaton model with per-episode variability:
  - Variable number of ignition sources (2–4)
  - Variable spread rate (p_spread)
  - Wind direction (8 directions + CALM) biases spread: 2× downwind, 0.5× upwind
  - Humidity suppresses ignition probability
  - Closed doors reduce spread to ~15% of normal
  - Walls are completely impassable to fire
  - Smoke propagates faster than fire, weakly through doors
  - Burning cells accumulate a timer; after BURNOUT_TICKS they become obstacles
  - Per-cell fuel_map scales ignition probability and intensity gain (office rooms burn faster)
  - Per-cell ventilation_map replaces the global SMOKE_DECAY constant (open areas clear faster)

Wind directions (borrowed from wildfire reference):
  N, NE, E, SE, S, SW, W, NW, CALM
"""

import random
from typing import List, Optional, Tuple

# Cell type constants (mirrors models.py)
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

# Fire intensity thresholds
FIRE_IGNITION = 0.1
FIRE_BURNING = 0.3
FIRE_INTENSITY_GAIN = 0.15
BURNOUT_TICKS = 5

# Door fire reduction factor
DOOR_CLOSED_FIRE_FACTOR = 0.15

# Smoke parameters
SMOKE_SPREAD_RATE = 0.20
SMOKE_DOOR_FACTOR = 0.4
SMOKE_DECAY = 0.02

# Smoke level thresholds
SMOKE_NONE = 0.2
SMOKE_LIGHT = 0.5
SMOKE_MODERATE = 0.8

# Fire intensity at which an exit cell is considered blocked
EXIT_BLOCKED_FIRE_THRESHOLD = 0.5

# Wind direction vectors (dx, dy in grid coords — positive y = south)
WIND_DIRS = {
    "N":    (0, -1),
    "NE":   (1, -1),
    "E":    (1,  0),
    "SE":   (1,  1),
    "S":    (0,  1),
    "SW":  (-1,  1),
    "W":   (-1,  0),
    "NW":  (-1, -1),
    "CALM": (0,  0),
}

_CARDINAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E


def smoke_level_label(density: float) -> str:
    if density < SMOKE_NONE:
        return "none"
    if density < SMOKE_LIGHT:
        return "light"
    if density < SMOKE_MODERATE:
        return "moderate"
    return "heavy"


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _wind_multiplier(dx: int, dy: int, wind_x: int, wind_y: int) -> float:
    """Return spread multiplier based on direction relative to wind.

    Downwind (dot > 0) → 2×, upwind (dot < 0) → 0.5×, crosswind → 1×.
    For diagonal wind components each cardinal direction gets a partial boost.
    """
    if wind_x == 0 and wind_y == 0:
        return 1.0
    dot = dx * wind_x + dy * wind_y
    if dot > 0:
        return 2.0
    elif dot < 0:
        return 0.5
    else:
        return 1.0


class FireSim:
    """Cellular automaton for fire and smoke dynamics.

    All variable parameters are set at construction time so each episode
    gets its own FireSim instance with unique fire behaviour.
    """

    def __init__(
        self,
        w: int,
        h: int,
        rng: random.Random,
        p_spread: float = 0.25,
        wind_dir: str = "CALM",
        humidity: float = 0.25,
        burnout_ticks: int = BURNOUT_TICKS,
        fuel_map: Optional[List[float]] = None,
        ventilation_map: Optional[List[float]] = None,
    ):
        self.w = w
        self.h = h
        self.rng = rng
        self.p_spread = p_spread
        self.wind_dir = wind_dir
        self.humidity = humidity
        self.burnout_ticks = burnout_ticks
        # None → uniform fuel and ventilation (backward-compatible)
        self._fuel_map = fuel_map
        self._ventilation_map = ventilation_map

        wind_vec = WIND_DIRS.get(wind_dir, (0, 0))
        self._wind_x = wind_vec[0]
        self._wind_y = wind_vec[1]
        # Humidity suppresses ignition: effective spread = p_spread × (1 - humidity)
        self._effective_spread = p_spread * max(0.0, 1.0 - humidity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        cell_grid: List[int],
        fire_grid: List[float],
        smoke_grid: List[float],
        burn_timers: List[int],
    ) -> List[Tuple[int, int]]:
        """Advance fire and smoke by one step.

        Mutates fire_grid, smoke_grid, burn_timers in place.
        May mutate cell_grid (burned-out cells become obstacles).

        Returns list of (x, y) cells that burned out this step.
        """
        w, h = self.w, self.h
        burned_out: List[Tuple[int, int]] = []

        # --- Phase 1: Compute fire ignitions ---
        ignite: List[bool] = [False] * (w * h)

        for y in range(h):
            for x in range(w):
                i = _idx(x, y, w)
                ct = cell_grid[i]

                if fire_grid[i] < FIRE_BURNING:
                    continue

                for dx, dy in _CARDINAL:
                    nx, ny = x + dx, y + dy
                    if not _in_bounds(nx, ny, w, h):
                        continue
                    ni = _idx(nx, ny, w)
                    nct = cell_grid[ni]

                    if nct in (WALL, OBSTACLE):
                        continue
                    if fire_grid[ni] > 0:
                        continue

                    # Base spread probability
                    if nct == DOOR_CLOSED:
                        p = self._effective_spread * DOOR_CLOSED_FIRE_FACTOR
                    else:
                        p = self._effective_spread

                    # Wind multiplier
                    p *= _wind_multiplier(dx, dy, self._wind_x, self._wind_y)

                    # Fuel in the target cell scales ignition probability
                    if self._fuel_map is not None:
                        p *= self._fuel_map[ni]

                    p = min(1.0, p)

                    if self.rng.random() < p:
                        ignite[ni] = True

        # --- Phase 2: Apply ignitions and advance existing fire ---
        new_fire = fire_grid[:]
        new_burn_timers = burn_timers[:]

        for y in range(h):
            for x in range(w):
                i = _idx(x, y, w)
                ct = cell_grid[i]

                if ct in (WALL, OBSTACLE):
                    continue

                if fire_grid[i] > 0:
                    intensity_gain = FIRE_INTENSITY_GAIN
                    if self._fuel_map is not None:
                        intensity_gain *= self._fuel_map[i]
                    new_fire[i] = min(1.0, fire_grid[i] + intensity_gain)
                    if fire_grid[i] >= FIRE_BURNING:
                        new_burn_timers[i] = burn_timers[i] + 1
                    if new_burn_timers[i] >= self.burnout_ticks and new_fire[i] >= 1.0:
                        # Doors burn away to open floor (passage stays navigable).
                        # Everything else (furniture, walls that caught fire, etc.) becomes rubble.
                        cell_grid[i] = FLOOR if ct in (DOOR_OPEN, DOOR_CLOSED) else OBSTACLE
                        new_fire[i] = 0.0
                        new_burn_timers[i] = 0
                        burned_out.append((x, y))
                elif ignite[i]:
                    new_fire[i] = FIRE_IGNITION
                    new_burn_timers[i] = 0

        fire_grid[:] = new_fire
        burn_timers[:] = new_burn_timers

        # --- Phase 3: Smoke spread ---
        self._spread_smoke(cell_grid, fire_grid, smoke_grid)

        return burned_out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _spread_smoke(
        self,
        cell_grid: List[int],
        fire_grid: List[float],
        smoke_grid: List[float],
    ) -> None:
        w, h = self.w, self.h
        new_smoke = smoke_grid[:]

        for y in range(h):
            for x in range(w):
                i = _idx(x, y, w)
                ct = cell_grid[i]

                if ct in (WALL, OBSTACLE):
                    continue

                if fire_grid[i] >= FIRE_BURNING:
                    new_smoke[i] = min(1.0, smoke_grid[i] + 0.3)

                for dx, dy in _CARDINAL:
                    nx, ny = x + dx, y + dy
                    if not _in_bounds(nx, ny, w, h):
                        continue
                    ni = _idx(nx, ny, w)
                    nct = cell_grid[ni]

                    if nct in (WALL, OBSTACLE):
                        continue

                    if smoke_grid[i] > smoke_grid[ni]:
                        diff = smoke_grid[i] - smoke_grid[ni]
                        rate = SMOKE_SPREAD_RATE
                        if nct == DOOR_CLOSED:
                            rate *= SMOKE_DOOR_FACTOR
                        transfer = min(diff * rate, diff * 0.5)
                        new_smoke[ni] = min(1.0, new_smoke[ni] + transfer)

                decay = (
                    self._ventilation_map[i]
                    if self._ventilation_map is not None
                    else SMOKE_DECAY
                )
                new_smoke[i] = max(0.0, new_smoke[i] - decay)

        smoke_grid[:] = new_smoke
