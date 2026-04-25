"""
Fire and smoke simulation for Pyre.

Cellular automaton model:
- Fire spreads to adjacent floor cells with probability p_spread
- Closed doors reduce spread to 15% of normal rate
- Walls are completely impassable to fire
- Smoke propagates 2× faster than fire, weakly through doors
- Burning cells accumulate a timer; after BURNOUT_TICKS they become obstacles (cell type 5)
"""

import random
from typing import Dict, List, Optional, Tuple

# Cell type constants (mirrors models.py)
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

# Fire intensity thresholds
FIRE_IGNITION = 0.1       # initial intensity when a cell catches
FIRE_BURNING = 0.3        # threshold for active spreading
FIRE_INTENSITY_GAIN = 0.15  # intensity added per step for burning cells
BURNOUT_TICKS = 5         # steps until a fully burning cell burns out

# Spread probabilities
P_SPREAD_BASE = 0.25
P_SPREAD_THROUGH_CLOSED_DOOR = P_SPREAD_BASE * 0.15  # doors drastically slow fire

# Smoke parameters
SMOKE_SPREAD_RATE = 0.20  # smoke added to neighbor per step (vs fire's implicit 0.10)
SMOKE_DOOR_FACTOR = 0.4   # smoke passes through closed doors at this fraction
SMOKE_DECAY = 0.02        # slight natural dissipation per step

# Smoke level thresholds
SMOKE_NONE = 0.2
SMOKE_LIGHT = 0.5
SMOKE_MODERATE = 0.8
# above SMOKE_MODERATE = heavy


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


_CARDINAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E


class FireSim:
    """Cellular automaton for fire and smoke dynamics."""

    def __init__(
        self,
        w: int,
        h: int,
        rng: random.Random,
        p_spread: float = P_SPREAD_BASE,
        burnout_ticks: int = BURNOUT_TICKS,
    ):
        self.w = w
        self.h = h
        self.rng = rng
        self.p_spread = p_spread
        self.burnout_ticks = burnout_ticks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        cell_grid: List[int],   # modified in-place for burnouts
        fire_grid: List[float],  # modified in-place
        smoke_grid: List[float], # modified in-place
        burn_timers: List[int],  # modified in-place
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

                # Only burning cells spread fire
                if fire_grid[i] < FIRE_BURNING:
                    continue

                for dx, dy in _CARDINAL:
                    nx, ny = x + dx, y + dy
                    if not _in_bounds(nx, ny, w, h):
                        continue
                    ni = _idx(nx, ny, w)
                    nct = cell_grid[ni]

                    # Wall and obstacle cells don't ignite
                    if nct in (WALL, OBSTACLE):
                        continue

                    # Already on fire
                    if fire_grid[ni] > 0:
                        continue

                    # Exits can have fire (they're still physical cells)
                    if nct == DOOR_CLOSED:
                        p = self.p_spread * 0.15
                    else:
                        p = self.p_spread

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
                    # Existing fire grows
                    new_fire[i] = min(1.0, fire_grid[i] + FIRE_INTENSITY_GAIN)
                    if fire_grid[i] >= FIRE_BURNING:
                        new_burn_timers[i] = burn_timers[i] + 1
                    # Burn-out: become obstacle
                    if new_burn_timers[i] >= self.burnout_ticks and new_fire[i] >= 1.0:
                        cell_grid[i] = OBSTACLE
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

                # Fire cells generate smoke
                if fire_grid[i] >= FIRE_BURNING:
                    new_smoke[i] = min(1.0, smoke_grid[i] + 0.3)

                # Spread to neighbors
                for dx, dy in _CARDINAL:
                    nx, ny = x + dx, y + dy
                    if not _in_bounds(nx, ny, w, h):
                        continue
                    ni = _idx(nx, ny, w)
                    nct = cell_grid[ni]

                    if nct in (WALL, OBSTACLE):
                        continue

                    # Smoke moves from high to low concentration
                    if smoke_grid[i] > smoke_grid[ni]:
                        diff = smoke_grid[i] - smoke_grid[ni]
                        rate = SMOKE_SPREAD_RATE
                        if nct == DOOR_CLOSED:
                            rate *= SMOKE_DOOR_FACTOR
                        transfer = min(diff * rate, diff * 0.5)
                        new_smoke[ni] = min(1.0, new_smoke[ni] + transfer)

                # Natural decay (ventilation)
                new_smoke[i] = max(0.0, new_smoke[i] - SMOKE_DECAY)

        smoke_grid[:] = new_smoke
