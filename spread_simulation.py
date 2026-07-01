# spread_simulation.py
# Physics-informed cellular automaton fire spread engine
# Based on simplified Rothermel fire spread equations
# This is the core patent component — no existing system combines
# ML risk prediction with physics-based spread simulation in real-time

import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from preprocess import BASE

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
GRID_SIZE    = 50          # 50x50 grid cells
CELL_SIZE_M  = 100         # each cell = 100m x 100m
TIMESTEPS    = 6           # simulate 6 hours
_SP_LOCAL = r"C:\Users\natal\OneDrive\Desktop\WildFire AI\Wildfire Pred"
_SP_BASE  = _SP_LOCAL if os.path.exists(_SP_LOCAL) else os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(_SP_BASE, "models", "spread_engine.pkl")

# ── CELL STATES ───────────────────────────────────────────────────────────────
# Each cell in the grid has one of 4 states:
# 0 = Unburned (green)
# 1 = Burning  (red/orange)
# 2 = Burned   (black/grey)
# 3 = Firebreak (road, river — can't burn)
UNBURNED   = 0
BURNING    = 1
BURNED     = 2
FIREBREAK  = 3

class FireSpreadSimulator:
    """
    Physics-informed cellular automaton for wildfire spread.

    The spread probability from cell A to neighboring cell B is:
        P(spread) = P_base × W_wind × W_slope × W_fuel × W_moisture

    Where:
        P_base    = base ignition probability (from ML risk score)
        W_wind    = wind factor (Rothermel: exponential with wind speed)
        W_slope   = slope factor (fire spreads faster uphill)
        W_fuel    = fuel load factor (from NDVI/vegetation dryness)
        W_moisture= moisture suppression (from FFMC)

    This is a simplified but physically grounded version of the
    Rothermel (1972) fire spread model used by the US Forest Service.
    """

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size  = grid_size
        self.reset()

    def reset(self):
        """Initialize empty grid"""
        # Fire state grid
        self.grid       = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Fuel load: random vegetation density 0-1
        # Higher = more fuel = spreads faster
        self.fuel       = np.random.uniform(0.3, 1.0,
                          (self.grid_size, self.grid_size))
        # Terrain slope: random 0-30 degrees
        self.slope      = np.random.uniform(0, 30,
                          (self.grid_size, self.grid_size))
        # Burn intensity: how intensely each cell is burning (0-1)
        self.intensity  = np.zeros((self.grid_size, self.grid_size))
        # Time each cell started burning
        self.burn_time  = np.full((self.grid_size, self.grid_size), -1)
        # History of all timesteps for animation
        self.history    = []
        self.timestep   = 0

    def add_firebreaks(self, positions=None):
        """
        Add firebreaks (roads, rivers) that stop spread.
        positions: list of (row, col) tuples, or None for random breaks
        """
        if positions is None:
            # Add a horizontal and vertical firebreak
            mid = self.grid_size // 2
            self.grid[mid-1, :] = FIREBREAK
            self.grid[:, mid-1] = FIREBREAK
        else:
            for r, c in positions:
                self.grid[r, c] = FIREBREAK

    def ignite(self, row, col):
        """Start fire at a specific cell"""
        if self.grid[row, col] == UNBURNED:
            self.grid[row, col]     = BURNING
            self.intensity[row,col] = 1.0
            self.burn_time[row,col] = self.timestep

    def ignite_region(self, center_row, center_col, radius=2):
        """Start fire in a circular region — more realistic ignition"""
        for r in range(max(0, center_row-radius),
                       min(self.grid_size, center_row+radius+1)):
            for c in range(max(0, center_col-radius),
                           min(self.grid_size, center_col+radius+1)):
                if (r-center_row)**2 + (c-center_col)**2 <= radius**2:
                    if self.grid[r,c] == UNBURNED:
                        self.grid[r,c]     = BURNING
                        self.intensity[r,c]= 1.0
                        self.burn_time[r,c]= self.timestep

    def _wind_factor(self, wind_speed_kmh, wind_dir_deg,
                     from_row, from_col, to_row, to_col):
        """
        Rothermel wind factor:
        Fire spreads exponentially faster in the downwind direction.

        wind_dir_deg: direction wind is blowing FROM (meteorological convention)
        0=N, 90=E, 180=S, 270=W
        """
        # Direction from source cell to target cell
        dr = to_row - from_row   # positive = southward
        dc = to_col - from_col   # positive = eastward

        if dr == 0 and dc == 0:
            return 1.0

        # Angle of spread direction (from north, clockwise)
        spread_angle = np.degrees(np.arctan2(dc, dr)) % 360

        # Wind blows FROM wind_dir, so fire spreads TOWARD wind_dir+180
        downwind_angle = (wind_dir_deg + 180) % 360

        # Alignment between spread direction and downwind direction
        angle_diff = abs(spread_angle - downwind_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        alignment = np.cos(np.radians(angle_diff))  # 1=aligned, -1=opposed

        # Rothermel wind factor: exponential relationship
        # C and B are empirical constants from Rothermel (1972)
        C = 7.47
        B = 0.15
        wind_ms = wind_speed_kmh / 3.6   # convert to m/s

        if alignment > 0:
            # Spreading with the wind
            factor = 1 + C * np.exp(-B * wind_ms) * wind_ms * alignment
        else:
            # Spreading against the wind — slowed down
            factor = max(0.1, 1 + alignment * 0.5)

        return factor

    def _slope_factor(self, from_row, from_col, to_row, to_col):
        """
        Fire spreads faster uphill.
        Slope factor based on Rothermel: phi_s = 5.275 * beta^-0.3 * tan(slope)^2
        Simplified version used here.
        """
        slope_deg  = self.slope[to_row, to_col]
        slope_rad  = np.radians(slope_deg)
        # Determine if spreading uphill (simplified: use slope magnitude)
        # Positive slope = uphill = faster spread
        tan_slope  = np.tan(slope_rad)
        factor     = 1 + 0.533 * (tan_slope ** 2)
        return np.clip(factor, 0.5, 3.0)

    def _fuel_factor(self, row, col):
        """Denser fuel = faster spread"""
        return 0.5 + 1.5 * self.fuel[row, col]

    def _moisture_factor(self, ffmc):
        """
        Higher FFMC = drier fuel = faster spread.
        FFMC range: 0-101. Above 85 is dangerous.
        """
        normalized = (ffmc - 18.7) / (96.2 - 18.7)
        return 0.3 + 1.7 * np.clip(normalized, 0, 1)

    def _get_neighbors(self, row, col):
        """Get 8-connected neighbors (Moore neighborhood)"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    neighbors.append((nr, nc))
        return neighbors

    def step(self, weather):
        """
        Advance simulation by one timestep (1 hour).

        weather: dict with keys:
            'wind_speed' : km/h
            'wind_dir'   : degrees (meteorological, FROM direction)
            'ffmc'       : Fine Fuel Moisture Code
            'temp'       : temperature °C
            'rh'         : relative humidity %
            'risk_score' : ML risk score 0-1 (from XGBoost/fusion)
        """
        wind_speed = weather.get('wind_speed', 10)
        wind_dir   = weather.get('wind_dir',   270)    # default: westerly wind
        ffmc       = weather.get('ffmc',        85)
        risk_score = weather.get('risk_score',  0.5)

        new_grid      = self.grid.copy()
        new_intensity = self.intensity.copy()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] != BURNING:
                    continue

                # This cell is burning — try to spread to neighbors
                neighbors = self._get_neighbors(row, col)

                for nr, nc in neighbors:
                    if self.grid[nr, nc] != UNBURNED:
                        continue  # skip already burned/burning/firebreak

                    # Calculate spread probability using Rothermel factors
                    P_base     = 0.15 + 0.35 * risk_score
                    W_wind     = self._wind_factor(wind_speed, wind_dir,
                                                   row, col, nr, nc)
                    W_slope    = self._slope_factor(row, col, nr, nc)
                    W_fuel     = self._fuel_factor(nr, nc)
                    W_moisture = self._moisture_factor(ffmc)

                    P_spread = P_base * W_wind * W_slope * W_fuel * W_moisture
                    P_spread = np.clip(P_spread, 0, 0.95)

                    # Stochastic ignition
                    if np.random.random() < P_spread:
                        new_grid[nr, nc]      = BURNING
                        new_intensity[nr, nc] = min(1.0,
                            self.intensity[row,col] * W_wind * 0.9)
                        self.burn_time[nr, nc]= self.timestep

                # After burning for 2 timesteps, cell becomes burned-out
                if (self.timestep - self.burn_time[row,col]) >= 2:
                    new_grid[row, col]      = BURNED
                    new_intensity[row, col] = 0

        self.grid      = new_grid
        self.intensity = new_intensity
        self.timestep += 1

        # Save state for animation
        self.history.append({
            'grid'     : self.grid.copy(),
            'intensity': self.intensity.copy(),
            'timestep' : self.timestep,
            'weather'  : weather.copy()
        })

    def run(self, weather_sequence, ignition_point=None):
        """
        Run full simulation.

        weather_sequence: list of weather dicts, one per timestep
        ignition_point  : (row, col) or None for center
        """
        self.reset()

        # Default ignition at center of grid
        if ignition_point is None:
            ignition_point = (self.grid_size//2, self.grid_size//2)

        self.ignite_region(ignition_point[0], ignition_point[1], radius=2)

        print(f"\nRunning fire spread simulation...")
        print(f"Grid: {self.grid_size}x{self.grid_size} cells "
              f"({self.grid_size*CELL_SIZE_M/1000:.1f}km x "
              f"{self.grid_size*CELL_SIZE_M/1000:.1f}km)")

        for t, weather in enumerate(weather_sequence):
            self.step(weather)
            burning = (self.grid == BURNING).sum()
            burned  = (self.grid == BURNED).sum()
            area_ha = (burning + burned) * (CELL_SIZE_M**2) / 10000
            print(f"  Hour {t+1}: Burning={burning} cells | "
                  f"Burned={burned} cells | "
                  f"Total area={area_ha:.1f} ha | "
                  f"Wind={weather['wind_speed']:.0f}km/h "
                  f"@ {weather['wind_dir']:.0f}°")

        return self.history

    def get_evacuation_zones(self, risk_scores_6h):
        """
        Generate confidence-weighted evacuation zones.
        This is the novel component for the patent.

        Zone A (evacuate immediately): cells burning + adjacent + high confidence
        Zone B (prepare to evacuate): cells likely to burn in 2-4h
        Zone C (monitor):             cells at risk in 4-6h

        risk_scores_6h: list of 6 risk scores (one per hour forecast)
        """
        if not self.history:
            return None

        zones = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # 0 = safe, 1 = Zone C, 2 = Zone B, 3 = Zone A

        # Zone A: currently burning or burned
        zones[(self.grid == BURNING) | (self.grid == BURNED)] = 3

        # Zone B: adjacent to burning cells, weighted by hour-2 risk
        confidence_2h = risk_scores_6h[1] if len(risk_scores_6h) > 1 else 0.5
        if confidence_2h > 0.4:
            burning_mask = (self.grid == BURNING).astype(float)
            # Dilate the burning mask by 3 cells
            from scipy.ndimage import binary_dilation
            zone_b_mask = binary_dilation(burning_mask, iterations=3)
            zones[zone_b_mask & (zones == 0)] = 2

        # Zone C: cells that WILL be burning based on simulation trajectory
        confidence_4h = risk_scores_6h[3] if len(risk_scores_6h) > 3 else 0.5
        if confidence_4h > 0.3:
            from scipy.ndimage import binary_dilation
            burning_mask = (self.grid == BURNING).astype(float)
            zone_c_mask  = binary_dilation(burning_mask, iterations=6)
            zones[zone_c_mask & (zones == 0)] = 1

        return zones

    def get_summary_stats(self):
        """Return summary statistics for the dashboard"""
        if not self.history:
            return {}

        final_state = self.history[-1]
        grid        = final_state['grid']
        total_cells = self.grid_size ** 2
        burning     = (grid == BURNING).sum()
        burned      = (grid == BURNED).sum()
        affected    = burning + burned
        area_ha     = affected * (CELL_SIZE_M**2) / 10000

        # Spread rate: cells per hour
        if len(self.history) >= 2:
            prev = self.history[-2]['grid']
            new_cells = ((grid == BURNING) & (prev == UNBURNED)).sum()
            spread_rate_ha = new_cells * (CELL_SIZE_M**2) / 10000
        else:
            spread_rate_ha = 0

        return {
            'total_area_ha'    : round(area_ha, 1),
            'burning_cells'    : int(burning),
            'burned_cells'     : int(burned),
            'affected_pct'     : round(100 * affected / total_cells, 1),
            'spread_rate_ha_h' : round(spread_rate_ha, 1),
            'timesteps'        : len(self.history)
        }


# ── TEST THE SIMULATOR ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("Testing Fire Spread Simulator")
    print("="*55)

    sim = FireSpreadSimulator(grid_size=GRID_SIZE)

    # Simulate an extreme fire weather scenario
    weather_seq = [
        {'wind_speed':25, 'wind_dir':270, 'ffmc':92,
         'temp':38, 'rh':15, 'risk_score':0.85},
        {'wind_speed':28, 'wind_dir':265, 'ffmc':93,
         'temp':40, 'rh':12, 'risk_score':0.90},
        {'wind_speed':30, 'wind_dir':270, 'ffmc':94,
         'temp':41, 'rh':10, 'risk_score':0.92},
        {'wind_speed':28, 'wind_dir':275, 'ffmc':93,
         'temp':39, 'rh':13, 'risk_score':0.88},
        {'wind_speed':25, 'wind_dir':270, 'ffmc':92,
         'temp':37, 'rh':15, 'risk_score':0.85},
        {'wind_speed':22, 'wind_dir':265, 'ffmc':91,
         'temp':35, 'rh':18, 'risk_score':0.80},
    ]

    history = sim.run(weather_seq)
    stats   = sim.get_summary_stats()

    print(f"\n=== Simulation Results ===")
    print(f"Total area burned   : {stats['total_area_ha']} ha")
    print(f"Currently burning   : {stats['burning_cells']} cells")
    print(f"Already burned out  : {stats['burned_cells']} cells")
    print(f"Grid affected       : {stats['affected_pct']}%")
    print(f"Spread rate         : {stats['spread_rate_ha_h']} ha/hour")

    # Test evacuation zones
    risk_scores = [0.85, 0.90, 0.92, 0.88, 0.85, 0.80]
    zones = sim.get_evacuation_zones(risk_scores)
    if zones is not None:
        zone_a = (zones == 3).sum()
        zone_b = (zones == 2).sum()
        zone_c = (zones == 1).sum()
        print(f"\n=== Evacuation Zones ===")
        print(f"Zone A (evacuate now)   : {zone_a} cells "
              f"({zone_a*(CELL_SIZE_M**2)/10000:.1f} ha)")
        print(f"Zone B (prepare)        : {zone_b} cells "
              f"({zone_b*(CELL_SIZE_M**2)/10000:.1f} ha)")
        print(f"Zone C (monitor)        : {zone_c} cells "
              f"({zone_c*(CELL_SIZE_M**2)/10000:.1f} ha)")

    # Save simulator class for use in dashboard
    pickle.dump(sim, open(SAVE_PATH, 'wb'))
    print(f"\nSimulator saved to: {SAVE_PATH}")
    print("\nNext step: update app.py with new tabs")