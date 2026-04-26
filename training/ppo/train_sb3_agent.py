import sys
import os
import types

# Windows AppControl can block matplotlib's compiled C extensions.
# Stub the minimal surface that stable_baselines3.common.logger imports
# at module level so SB3 loads cleanly even without a working matplotlib.
def _stub_matplotlib():
    if "matplotlib" in sys.modules:
        return
    _mpl = types.ModuleType("matplotlib")
    _mpl.figure = types.ModuleType("matplotlib.figure")
    _mpl.figure.Figure = object
    _mpl.use = lambda *a, **kw: None
    _mpl.__version__ = "0.0.0"
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.figure"] = _mpl.figure
    for sub in ("matplotlib.pyplot", "matplotlib.ticker", "matplotlib.patches",
                "matplotlib.gridspec", "matplotlib.colors", "matplotlib.cm",
                "matplotlib.backend_bases", "matplotlib.backends",
                "matplotlib.backends.backend_agg"):
        m = types.ModuleType(sub)
        sys.modules[sub] = m

_stub_matplotlib()

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyre_env.models import PyreAction, PyreObservation
from pyre_env.server.pyre_env_environment import PyreEnvironment
import torch as th
sys.path.append(os.getcwd())

class PyreGymEnv(gym.Env):
    """Gymnasium wrapper for PyreEnvironment."""
    
    def __init__(self, difficulty="easy", max_steps=150, observation_mode="visible"):
        super().__init__()
        self.env = PyreEnvironment(max_steps=max_steps)
        self.difficulty = difficulty
        self.observation_mode = observation_mode
        
        # Action space: 
        # 0-3: Move (N, S, W, E)
        # 4-7: Look (N, S, W, E)
        # 8: Wait
        # 9-24: Open Door 1-16
        # 25-40: Close Door 1-16
        self.action_space = spaces.Discrete(41)
        
        # Observation space: Multi-input
        # 1. Grid: 24x24x7 (Floor, Wall, Door_Open, Door_Closed, Exit, Obstacle, Fire, Smoke)
        # 2. Global: [health, oxygen, step_progress, fire_spread, humidity, agent_x, agent_y, nearest_exit_dist, is_coughing]
        # 3. Heat Sensor: 3x3
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(7, 24, 24), dtype=np.float32),
            "global": spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32),
            "heat": spaces.Box(low=0, high=1, shape=(1, 3, 3), dtype=np.float32)
        })

    def _get_obs(self, pyre_obs: PyreObservation):
        map_state = pyre_obs.map_state
        w, h = map_state.grid_w, map_state.grid_h
        
        # Build 7-channel grid
        # Channels: 0:Wall, 1:Door_Open, 2:Door_Closed, 3:Exit, 4:Obstacle, 5:Fire, 6:Smoke
        # (Floor is implicit as all zeros in other channels)
        grid = np.zeros((7, 24, 24), dtype=np.float32)
        
        visible = {(x, y) for x, y in map_state.visible_cells}
        for y in range(h):
            for x in range(w):
                if self.observation_mode == "visible" and (x, y) not in visible and (x, y) != (map_state.agent_x, map_state.agent_y):
                    continue
                
                i = y * w + x
                ct = map_state.cell_grid[i]
                if ct == 1: grid[0, y, x] = 1.0 # Wall
                elif ct == 2: grid[1, y, x] = 1.0 # Door Open
                elif ct == 3: grid[2, y, x] = 1.0 # Door Closed
                elif ct == 4: grid[3, y, x] = 1.0 # Exit
                elif ct == 5: grid[4, y, x] = 1.0 # Obstacle
                
                grid[5, y, x] = float(map_state.fire_grid[i])
                grid[6, y, x] = float(map_state.smoke_grid[i])
        
        # Global features
        metadata = pyre_obs.metadata or {}
        nearest_exit = float(metadata.get("nearest_exit_distance", 48) or 48.0) / 48.0
        # smoke_level → is_coughing proxy (moderate/heavy smoke = coughing)
        smoke = getattr(pyre_obs, "smoke_level", "none") or "none"
        is_coughing = 1.0 if smoke in ("moderate", "heavy") else 0.0

        global_feats = np.array([
            float(pyre_obs.agent_health) / 100.0,
            float(pyre_obs.agent_health) / 100.0,   # oxygen_level proxy
            float(map_state.step_count) / float(map_state.max_steps),
            float(map_state.fire_spread_rate),
            float(map_state.humidity),
            float(map_state.agent_x) / 24.0,
            float(map_state.agent_y) / 24.0,
            nearest_exit,
            is_coughing,
        ], dtype=np.float32)

        # Heat sensor — derive 3×3 fire neighbourhood around agent from the fire grid
        ax, ay = map_state.agent_x, map_state.agent_y
        gw, gh = map_state.grid_w, map_state.grid_h
        heat_vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < gw and 0 <= ny < gh:
                    heat_vals.append(float(map_state.fire_grid[ny * gw + nx]))
                else:
                    heat_vals.append(0.0)
        heat = np.array(heat_vals, dtype=np.float32).reshape(1, 3, 3)
        
        return {
            "grid": grid,
            "global": global_feats,
            "heat": heat
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        difficulty = options.get("difficulty", self.difficulty) if options else self.difficulty
        pyre_obs = self.env.reset(seed=seed, difficulty=difficulty)
        return self._get_obs(pyre_obs), {}

    def step(self, action_idx):
        # Map Discrete action to PyreAction
        if action_idx < 4:
            dirs = ["north", "south", "west", "east"]
            action = PyreAction(action="move", direction=dirs[action_idx])
        elif action_idx < 8:
            dirs = ["north", "south", "west", "east"]
            action = PyreAction(action="look", direction=dirs[action_idx - 4])
        elif action_idx == 8:
            action = PyreAction(action="wait")
        elif action_idx < 9 + 16:
            action = PyreAction(action="door", target_id=f"door_{action_idx - 8}", door_state="open")
        else:
            action = PyreAction(action="door", target_id=f"door_{action_idx - 24}", door_state="close")
            
        pyre_obs = self.env.step(action)
        
        obs = self._get_obs(pyre_obs)
        reward = pyre_obs.reward
        terminated = pyre_obs.done
        truncated = False # Step limit handled by env.done
        
        return obs, reward, terminated, truncated, {"pyre_obs": pyre_obs}

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500, help="Total episodes to train across all levels")
    parser.add_argument("--difficulty", type=str, default="curriculum", help="easy, medium, hard, random, or curriculum")
    parser.add_argument("--output", type=str, default="artifacts/ppo_pyre_multilevel")
    args = parser.parse_args()
    
    from gymnasium.wrappers import RecordEpisodeStatistics
    
    # Custom wrapper to handle difficulty changes
    class MultiLevelWrapper(gym.Wrapper):
        def __init__(self, env, mode="curriculum"):
            super().__init__(env)
            self.mode = mode
            self.current_difficulty = "easy"
            self.step_count = 0
            self.total_steps = 0
            
        def reset(self, **kwargs):
            if self.mode == "random":
                self.current_difficulty = np.random.choice(["easy", "medium", "hard"])
            elif self.mode == "curriculum":
                if self.total_steps < 0.33 * total_training_steps:
                    self.current_difficulty = "easy"
                elif self.total_steps < 0.66 * total_training_steps:
                    self.current_difficulty = "medium"
                else:
                    self.current_difficulty = "hard"
            else:
                self.current_difficulty = self.mode
            
            # Extract options from kwargs if present, or create new
            options = kwargs.get("options")
            if options is None:
                options = {}
            options["difficulty"] = self.current_difficulty
            kwargs["options"] = options
                
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            self.total_steps += 1
            info["difficulty"] = self.current_difficulty
            return obs, reward, term, trunc, info

    total_training_steps = args.episodes * 60
    
    env = PyreGymEnv(difficulty="easy") # Base difficulty
    env = MultiLevelWrapper(env, mode=args.difficulty)
    env = RecordEpisodeStatistics(env)
    
    # Custom CNN policy for the grid
    # Increased network capacity for multiple levels
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[256, 128], qf=[256, 128])
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=2e-4, # Slightly lower LR for stability across levels
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02, # Higher entropy to encourage exploration in procedural maps
    )
    
    print(f"Starting multi-level training (mode: {args.difficulty})...")
    
    # Add a simple callback to log episode rewards to a CSV
    from stable_baselines3.common.callbacks import BaseCallback
    import csv
    from pathlib import Path
    
    class CSVLogCallback(BaseCallback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename
            self.results = []
        def _on_step(self):
            # Check every step for finished episodes
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.results.append({
                        "step": self.num_timesteps,
                        "reward": info["episode"]["r"],
                        "length": info["episode"]["l"]
                    })
            return True
        def _on_rollout_end(self):
            # Save every rollout
            if self.results:
                with open(self.filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["step", "reward", "length"])
                    writer.writeheader()
                    writer.writerows(self.results)
            return True

    csv_path = args.output + ".csv"
    callback = CSVLogCallback(csv_path)

    # CNN MultiInputPolicy needs far more steps than a flat MLP to warm up.
    # episodes * 50 ≈ 15k steps (too few). Use episodes * 500 for meaningful learning.
    model.learn(total_timesteps=args.episodes * 500, callback=callback)
    
    model.save(args.output)
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {csv_path}")

    # Generate a quick SVG graph if we have results
    if callback.results:
        try:
            from examples.train_rl_agent import save_training_graph
            # Mocking the row format expected by the baseline plotter
            rows = [{"episode": i, "reward": r["reward"], "evacuated": 0} for i, r in enumerate(callback.results)]
            save_training_graph(Path(args.output + ".svg"), rows, [])
            print(f"Graph saved to {args.output}.svg")
        except Exception as e:
            print(f"Could not generate SVG automatically: {e}")
            print("CSV is available at " + csv_path)
