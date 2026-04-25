# Pyre — Frontend Visualization

A cinematic real-time visualization for the **Pyre Crisis Navigation Environment** — a reinforcement learning environment where an LLM agent navigates a burning building.

## Quick start

```bash
# Open directly in a browser — no build step needed
open frontend/index.html
```

The app runs entirely in-browser. **Demo mode** simulates the fire physics in JavaScript (no server required). **Live mode** connects to the deployed environment.

---

## Demo mode vs Live mode

| | Demo | Live |
|---|---|---|
| Server needed | ✗ | ✓ |
| Fire physics | JS port (exact match) | Python server |
| Full reward rubric | Simplified | Complete |
| Toggle | Default | Click "Live" in topbar |

**Live server:** `https://krooz-pyre-env.hf.space`

---

## Controls

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `→` | Single step |
| `R` | New episode |
| `1`–`5` | Speed ½× / 1× / 2× / 4× / 8× |

Bottom bar: difficulty selector, seed input, speed control, reset.

---

## Recording episodes (Python)

```bash
pip install requests  # only stdlib used, no install needed

python bridge/recorder.py \
  --url https://krooz-pyre-env.hf.space/web \
  --episodes 10 \
  --difficulty medium \
  --out episodes/
```

Episodes are saved as JSON files to `episodes/`. Each file contains full frame-by-frame grid data (cell, fire, smoke grids + agent position + visible cells).

---

## File structure

```
frontend/
├── index.html       Main app — open this
└── js/
    ├── sim.js       JS port of pyre_env fire simulation + floor plans
    ├── renderer.js  Canvas2D rendering (fire particles, fog-of-war, agent trail)
    └── app.js       App controller, charts, HUD, live/demo modes

bridge/
└── recorder.py      Record live episodes to JSON for replay
```

---

## Architecture notes

**Rendering:** HTML5 Canvas 2D — sufficient at 60fps for 16×16 grids; additive blending (`globalCompositeOperation: lighter`) for fire glow; ember particle pool (200 max); fog-of-war via per-cell alpha overlay.

**Demo agent:** BFS toward nearest unblocked exit, 15% random exploration, avoids fire cells > 0.4 intensity.

**Live bridge:** Polls `/web/scene` every 800ms; applies grid state to the same rendering pipeline.

---

## Demo script (30-second stage walkthrough)

1. **Open** `frontend/index.html` — fire simulation starts automatically at 1×
2. **Point out** the dark floor plan canvas with glowing fire cells, fog-of-war, and cyan agent dot
3. **Slow to ½×** to show per-step fire propagation and smoke spread
4. **Speed to 4×** — show agent navigating toward exits (green glow), closing doors (blue bars) to slow fire
5. **Highlight** the side panel: cumulative reward curve dipping on smoke exposure, fire cell count climbing, action histogram
6. **Describe partial observability** — the dark unexplored cells vs. visible corridor
7. **Reset (R)** with a different seed to show episode variety
8. If server is available: click **Live** — "Connected" chip turns green, real Python environment takes over
