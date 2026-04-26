import React, { useRef, useEffect } from 'react';
import type { Observation } from '../types';

interface Map2DProps {
  observation: Observation | null;
  agentMoveFlash: number;
}

/* ── Cell type constants ── */
const WALL        = 1;
const DOOR_OPEN   = 2;
const DOOR_CLOSED = 3;
const EXIT        = 4;
const OBSTACLE    = 5;

/**
 * Map raw fire intensity (often 0.1–0.35 while smoldering / spreading) to a
 * stronger draw weight so corridor and door-adjacent cells read clearly under fog.
 */
function fireDrawWeight(raw: number): number {
  return Math.min(1, 0.12 + 0.88 * Math.min(1, raw));
}

/* ── Wind direction vectors ── */
const WIND_DIRS: Record<string, [number, number]> = {
  N: [0, -1], S: [0, 1], E: [1, 0], W: [-1, 0],
  NW: [-0.7, -0.7], NE: [0.7, -0.7], SW: [-0.7, 0.7], SE: [0.7, 0.7],
  CALM: [0, 0],
};

/* ── Agent appearance per health tier ── */
const AGENT_THEMES = {
  healthy:  { body: '#3b82f6', dark: '#1d4ed8', arm: '#2563eb', ring: '#fbbf24', ringGlow: 'rgba(251,191,36,0.5)' },
  moderate: { body: '#f97316', dark: '#c2410c', arm: '#ea580c', ring: '#fb923c', ringGlow: 'rgba(251,146,60,0.5)' },
  low:      { body: '#dc2626', dark: '#991b1b', arm: '#b91c1c', ring: '#f87171', ringGlow: 'rgba(248,113,113,0.5)' },
  critical: { body: '#7c3aed', dark: '#5b21b6', arm: '#6d28d9', ring: '#c4b5fd', ringGlow: 'rgba(196,181,253,0.5)' },
};

/* ── Ember particle ── */
class Ember {
  x: number; y: number; vx: number; vy: number;
  life: number; decay: number; size: number;
  type: 'ember' | 'spark';

  constructor(x: number, y: number, windX: number) {
    const speed = 0.4 + Math.random() * 1.0;
    const angle = -Math.PI / 2 + (Math.random() - 0.5) * 1.6;
    this.x = x + (Math.random() - 0.5) * 3;
    this.y = y + (Math.random() - 0.5) * 3;
    this.vx = Math.cos(angle) * speed + windX * 0.7;
    this.vy = Math.sin(angle) * speed - 0.22;
    this.life = 1.0;
    this.decay = 0.012 + Math.random() * 0.015;
    this.size = 1.2 + Math.random() * 2.2;
    this.type = Math.random() > 0.4 ? 'ember' : 'spark';
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.vy -= 0.012;
    this.vx *= 0.97;
    this.life -= this.decay;
  }
}


/* ── Minecraft pixel-art character ── */
function drawMinecraftAgent(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, cs: number,
  theme: typeof AGENT_THEMES.healthy
) {
  const u = cs / 18;
  const left = cx - 5 * u;
  const top  = cy - 8.5 * u;

  const px = (rx: number, ry: number, rw: number, rh: number, color: string) => {
    ctx.fillStyle = color;
    ctx.fillRect(left + rx * u, top + ry * u, rw * u, rh * u);
  };

  /* Helmet */
  px(2, 0, 6, 1, '#5c4a3d');
  /* Head */
  px(2, 1, 6, 5, '#f5d5a0');
  /* Face features */
  px(3, 3, 1, 1, '#3d2b1a'); /* left eye */
  px(6, 3, 1, 1, '#3d2b1a'); /* right eye */
  px(4, 5, 2, 1, '#c8937a'); /* mouth */
  /* Hair accent */
  px(2, 1, 6, 1, '#7a5c3e');

  /* Body */
  px(3, 6, 4, 4, theme.body);
  px(3, 6, 4, 1, theme.dark);

  /* Arms */
  px(1, 6, 2, 4, theme.arm);
  px(7, 6, 2, 4, theme.arm);

  /* Legs */
  px(3, 10, 2, 4, '#1e40af');
  px(5, 10, 2, 4, '#1e3a8a');

  /* Boots */
  px(3, 14, 2, 2, '#3a2e26');
  px(5, 14, 2, 2, '#2e2420');
}

/* ── Main canvas component ── */
const Map2D: React.FC<Map2DProps> = ({ observation, agentMoveFlash }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const embersRef = useRef<Ember[]>([]);
  const trailRef  = useRef<{ x: number; y: number; t: number }[]>([]);
  const timeRef   = useRef(0);
  const rafRef    = useRef(0);

  const CS = 40;

  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas || !observation) { rafRef.current = requestAnimationFrame(animate); return; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    timeRef.current += 0.016;
    const t = timeRef.current;

    const { map_state, agent_health, wind_dir } = observation;
    const { grid_w: W, grid_h: H, cell_grid, fire_grid, smoke_grid, agent_x, agent_y } = map_state;
    const cs = CS;
    const wv = WIND_DIRS[wind_dir] ?? [0, 0];
    const idx = (x: number, y: number) => y * W + x;
    const visible = new Set(map_state.visible_cells.map(([vx, vy]) => `${vx},${vy}`));

    /* ── Canvas bg ── */
    ctx.fillStyle = '#c8b890';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    /* ── Base layer ── */
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const ct = cell_grid[idx(x, y)];
        const px = x * cs, py = y * cs;

        switch (ct) {
          case WALL: {
            /* Animated heat tint — walls near fire glow ember-red */
            const nearFire = fire_grid[idx(x, y)] > 0.05
              ? fire_grid[idx(x, y)]
              : (
                  (x > 0     && fire_grid[idx(x-1, y)] > 0.05 ? fire_grid[idx(x-1, y)] : 0) +
                  (x < W-1   && fire_grid[idx(x+1, y)] > 0.05 ? fire_grid[idx(x+1, y)] : 0) +
                  (y > 0     && fire_grid[idx(x, y-1)] > 0.05 ? fire_grid[idx(x, y-1)] : 0) +
                  (y < H-1   && fire_grid[idx(x, y+1)] > 0.05 ? fire_grid[idx(x, y+1)] : 0)
                ) * 0.28;

            const heatShift = Math.min(1, nearFire * 2.2);
            const wallFlicker = 0.88 + 0.12 * Math.sin(t * 7.3 + x * 2.1 + y * 3.7);

            /* Base stone colour, heat-shifted toward deep orange-red */
            const br = Math.round(94  + heatShift * 100 * wallFlicker);
            const bg = Math.round(88  - heatShift * 52);
            const bb = Math.round(80  - heatShift * 70);
            ctx.fillStyle = `rgb(${br},${bg},${bb})`;
            ctx.fillRect(px, py, cs, cs);

            /* Brick rows — two horizontal bands */
            const brickH = cs / 2;
            for (let row = 0; row < 2; row++) {
              const by = py + row * brickH;
              /* mortar gap between rows */
              ctx.fillStyle = `rgba(0,0,0,${0.28 + heatShift * 0.12})`;
              ctx.fillRect(px, by + brickH - 1, cs, 1);
              /* vertical mortar — staggered per row */
              const mortarX = px + ((x + row) % 2 === 0 ? cs / 2 : cs / 4);
              ctx.fillStyle = `rgba(0,0,0,${0.22 + heatShift * 0.10})`;
              ctx.fillRect(mortarX, by, 1, brickH - 1);
            }

            /* Top-left highlight bevel */
            ctx.fillStyle = `rgba(255,${200 - Math.round(heatShift * 80)},${160 - Math.round(heatShift * 140)},${0.32 + heatShift * 0.15})`;
            ctx.fillRect(px, py, cs, 2);
            ctx.fillRect(px, py + 2, 2, cs - 2);

            /* Bottom-right shadow bevel */
            ctx.fillStyle = `rgba(0,0,0,${0.50 + heatShift * 0.20})`;
            ctx.fillRect(px, py + cs - 2, cs, 2);
            ctx.fillRect(px + cs - 2, py, 2, cs - 2);

            /* Heat glow overlay on wall face */
            if (heatShift > 0.05) {
              const glowA = heatShift * 0.35 * wallFlicker;
              ctx.fillStyle = `rgba(255,${Math.round(80 - heatShift * 60)},0,${glowA})`;
              ctx.fillRect(px + 2, py + 2, cs - 4, cs - 4);

              /* Hot crack lines radiating from fire side */
              ctx.strokeStyle = `rgba(255,${Math.round(160 - heatShift * 120)},0,${heatShift * 0.6 * wallFlicker})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(px + cs * 0.3, py + cs * 0.2);
              ctx.lineTo(px + cs * 0.5, py + cs * 0.55);
              ctx.lineTo(px + cs * 0.7, py + cs * 0.4);
              ctx.stroke();
              if (heatShift > 0.4) {
                ctx.beginPath();
                ctx.moveTo(px + cs * 0.2, py + cs * 0.7);
                ctx.lineTo(px + cs * 0.45, py + cs * 0.85);
                ctx.stroke();
              }
            }
            break;
          }
          case OBSTACLE: {
            /* Charred debris — dark with ember glow */
            const obsNearFire = (
              (x > 0   && fire_grid[idx(x-1, y)] > 0.05 ? fire_grid[idx(x-1, y)] : 0) +
              (x < W-1 && fire_grid[idx(x+1, y)] > 0.05 ? fire_grid[idx(x+1, y)] : 0) +
              (y > 0   && fire_grid[idx(x, y-1)] > 0.05 ? fire_grid[idx(x, y-1)] : 0) +
              (y < H-1 && fire_grid[idx(x, y+1)] > 0.05 ? fire_grid[idx(x, y+1)] : 0)
            ) * 0.4 + fire_grid[idx(x, y)] * 0.8;
            const obsHeat = Math.min(1, obsNearFire);
            const obsFlicker = 0.82 + 0.18 * Math.sin(t * 9.1 + x * 1.9 + y * 2.5);

            ctx.fillStyle = '#2a2520';
            ctx.fillRect(px, py, cs, cs);

            /* Rubble texture patches */
            ctx.fillStyle = 'rgba(60,50,40,0.7)';
            ctx.fillRect(px + 3, py + 3, cs * 0.4, cs * 0.35);
            ctx.fillRect(px + cs * 0.55, py + cs * 0.5, cs * 0.35, cs * 0.4);
            ctx.fillStyle = 'rgba(80,65,50,0.5)';
            ctx.fillRect(px + cs * 0.25, py + cs * 0.6, cs * 0.45, cs * 0.3);

            /* Ember glow if near fire */
            if (obsHeat > 0.05) {
              const eg = Math.round(40 + obsHeat * 90 * obsFlicker);
              ctx.fillStyle = `rgba(255,${eg},0,${obsHeat * 0.55 * obsFlicker})`;
              ctx.fillRect(px + 2, py + 2, cs - 4, cs - 4);

              /* Glowing edge cracks */
              ctx.strokeStyle = `rgba(255,${Math.round(120 * obsHeat * obsFlicker)},0,${obsHeat * 0.8})`;
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(px + cs * 0.1, py + cs * 0.5);
              ctx.lineTo(px + cs * 0.4, py + cs * 0.3);
              ctx.lineTo(px + cs * 0.6, py + cs * 0.7);
              ctx.lineTo(px + cs * 0.9, py + cs * 0.4);
              ctx.stroke();
            }

            /* Orange danger frame */
            ctx.strokeStyle = `rgba(255,${Math.round(80 + obsHeat * 60)},0,${0.55 + obsHeat * 0.35})`;
            ctx.lineWidth = 2;
            ctx.strokeRect(px + 1, py + 1, cs - 2, cs - 2);

            /* Corner bolts */
            ctx.fillStyle = `rgba(255,${Math.round(100 + obsHeat * 80)},0,${0.7 + obsHeat * 0.3})`;
            [[4,4],[cs-6,4],[4,cs-6],[cs-6,cs-6]].forEach(([bx, by]) => {
              ctx.beginPath(); ctx.arc(px+bx, py+by, 2, 0, Math.PI*2); ctx.fill();
            });
            break;
          }
          default: {
            /* Checkerboard floor with warm heat tint near fire */
            const floorFire = Math.min(1,
              fire_grid[idx(x, y)] * 1.5 +
              (x > 0   ? fire_grid[idx(x-1, y)] : 0) * 0.3 +
              (x < W-1 ? fire_grid[idx(x+1, y)] : 0) * 0.3 +
              (y > 0   ? fire_grid[idx(x, y-1)] : 0) * 0.3 +
              (y < H-1 ? fire_grid[idx(x, y+1)] : 0) * 0.3
            );
            const base = (x + y) % 2 === 0;
            const fr = Math.round((base ? 232 : 208) + floorFire * 23);
            const fg = Math.round((base ? 216 : 190) - floorFire * 40);
            const fb = Math.round((base ? 184 : 152) - floorFire * 80);
            ctx.fillStyle = `rgb(${fr},${fg},${fb})`;
            ctx.fillRect(px, py, cs, cs);
            /* tile bevel */
            ctx.fillStyle = 'rgba(255,255,255,0.20)';
            ctx.fillRect(px, py, cs, 2);
            ctx.fillRect(px, py + 2, 2, cs - 2);
            ctx.fillStyle = 'rgba(0,0,0,0.18)';
            ctx.fillRect(px, py + cs - 2, cs, 2);
            ctx.fillRect(px + cs - 2, py, 2, cs - 2);
          }
        }
      }
    }

    /* ── Fire ambient: multiply scorches the floor tiles (under fog) ── */
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const fire = fire_grid[idx(x, y)];
        if (fire < 0.035) continue;
        const px = x * cs + cs / 2, py = y * cs + cs / 2;
        const fw = fireDrawWeight(fire);
        const radius = cs * (1.2 + fire * 1.8);
        const a = Math.min(0.85, fw * 0.9);
        const gr = ctx.createRadialGradient(px, py, 0, px, py, radius);
        gr.addColorStop(0,   `rgba(255,80,0,${a})`);
        gr.addColorStop(0.4, `rgba(220,40,0,${a * 0.5})`);
        gr.addColorStop(1,   'rgba(0,0,0,0)');
        ctx.fillStyle = gr;
        ctx.fillRect(px - radius, py - radius, radius * 2, radius * 2);
      }
    }
    ctx.restore();

    /* ── Smoke (dark on light bg, under fog) ── */
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const smoke = smoke_grid[idx(x, y)];
        if (smoke < 0.1) continue;
        const px = x * cs + cs / 2, py = y * cs + cs / 2;
        const offX = Math.sin(t * 0.5 + x) * 2;
        const offY = Math.cos(t * 0.4 + y) * 2;
        const alpha = Math.min(0.68, smoke * 0.8);
        const gr = ctx.createRadialGradient(px + offX, py + offY, 0, px + offX, py + offY, cs * 0.82);
        gr.addColorStop(0, `rgba(72,82,96,${alpha})`);
        gr.addColorStop(1, 'rgba(72,82,96,0)');
        ctx.fillStyle = gr;
        ctx.beginPath(); ctx.arc(px + offX, py + offY, cs * 0.82, 0, Math.PI * 2); ctx.fill();
      }
    }

    /* ── Exits & Doors ── */
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const ct = cell_grid[idx(x, y)];
        const px = x * cs, py = y * cs;
        const pulse = 0.7 + 0.3 * Math.sin(t * 3);

        if (ct === EXIT) {
          ctx.fillStyle = '#e6f4ec';
          ctx.fillRect(px + 2, py + 2, cs - 4, cs - 4);
          ctx.strokeStyle = `rgba(22,163,74,${0.7 + 0.3 * pulse})`;
          ctx.lineWidth = 2 * pulse;
          ctx.strokeRect(px + 5, py + 5, cs - 10, cs - 10);
          /* EXIT symbol */
          ctx.fillStyle = `rgba(22,163,74,${0.85 + 0.15 * pulse})`;
          ctx.font = `bold ${cs * 0.26}px var(--mono, monospace)`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('EXIT', px + cs / 2, py + cs / 2);
        } else if (ct === DOOR_CLOSED) {
          ctx.fillStyle = '#7c5c3c';
          ctx.fillRect(px + 4, py + 2, cs - 8, cs - 4);
          ctx.fillStyle = '#4a3020';
          ctx.fillRect(px + 2, py, cs - 4, 2);
          ctx.fillRect(px + 2, py + cs - 2, cs - 4, 2);
          /* handle */
          ctx.fillStyle = '#f0b030';
          ctx.beginPath();
          ctx.arc(px + cs - 10, py + cs / 2, 2.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (ct === DOOR_OPEN) {
          ctx.fillStyle = '#4a3020';
          ctx.fillRect(px + 2, py, 4, cs);
          ctx.fillRect(px + cs - 6, py, 4, cs);
        }
      }
    }

    /* ── Fog of War (dim only — fire still punches through above) ── */
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const key = `${x},${y}`;
        if (!visible.has(key)) {
          const f = fire_grid[idx(x, y)];
          /* Let smolder / spread show through — uniform 0.55 grey hid weak corridor fire */
          const fogA = f > 0.18 ? 0.08 : f > 0.08 ? 0.18 : f > 0.025 ? 0.32 : 0.55;
          ctx.fillStyle = `rgba(140,134,126,${fogA})`;
          ctx.fillRect(x * cs, y * cs, cs, cs);
        }
      }
    }

    /* ── Fire volumetric: drawn ABOVE fog — always visible ── */
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const fire = fire_grid[idx(x, y)];
        if (fire < 0.025) continue;
        const px = x * cs + cs / 2, py = y * cs + cs / 2;
        const flicker = 0.80 + 0.20 * Math.sin(t * 11.0 + x * 3.1 + y * 2.7);
        const eff = Math.min(1, fireDrawWeight(fire) * flicker);
        const isVisible = visible.has(`${x},${y}`);

        const windDx = wv[0] * cs * 0.25 * eff;
        const windDy = wv[1] * cs * 0.25 * eff - cs * 0.06;

        /* Wide warning beacon glow for fire in fog — always shown */
        if (!isVisible) {
          const beaconPulse = 0.6 + 0.4 * Math.sin(t * 6.0 + x * 1.7 + y * 2.3);
          const beaconR = cs * (1.55 + beaconPulse * 0.75);
          const b = Math.min(1, eff * 1.35);
          const beaconGr = ctx.createRadialGradient(px, py, 0, px, py, beaconR);
          beaconGr.addColorStop(0,   `rgba(255,100,0,${b * beaconPulse * 0.90})`);
          beaconGr.addColorStop(0.3, `rgba(255,60,0,${b * beaconPulse * 0.55})`);
          beaconGr.addColorStop(1,   'rgba(220,30,0,0)');
          ctx.fillStyle = beaconGr;
          ctx.beginPath(); ctx.arc(px, py, beaconR, 0, Math.PI * 2); ctx.fill();
        }

        /* Outer dark-red base */
        {
          const r = cs * 0.70 * (0.7 + eff * 0.3);
          const cx2 = px + windDx * 0.5, cy2 = py + windDy * 0.5;
          const gr = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
          gr.addColorStop(0,   `rgba(200,20,0,${eff * 0.65})`);
          gr.addColorStop(0.55,`rgba(170,10,0,${eff * 0.35})`);
          gr.addColorStop(1,   'rgba(80,0,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
        }

        /* Mid vivid-orange body */
        {
          const r = cs * 0.46 * (0.8 + eff * 0.2);
          const cx2 = px + windDx * 0.35, cy2 = py + windDy * 0.35 - cs * 0.04 * eff;
          const gr = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
          gr.addColorStop(0,   `rgba(255,110,0,${eff * 0.92})`);
          gr.addColorStop(0.45,`rgba(255,60,0,${eff * 0.62})`);
          gr.addColorStop(1,   'rgba(220,20,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
        }

        /* Inner bright-yellow core */
        {
          const r = cs * 0.28 * eff;
          const cx2 = px + windDx * 0.15, cy2 = py + windDy * 0.15 - cs * 0.10 * eff;
          const gr = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
          gr.addColorStop(0,   `rgba(255,235,90,${eff * 0.97})`);
          gr.addColorStop(0.35,`rgba(255,175,25,${eff * 0.78})`);
          gr.addColorStop(1,   'rgba(255,80,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
        }

        /* White-hot tip (only for intense fire) */
        if (eff > 0.55) {
          const r = cs * 0.14 * eff;
          const cx2 = px + windDx * 0.1, cy2 = py + windDy * 0.1 - cs * 0.18 * eff;
          const gr = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
          gr.addColorStop(0, `rgba(255,255,230,${eff * 0.95})`);
          gr.addColorStop(1, 'rgba(255,220,60,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
        }

        /* Wind-carried plume tip */
        if (fire > 0.35) {
          const r  = cs * 0.30 * eff;
          const cx2 = px + windDx, cy2 = py + windDy - cs * 0.22 * eff;
          const gr = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, r);
          gr.addColorStop(0, `rgba(255,165,10,${eff * 0.68})`);
          gr.addColorStop(1, 'rgba(255,60,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
        }

        /* Outer visible bloom ring (makes fire pop even in fog) */
        {
          const bloomR = cs * (0.85 + eff * 0.35);
          const bloomGr = ctx.createRadialGradient(px, py, cs * 0.2, px, py, bloomR);
          bloomGr.addColorStop(0, 'rgba(255,120,0,0)');
          bloomGr.addColorStop(0.6, `rgba(255,80,0,${eff * 0.22})`);
          bloomGr.addColorStop(1,   'rgba(200,30,0,0)');
          ctx.fillStyle = bloomGr;
          ctx.beginPath(); ctx.arc(px, py, bloomR, 0, Math.PI * 2); ctx.fill();
        }

        if (fire > 0.45 && Math.random() < 0.09 && embersRef.current.length < 120) {
          embersRef.current.push(new Ember(px, py, wv[0]));
        }
      }
    }

    /* ── Vision lantern glow around agent ── */
    const apx = agent_x * cs + cs / 2, apy = agent_y * cs + cs / 2;
    const lanternR = cs * 3.5;
    const lanternGr = ctx.createRadialGradient(apx, apy, 0, apx, apy, lanternR);
    lanternGr.addColorStop(0,   'rgba(255,240,190,0.18)');
    lanternGr.addColorStop(0.5, 'rgba(255,220,140,0.08)');
    lanternGr.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = lanternGr;
    ctx.fillRect(apx - lanternR, apy - lanternR, lanternR * 2, lanternR * 2);

    /* ── Agent trail ── */
    const now = timeRef.current;
    if (
      trailRef.current.length === 0 ||
      Math.abs(trailRef.current[0].x - apx) > 1 ||
      Math.abs(trailRef.current[0].y - apy) > 1
    ) {
      trailRef.current.unshift({ x: apx, y: apy, t: now });
    }
    if (trailRef.current.length > 20) trailRef.current.pop();

    trailRef.current.forEach((p, i) => {
      const alpha = (1 - i / trailRef.current.length) * 0.70;
      ctx.fillStyle = `rgba(2,132,199,${alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, cs * 0.12 * (1 - i / 22), 0, Math.PI * 2);
      ctx.fill();
    });

    /* ── Agent rendering ── */
    const theme =
      agent_health >= 60 ? AGENT_THEMES.healthy  :
      agent_health >= 30 ? AGENT_THEMES.moderate :
      agent_health >  0  ? AGENT_THEMES.low       :
                           AGENT_THEMES.critical;

    const pulse = 0.85 + 0.15 * Math.sin(t * 4);
    const ringR = cs * 0.48;

    /* pulsing gold aura */
    const auraR = ringR * (1.5 + 0.2 * pulse);
    const auraGr = ctx.createRadialGradient(apx, apy, ringR * 0.7, apx, apy, auraR);
    auraGr.addColorStop(0, `rgba(251,191,36,${0.28 * pulse})`);
    auraGr.addColorStop(1, 'rgba(251,191,36,0)');
    ctx.fillStyle = auraGr;
    ctx.beginPath(); ctx.arc(apx, apy, auraR, 0, Math.PI * 2); ctx.fill();

    /* ground shadow */
    ctx.save();
    ctx.globalAlpha = 0.22;
    const shadowGr = ctx.createRadialGradient(apx, apy + cs * 0.32, 0, apx, apy + cs * 0.32, cs * 0.38);
    shadowGr.addColorStop(0, 'rgba(0,0,0,0.6)');
    shadowGr.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = shadowGr;
    ctx.beginPath(); ctx.ellipse(apx, apy + cs * 0.32, cs * 0.38, cs * 0.14, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    /* Minecraft character */
    drawMinecraftAgent(ctx, apx, apy, cs, theme);

    /* health arc ring — gold ring + colored fill */
    const hRatio = Math.max(0, Math.min(1, agent_health / 100));
    /* ring track */
    ctx.beginPath();
    ctx.arc(apx, apy, ringR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();
    /* gold base ring */
    ctx.beginPath();
    ctx.arc(apx, apy, ringR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(251,191,36,0.25)';
    ctx.lineWidth = 3.5;
    ctx.stroke();
    /* health fill */
    ctx.beginPath();
    ctx.arc(apx, apy, ringR, -Math.PI / 2, -Math.PI / 2 + hRatio * Math.PI * 2);
    ctx.strokeStyle = theme.ring;
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.stroke();
    /* ring glow */
    ctx.beginPath();
    ctx.arc(apx, apy, ringR, -Math.PI / 2, -Math.PI / 2 + hRatio * Math.PI * 2);
    ctx.strokeStyle = theme.ringGlow;
    ctx.lineWidth = 6;
    ctx.stroke();

    /* move flash */
    if (agentMoveFlash > 0) {
      const fa = agentMoveFlash / 18;
      ctx.strokeStyle = `rgba(255,255,255,${fa * 0.8})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(apx, apy, ringR * (1.8 + (1 - fa) * 0.6), 0, Math.PI * 2);
      ctx.stroke();
    }

    /* ── Embers ── */
    for (let i = embersRef.current.length - 1; i >= 0; i--) {
      const e = embersRef.current[i];
      e.update();
      if (e.life <= 0) { embersRef.current.splice(i, 1); continue; }
      ctx.fillStyle = `rgba(255,${Math.floor(80 + 175 * e.life)},0,${e.life})`;
      ctx.beginPath(); ctx.arc(e.x, e.y, e.size * e.life, 0, Math.PI * 2); ctx.fill();
    }

    rafRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [observation, agentMoveFlash]);

  const W = observation?.map_state.grid_w ?? 16;
  const H = observation?.map_state.grid_h ?? 16;

  return (
    <canvas
      ref={canvasRef}
      id="map-canvas"
      width={W * CS}
      height={H * CS}
    />
  );
};

export default Map2D;
