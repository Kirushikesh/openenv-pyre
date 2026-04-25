/**
 * renderer.js — Canvas2D rendering engine for Pyre visualization
 * Overhauled: natural building materials, vibrant fire, rich atmospheric depth.
 */

// ── Palette ──────────────────────────────────────────────────────────────────

const COLORS = {
  bg:           '#1e242c', // A dark slate/navy instead of pitch black
  // Walls — warm concrete / painted interior
  wallBase:     '#4f443b',
  wallHT:       'rgba(255,248,232,0.25)',   // top catch-light
  wallHL:       'rgba(255,248,232,0.15)',   // left highlight
  wallSB:       'rgba(0,0,0,0.60)',         // bottom shadow
  wallSR:       'rgba(0,0,0,0.45)',         // right shadow
  wallMortar:   'rgba(0,0,0,0.20)',         // mortar / texture lines
  // Floor — worn office tile
  floorBase:    '#261f1a',
  floorVis:     '#302620',
  floorGrout:   'rgba(0,0,0,0.3)',
  // Obstacle — charred ruin
  obstBase:     '#140d0a',
  obstEmber:    'rgba(255,80,0,0.35)',
  // Exit
  exitBase:     '#062210',
  exitGlow:     '#22c55e',
  exitGlow2:    '#4ade80',
  // Door
  doorWood:     '#5c3621',
  doorDark:     '#331c0d',
  doorFrame:    '#291508',
  doorPanel:    'rgba(0,0,0,0.35)',
  doorGlow:     '#7dd3fc',
  // Agent
  agentCore:    '#0ea5e9',
  agentGlow:    '#38bdf8',
  agentBright:  '#e0f2fe',
  // Fog
  fogSeen:      'rgba(30,36,44,0.3)', // increased visibility
  fogUnseen:    'rgba(20,24,30,0.95)', // darker, more contrast
  // Smoke channels
  smokeR: 200, smokeG: 210, smokeB: 220, // Bright, thick white/grey smoke
};

// Fire color ramp: black → deep red → orange → yellow → white
function fireColor(t, alpha=1) {
  t = Math.min(1, Math.max(0, t));
  let r,g,b;
  if      (t < 0.15) { const s=t/0.15;        r=~~(98+88*s);  g=~~(6*s);    b=0; }
  else if (t < 0.40) { const s=(t-0.15)/0.25; r=~~(186+60*s); g=~~(6+60*s); b=0; }
  else if (t < 0.65) { const s=(t-0.40)/0.25; r=~~(246+9*s);  g=~~(66+112*s); b=~~(9*s); }
  else if (t < 0.85) { const s=(t-0.65)/0.20; r=255;          g=~~(178+57*s); b=~~(9+32*s); }
  else               { const s=(t-0.85)/0.15; r=255;          g=~~(235+20*s); b=~~(41+214*s); }
  return `rgba(${r},${g},${b},${alpha})`;
}

function lerpColor(c1, c2, t) {
  return [~~(c1[0]+(c2[0]-c1[0])*t), ~~(c1[1]+(c2[1]-c1[1])*t), ~~(c1[2]+(c2[2]-c1[2])*t)];
}

// ── Ember particle system ─────────────────────────────────────────────────────

class EmberSystem {
  constructor(maxCount=220) {
    this.particles = [];
    this.maxCount = maxCount;
  }

  spawn(x, y, windX, windY, fireIntensity) {
    if (this.particles.length >= this.maxCount) return;
    if (Math.random() > fireIntensity * 0.48) return;
    const speed = 0.4 + Math.random() * 1.0;
    const angle = -Math.PI/2 + (Math.random()-0.5)*1.5;
    this.particles.push({
      x: x + (Math.random()-0.5)*3,
      y: y + (Math.random()-0.5)*3,
      vx: Math.cos(angle)*speed + windX*0.7,
      vy: Math.sin(angle)*speed - 0.22,
      life: 1.0,
      decay: 0.010 + Math.random()*0.015,
      size: 1.3 + Math.random()*2.4,
      type: Math.random() > 0.4 ? 'ember' : 'spark',
    });
  }

  update() {
    for (let i=this.particles.length-1;i>=0;i--) {
      const p=this.particles[i];
      p.x += p.vx; p.y += p.vy;
      p.vy -= 0.012;
      p.vx *= 0.97;
      p.life -= p.decay;
      if (p.life <= 0) this.particles.splice(i,1);
    }
  }

  draw(ctx) {
    for (const p of this.particles) {
      const a = p.life * 0.92;
      const r = p.size * p.life;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI*2);
      if (p.type === 'ember')
        ctx.fillStyle=`rgba(255,${~~(80+100*p.life)},0,${a})`;
      else
        ctx.fillStyle=`rgba(255,${~~(210+45*p.life)},${~~(60*p.life)},${a})`;
      ctx.fill();
    }
  }
}

// ── Smoke drift layer ─────────────────────────────────────────────────────────

class SmokeLayer {
  constructor(w, h) {
    this.w=w; this.h=h;
    this.phases = new Float32Array(w*h).map(()=>Math.random()*Math.PI*2);
  }
  getOffset(x, y, time) {
    const ph = this.phases[y*this.w+x];
    return { dx: Math.sin(time*0.35+ph)*1.8, dy: Math.cos(time*0.28+ph+1.2)*1.8 };
  }
}

// ── Main Renderer ─────────────────────────────────────────────────────────────

class PyreRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.time = 0;
    this.embers = new EmberSystem(220);
    this.agentTrail = [];
    this.prevInterp = null;
    this.smokeLayer = null;
    this.exploreCanvas = null;
    this.exploreCtx = null;
    this._lastExploreSet = null;
    this.cellSize = 0;
    this.windVec = [0,0];
  }

  init(episode) {
    const cs = Math.floor(this.canvas.width / episode.w);
    this.cellSize = cs;
    this.smokeLayer = new SmokeLayer(episode.w, episode.h);
    this.exploreCanvas = document.createElement('canvas');
    this.exploreCanvas.width = this.canvas.width;
    this.exploreCanvas.height = this.canvas.height;
    this.exploreCtx = this.exploreCanvas.getContext('2d');
    this.exploreCtx.fillStyle = 'black';
    this.exploreCtx.fillRect(0,0,this.exploreCanvas.width,this.exploreCanvas.height);
    this.agentTrail = [];
    this._lastExploreSet = new Set();
  }

  frame(episode, lerpT=1) {
    this.time += 0.016;
    const {w,h,cellGrid,fireGrid,smokeGrid,agentX,agentY,visibleCells,exploreSet} = episode;
    const cs = this.cellSize;
    this._w = w;
    const ctx = this.ctx;
    const wv = WIND_DIRS[episode.windDir]||[0,0];
    this.windVec = wv;

    // ── Update explore canvas ─────────────────────────────────────────────
    if (exploreSet && exploreSet !== this._lastExploreSet) {
      this._lastExploreSet = exploreSet;
      for (const key of exploreSet) {
        const [ex,ey] = key.split(',').map(Number);
        this.exploreCtx.clearRect(ex*cs, ey*cs, cs, cs);
      }
    }

    // ── Background ───────────────────────────────────────────────────────
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0,0,this.canvas.width,this.canvas.height);

    // ── Cell base layer ───────────────────────────────────────────────────
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      this._drawCellBase(ctx, x, y, cellGrid, cs);
    }

    // ── Fire ambient — warm glow radiating onto floor (normal blend) ──────
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x;
      if (fireGrid[i] < 0.28) continue;
      this._drawFireAmbient(ctx, x, y, fireGrid[i], cs);
    }

    // ── Fire glow (screen blend for natural layering) ─────────────────────
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x;
      const fire=fireGrid[i];
      if (fire < 0.05) continue;
      this._drawFireCell(ctx, x, y, fire, cs);
      if (fire >= FIRE_BURNING && Math.random()<0.09) {
        this.embers.spawn(x*cs+cs/2, y*cs+cs/2, wv[0], wv[1], fire);
      }
    }
    ctx.restore();

    // ── Smoke overlay ─────────────────────────────────────────────────────
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x;
      const smoke=smokeGrid[i];
      if (smoke < 0.08) continue;
      this._drawSmoke(ctx, x, y, smoke, cs);
    }

    // ── Doors & exits ─────────────────────────────────────────────────────
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const ct=cellGrid[y*w+x];
      if (ct===EXIT) this._drawExit(ctx,x,y,cs,fireGrid[y*w+x]);
      else if (ct===DOOR_OPEN||ct===DOOR_CLOSED) this._drawDoor(ctx,x,y,ct,cs);
    }

    // ── Fog of war ────────────────────────────────────────────────────────
    if (visibleCells && episode.exploreSet) {
      for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
        const key=`${x},${y}`;
        if (visibleCells.has(key)) continue;
        const explored = episode.exploreSet.has(key);
        ctx.fillStyle = explored ? COLORS.fogSeen : COLORS.fogUnseen;
        ctx.fillRect(x*cs, y*cs, cs, cs);
      }
    }

    // ── Agent trail ───────────────────────────────────────────────────────
    this._updateTrail(agentX, agentY, cs);
    this._drawTrail(ctx, cs);

    // ── Agent ─────────────────────────────────────────────────────────────
    this._drawAgent(ctx, agentX, agentY, episode.agentHealth, episode.agentAlive, episode.agentEvacuated, cs);

    // ── Embers (drawn above fog for dramatic effect) ──────────────────────
    this.embers.update();
    this.embers.draw(ctx);

    // ── Tile grid overlay (very subtle) ───────────────────────────────────
    ctx.strokeStyle='rgba(255,255,255,0.04)';
    ctx.lineWidth=1;
    for (let x=0;x<=w;x++) { ctx.beginPath(); ctx.moveTo(x*cs,0); ctx.lineTo(x*cs,h*cs); ctx.stroke(); }
    for (let y=0;y<=h;y++) { ctx.beginPath(); ctx.moveTo(0,y*cs); ctx.lineTo(w*cs,y*cs); ctx.stroke(); }
  }

  // ── Cell base rendering ───────────────────────────────────────────────────

  _drawCellBase(ctx, x, y, cellGrid, cs) {
    const ct = cellGrid[y * (this._w||16) + x];
    const px=x*cs, py=y*cs;

    switch(ct) {
      case WALL: {
        const wallH = 8; // Extrude wall upward for 3D depth
        
        // Front shadow face (gives depth)
        ctx.fillStyle = COLORS.wallSB;
        ctx.fillRect(px, py+cs-wallH, cs, wallH);

        // Right side shadow face if applicable, but for simplicity, just a slight drop shadow 
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillRect(px + cs, py - wallH + 4, 3, cs);

        // Top face (shifted up by wallH)
        ctx.fillStyle = COLORS.wallBase;
        ctx.fillRect(px, py - wallH, cs, cs);

        // Top face catch-light
        ctx.fillStyle = COLORS.wallHT;
        ctx.fillRect(px, py - wallH, cs, 3);
        
        // Left highlight
        ctx.fillStyle = COLORS.wallHL;
        ctx.fillRect(px, py - wallH + 3, 2, cs - 3);

        // Brick texture on the top face
        ctx.fillStyle = COLORS.wallMortar;
        ctx.fillRect(px+3, py - wallH + ~~(cs*0.34), cs-6, 1);
        ctx.fillRect(px+3, py - wallH + ~~(cs*0.67), cs-6, 1);
        ctx.fillRect(px+~~(cs*0.3), py - wallH + 3, 1, ~~(cs*0.34)-3);
        ctx.fillRect(px+~~(cs*0.7), py - wallH + ~~(cs*0.34)+1, 1, ~~(cs*0.33));
        ctx.fillRect(px+~~(cs*0.5), py - wallH + ~~(cs*0.67)+1, 1, ~~(cs*0.33)-3);
        break;
      }

      case OBSTACLE: {
        const obsH = 4; // Charred ruin slight 3D elevation
        
        // Front shadow face
        ctx.fillStyle = '#0a0705';
        ctx.fillRect(px, py+cs-obsH, cs, obsH);

        // Top face
        ctx.fillStyle = COLORS.obstBase;
        ctx.fillRect(px, py - obsH, cs, cs);

        // Diagonal char texture
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.lineWidth = 0.8;
        for (let d=0; d<cs+cs; d+=5) {
          ctx.beginPath();
          ctx.moveTo(px+Math.max(0,d-cs), py - obsH + Math.min(d,cs));
          ctx.lineTo(px+Math.min(d,cs), py - obsH + Math.max(0,d-cs));
          ctx.stroke();
        }

        // Faint ember glow at edges
        ctx.fillStyle = COLORS.obstEmber;
        ctx.fillRect(px,      py - obsH,      cs, 1);
        ctx.fillRect(px,      py - obsH +cs-1, cs, 1);
        ctx.fillRect(px,      py - obsH +1,    1,  cs-2);
        ctx.fillRect(px+cs-1, py - obsH +1,    1,  cs-2);
        break;
      }

      case EXIT: {
        ctx.fillStyle = COLORS.exitBase;
        ctx.fillRect(px, py, cs, cs);
        // Subtle green tint
        ctx.fillStyle = 'rgba(34,197,94,0.07)';
        ctx.fillRect(px+2, py+2, cs-4, cs-4);
        break;
      }

      default: { // FLOOR, DOOR_OPEN, DOOR_CLOSED
        ctx.fillStyle = COLORS.floorVis;
        ctx.fillRect(px, py, cs, cs);

        // Checkered tile effect
        if ((x + y) % 2 === 0) {
          ctx.fillStyle = 'rgba(255,248,230,0.035)';
          ctx.fillRect(px, py, cs, cs);
        }

        // Tile grout lines (top + left edge per cell → creates tile grid)
        ctx.fillStyle = COLORS.floorGrout;
        ctx.fillRect(px,   py, cs, 1);
        ctx.fillRect(px, py+1,  1, cs-1);
        
        // Inner tile details for realism
        ctx.fillStyle = 'rgba(0,0,0,0.06)';
        ctx.fillRect(px+2, py+2, cs-4, 2);
        ctx.fillRect(px+2, py+2, 2, cs-4);
        break;
      }
    }
  }

  // ── Fire rendering ────────────────────────────────────────────────────────

  _drawFireAmbient(ctx, x, y, fire, cs) {
    // Warm orange-red ambient glow radiating onto surrounding cells
    const px=x*cs+cs/2, py=y*cs+cs/2;
    const radius = cs * (2.8 + fire * 2.0); // much larger glow
    const alpha  = fire * 0.65; // much brighter glow
    const gr = ctx.createRadialGradient(px,py,0,px,py,radius);
    gr.addColorStop(0,   `rgba(255,100,10,${alpha})`);
    gr.addColorStop(0.45,`rgba(200,60,0,${alpha*0.5})`);
    gr.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = gr;
    ctx.fillRect(px-radius, py-radius, radius*2, radius*2);
  }

  _drawFireCell(ctx, x, y, fire, cs) {
    const px=x*cs+cs/2, py=y*cs+cs/2;
    const f1 = 0.82 + 0.18*Math.sin(this.time*9.5+x*2.7+y*3.14);
    const eff  = fire * f1;
    
    // Wind offsets for volumetric flame leaning
    const windDx = this.windVec ? this.windVec[0] * cs * 0.35 * eff : 0;
    const windDy = this.windVec ? this.windVec[1] * cs * 0.35 * eff : -cs * 0.15; // default lean up

    // Four layered volumetric gradients
    const layers = [
      {r:cs*0.35, aMax:0.98, ox: windDx*0.2, oy: windDy*0.2 - cs*0.1},   // bright core
      {r:cs*0.65, aMax:0.75, ox: windDx*0.5, oy: windDy*0.5},   // inner flame
      {r:cs*0.95, aMax:0.45, ox: windDx,     oy: windDy},       // flame body
      {r:cs*1.40, aMax:0.18, ox: windDx*1.5, oy: windDy*1.5},   // outer heat glow
    ];
    for (const {r,aMax,ox,oy} of layers) {
      const cx = px + ox;
      const cy = py + oy;
      const gr = ctx.createRadialGradient(cx,cy,0,cx,cy,r);
      gr.addColorStop(0,   fireColor(eff,       aMax));
      gr.addColorStop(0.4, fireColor(eff*0.75,  aMax*0.55));
      gr.addColorStop(0.8, fireColor(eff*0.32,  aMax*0.18));
      gr.addColorStop(1,   fireColor(eff*0.08,  0));
      ctx.fillStyle = gr;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI*2);
      ctx.fill();
    }

    // White-hot core for high intensity fire
    if (eff > 0.48) {
      const cx = px + windDx*0.1;
      const cy = py + windDy*0.1 - cs*0.1;
      const gr2 = ctx.createRadialGradient(cx,cy,0,cx,cy,cs*0.17);
      gr2.addColorStop(0, `rgba(255,255,230,${(eff-0.48)*1.7})`);
      gr2.addColorStop(1, 'rgba(255,210,50,0)');
      ctx.fillStyle = gr2;
      ctx.beginPath();
      ctx.arc(cx, cy, cs*0.17, 0, Math.PI*2);
      ctx.fill();
    }
  }

  _drawSmoke(ctx, x, y, smoke, cs) {
    const px=x*cs+cs/2, py=y*cs+cs/2;
    const off = this.smokeLayer.getOffset(x,y,this.time);
    const alpha = Math.min(0.85, smoke * 0.95);
    const {smokeR:sr, smokeG:sg, smokeB:sb} = COLORS;

    // Draw multiple overlapping soft circles to form a volumetric cloud
    const cloudRadius = cs * 0.85; // overlaps neighboring cells
    
    ctx.save();
    // Shift slightly based on smoke layer offset for animation
    ctx.translate(px + off.dx, py + off.dy);
    
    // Base fluffy cloud layer
    const gr = ctx.createRadialGradient(0, 0, 0, 0, 0, cloudRadius);
    gr.addColorStop(0, `rgba(${sr},${sg},${sb},${alpha})`);
    gr.addColorStop(0.5, `rgba(${sr},${sg},${sb},${alpha*0.75})`);
    gr.addColorStop(1, `rgba(${sr},${sg},${sb},0)`);
    
    ctx.fillStyle = gr;
    ctx.beginPath();
    ctx.arc(0, 0, cloudRadius, 0, Math.PI*2);
    ctx.fill();

    // Secondary offset puff for 3D depth and density
    if (smoke > 0.3) {
      const puffRadius = cs * 0.6;
      const puffOffX = off.dy * 1.5; // orthogonal offset
      const puffOffY = off.dx * 1.5;
      const gr2 = ctx.createRadialGradient(puffOffX, puffOffY, 0, puffOffX, puffOffY, puffRadius);
      gr2.addColorStop(0, `rgba(${sr+15},${sg+15},${sb+20},${alpha*0.5})`);
      gr2.addColorStop(1, `rgba(${sr+15},${sg+15},${sb+20},0)`);
      
      ctx.fillStyle = gr2;
      ctx.beginPath();
      ctx.arc(puffOffX, puffOffY, puffRadius, 0, Math.PI*2);
      ctx.fill();
    }
    
    ctx.restore();
  }

  _drawExit(ctx, x, y, cs, fire) {
    const px=x*cs, py=y*cs;
    const pulse  = 0.62 + 0.38*Math.sin(this.time*2.8 + x + y);
    const blocked = fire >= EXIT_BLOCKED_FIRE_THRESHOLD;
    const col1 = blocked ? '#ef4444' : COLORS.exitGlow;
    const col2 = blocked ? '#dc2626' : COLORS.exitGlow2;

    ctx.save();

    // Outer glow ring
    ctx.shadowBlur  = cs * 1.1 * pulse;
    ctx.shadowColor = col1;
    ctx.strokeStyle = col1;
    ctx.lineWidth   = 2;
    ctx.strokeRect(px+2, py+2, cs-4, cs-4);

    // Interior tint
    ctx.fillStyle = blocked ? 'rgba(220,38,38,0.11)' : 'rgba(34,197,94,0.11)';
    ctx.fillRect(px+3, py+3, cs-6, cs-6);

    // Center symbol
    ctx.shadowBlur = 0;
    const sa = 0.55 + 0.35*pulse;
    ctx.fillStyle = blocked
      ? `rgba(255,90,90,${sa})`
      : `rgba(74,222,128,${sa})`;
    ctx.font = `bold ${~~(cs*0.50)}px sans-serif`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(blocked ? '✕' : '⇥', px+cs/2, py+cs/2+1);

    // Corner bracket accents
    ctx.fillStyle = blocked ? 'rgba(248,113,113,0.75)' : 'rgba(74,222,128,0.75)';
    const ca=5;
    ctx.fillRect(px,      py,      ca, 1); ctx.fillRect(px, py, 1, ca);
    ctx.fillRect(px+cs-ca,py,      ca, 1); ctx.fillRect(px+cs-1,py,1,ca);
    ctx.fillRect(px,      py+cs-1, ca, 1); ctx.fillRect(px, py+cs-ca, 1, ca);
    ctx.fillRect(px+cs-ca,py+cs-1, ca, 1); ctx.fillRect(px+cs-1,py+cs-ca,1,ca);

    ctx.restore();
  }

  _drawDoor(ctx, x, y, ct, cs) {
    const px=x*cs, py=y*cs;

    if (ct === DOOR_OPEN) {
      // Open doorway — dark passage with door pressed to side
      ctx.fillStyle = 'rgba(0,0,0,0.38)';
      ctx.fillRect(px+4, py, cs-6, cs);

      // Door panel pushed to left wall
      ctx.fillStyle = COLORS.doorWood;
      ctx.fillRect(px, py, 4, cs);

      // Door frame along top
      ctx.fillStyle = COLORS.doorFrame;
      ctx.fillRect(px, py, cs, 2);

      // Faint blue passage glow
      ctx.fillStyle = 'rgba(125,211,252,0.10)';
      ctx.fillRect(px+5, py+2, cs-7, cs-4);
    } else {
      // Closed door — wood panel with visible detail
      // Door body
      ctx.fillStyle = COLORS.doorWood;
      ctx.fillRect(px+2, py+1, cs-4, cs-2);

      // Door frame
      ctx.fillStyle = COLORS.doorFrame;
      ctx.fillRect(px, py,        cs, 2);         // top rail
      ctx.fillRect(px, py+cs-2,   cs, 2);         // bottom rail
      ctx.fillRect(px, py+2,      2,  cs-4);      // left stile
      ctx.fillRect(px+cs-2, py+2, 2,  cs-4);      // right stile

      // Panel lines (3 horizontal panels)
      ctx.fillStyle = COLORS.doorPanel;
      ctx.fillRect(px+4, py+~~(cs*0.14), cs-8, 1);
      ctx.fillRect(px+4, py+~~(cs*0.50), cs-8, 1);
      ctx.fillRect(px+4, py+~~(cs*0.82), cs-8, 1);

      // Panel inset shadow
      ctx.fillStyle = 'rgba(0,0,0,0.13)';
      ctx.fillRect(px+4, py+~~(cs*0.14)+2, cs-8, ~~(cs*0.34));
      ctx.fillRect(px+4, py+~~(cs*0.50)+2, cs-8, ~~(cs*0.30));

      // Doorknob — gold circle
      ctx.beginPath();
      ctx.arc(px+cs-8, py+~~(cs/2), 2.5, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(210,168,75,0.88)';
      ctx.fill();
    }
  }

  // ── Agent ─────────────────────────────────────────────────────────────────

  _updateTrail(ax, ay, cs) {
    const px=ax*cs+cs/2, py=ay*cs+cs/2;
    if (!this.agentTrail.length
      || Math.abs(this.agentTrail[0].px-px)>1
      || Math.abs(this.agentTrail[0].py-py)>1) {
      this.agentTrail.unshift({px,py,t:this.time});
    }
    while (this.agentTrail.length > 14) this.agentTrail.pop();
  }

  _drawTrail(ctx, cs) {
    if (this.agentTrail.length < 2) return;
    for (let i=1; i<this.agentTrail.length; i++) {
      const alpha = (1-i/this.agentTrail.length)*0.42;
      const r = Math.max(1.5, (1-i/this.agentTrail.length)*cs*0.22);
      ctx.beginPath();
      ctx.arc(this.agentTrail[i].px, this.agentTrail[i].py, r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(14,165,233,${alpha})`;
      ctx.fill();
    }
  }

  _drawAgent(ctx, ax, ay, health, alive, evacuated, cs) {
    if (!alive && !evacuated) return;
    const px=ax*cs+cs/2, py=ay*cs+cs/2;
    const pulse  = 0.82 + 0.18*Math.sin(this.time*3.6);
    const pulse2 = 0.70 + 0.30*Math.sin(this.time*2.1+1.0);
    const r = cs * 0.38 * pulse; // increased size for better visibility

    const hRatio = Math.max(0, Math.min(1, health/100));
    const agentColor = evacuated  ? '#4ade80'
      : hRatio > 0.6 ? COLORS.agentCore
      : hRatio > 0.3 ? '#f59e0b'
      :                '#ef4444';

    ctx.save();

    // Outer aura pulse (very faint)
    const auraR = cs * 0.74 * pulse2;
    const auraGr = ctx.createRadialGradient(px,py,r*0.9,px,py,auraR);
    auraGr.addColorStop(0,   'rgba(0,0,0,0)');
    auraGr.addColorStop(0.55, evacuated ? 'rgba(74,222,128,0.09)' : 'rgba(56,189,248,0.09)');
    auraGr.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = auraGr;
    ctx.beginPath();
    ctx.arc(px, py, auraR, 0, Math.PI*2);
    ctx.fill();

    // Drop shadow
    ctx.fillStyle = 'rgba(0,0,0,0.32)';
    ctx.beginPath();
    ctx.ellipse(px+2, py+3, r*0.9, r*0.48, 0, 0, Math.PI*2);
    ctx.fill();

    // Glow
    ctx.shadowBlur  = cs * 0.90;
    ctx.shadowColor = evacuated ? '#22c55e' : agentColor;

    // Body — radial gradient for sphere illusion
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI*2);
    const bodyGr = ctx.createRadialGradient(px-r*0.28, py-r*0.28, 0, px, py, r);
    bodyGr.addColorStop(0,    '#ffffff');
    bodyGr.addColorStop(0.32, COLORS.agentBright);
    bodyGr.addColorStop(0.70, agentColor);
    bodyGr.addColorStop(1,    'rgba(0,80,160,0.55)');
    ctx.fillStyle = bodyGr;
    ctx.fill();

    // Outline
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.88)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();

    // Health arc ring
    ctx.save();
    const arcR = r * 1.65;
    // Track
    ctx.beginPath();
    ctx.arc(px, py, arcR, 0, Math.PI*2);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    // Fill arc
    if (hRatio > 0) {
      ctx.beginPath();
      ctx.arc(px, py, arcR, -Math.PI/2, -Math.PI/2 + hRatio*Math.PI*2);
      ctx.strokeStyle = hRatio>0.6?'#4ade80':hRatio>0.3?'#fbbf24':'#f87171'; // More vibrant health colors
      ctx.lineWidth = 3.0; // Slightly thicker
      ctx.lineCap = 'round';
      ctx.stroke();
      
      // Arc glow
      ctx.shadowBlur = 8;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.stroke();
    }
    ctx.restore();
  }

  // ── Wind rose ─────────────────────────────────────────────────────────────

  drawWindRose(canvas2, windDir, speed=1) {
    if (!canvas2) return;
    const ctx=canvas2.getContext('2d');
    const w=canvas2.width, h=canvas2.height, cx=w/2, cy=h/2;
    ctx.clearRect(0,0,w,h);

    const wv=WIND_DIRS[windDir]||[0,0];
    const angle=Math.atan2(wv[1],wv[0]);
    const isCalm=windDir==='CALM';

    // Compass ring
    ctx.beginPath();
    ctx.arc(cx,cy,cx-4,0,Math.PI*2);
    ctx.strokeStyle='rgba(120,140,220,0.28)';
    ctx.lineWidth=1.5;
    ctx.stroke();

    // Cardinal labels
    const labels=[['N',0,-1],['E',1,0],['S',0,1],['W',-1,0]];
    ctx.font=`${w*0.16}px monospace`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    for (const [l,dx,dy] of labels) {
      ctx.fillStyle='rgba(160,172,232,0.55)';
      ctx.fillText(l, cx+dx*(cx-10), cy+dy*(cy-10));
    }

    if (!isCalm) {
      const arrowLen=cx*0.60;
      const ax2=cx+Math.cos(angle)*arrowLen, ay2=cy+Math.sin(angle)*arrowLen;
      ctx.beginPath();
      ctx.moveTo(cx,cy);
      ctx.lineTo(ax2,ay2);
      ctx.strokeStyle='rgba(251,191,36,0.95)';
      ctx.lineWidth=2.5;
      ctx.stroke();

      const headLen=8, ha=0.40;
      ctx.beginPath();
      ctx.moveTo(ax2,ay2);
      ctx.lineTo(ax2-headLen*Math.cos(angle-ha), ay2-headLen*Math.sin(angle-ha));
      ctx.lineTo(ax2-headLen*Math.cos(angle+ha), ay2-headLen*Math.sin(angle+ha));
      ctx.closePath();
      ctx.fillStyle='rgba(251,191,36,0.95)';
      ctx.fill();

      ctx.font=`bold ${w*0.18}px monospace`;
      ctx.fillStyle='rgba(255,210,80,0.95)';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(windDir, cx, cy);
    } else {
      ctx.font=`${w*0.18}px monospace`;
      ctx.fillStyle='rgba(160,182,232,0.75)';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('CALM', cx, cy);
    }
  }
}
