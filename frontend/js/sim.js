/**
 * sim.js — JavaScript port of pyre_env fire simulation
 * Mirrors server/fire_sim.py and server/floor_plan.py exactly.
 */

// Cell types
const FLOOR = 0, WALL = 1, DOOR_OPEN = 2, DOOR_CLOSED = 3, EXIT = 4, OBSTACLE = 5;

const FIRE_IGNITION = 0.1;
const FIRE_BURNING = 0.3;
const FIRE_INTENSITY_GAIN = 0.15;
const BURNOUT_TICKS = 5;
const DOOR_CLOSED_FIRE_FACTOR = 0.15;
const SMOKE_SPREAD_RATE = 0.20;
const SMOKE_DOOR_FACTOR = 0.4;
const SMOKE_DECAY = 0.02;
const EXIT_BLOCKED_FIRE_THRESHOLD = 0.5;

const WIND_DIRS = {
  N:[0,-1], NE:[1,-1], E:[1,0], SE:[1,1],
  S:[0,1], SW:[-1,1], W:[-1,0], NW:[-1,-1], CALM:[0,0]
};
const CARDINAL = [[0,-1],[0,1],[-1,0],[1,0]];

function windMult(dx, dy, wx, wy) {
  if (wx===0&&wy===0) return 1.0;
  const dot = dx*wx + dy*wy;
  return dot>0 ? 2.0 : dot<0 ? 0.5 : 1.0;
}

// Seeded PRNG (mulberry32)
function makePRNG(seed) {
  let s = seed >>> 0;
  return function() {
    s += 0x6D2B79F5;
    let t = s;
    t = Math.imul(t ^ t>>>15, t|1);
    t ^= t + Math.imul(t^t>>>7, t|61);
    return ((t^t>>>14)>>>0) / 4294967296;
  };
}

class FireSim {
  constructor(w, h, rng, pSpread=0.25, windDir='CALM', humidity=0.25, fuelMap=null, ventMap=null) {
    this.w=w; this.h=h; this.rng=rng;
    const wv = WIND_DIRS[windDir]||[0,0];
    this.wx=wv[0]; this.wy=wv[1];
    this.effectiveSpread = pSpread * Math.max(0, 1-humidity);
    this.fuelMap=fuelMap; this.ventMap=ventMap;
  }

  step(cellGrid, fireGrid, smokeGrid, burnTimers) {
    const {w, h} = this;
    const burned = [];
    const ignite = new Uint8Array(w*h);

    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x;
      if (fireGrid[i] < FIRE_BURNING) continue;
      for (const [dx,dy] of CARDINAL) {
        const nx=x+dx, ny=y+dy;
        if (nx<0||nx>=w||ny<0||ny>=h) continue;
        const ni=ny*w+nx, nct=cellGrid[ni];
        if (nct===WALL||nct===OBSTACLE||fireGrid[ni]>0) continue;
        let p = nct===DOOR_CLOSED ? this.effectiveSpread*DOOR_CLOSED_FIRE_FACTOR : this.effectiveSpread;
        p *= windMult(dx,dy,this.wx,this.wy);
        if (this.fuelMap) p *= this.fuelMap[ni];
        if (this.rng() < Math.min(1,p)) ignite[ni]=1;
      }
    }

    const newFire = fireGrid.slice();
    const newTimers = burnTimers.slice();
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x, ct=cellGrid[i];
      if (ct===WALL||ct===OBSTACLE) continue;
      if (fireGrid[i]>0) {
        let gain = FIRE_INTENSITY_GAIN;
        if (this.fuelMap) gain*=this.fuelMap[i];
        newFire[i] = Math.min(1, fireGrid[i]+gain);
        if (fireGrid[i]>=FIRE_BURNING) newTimers[i]++;
        if (newTimers[i]>=BURNOUT_TICKS && newFire[i]>=1.0) {
          cellGrid[i]=OBSTACLE; newFire[i]=0; newTimers[i]=0;
          burned.push([x,y]);
        }
      } else if (ignite[i]) {
        newFire[i]=FIRE_IGNITION; newTimers[i]=0;
      }
    }
    for (let i=0;i<fireGrid.length;i++){fireGrid[i]=newFire[i];burnTimers[i]=newTimers[i];}
    this._spreadSmoke(cellGrid, fireGrid, smokeGrid);
    return burned;
  }

  _spreadSmoke(cellGrid, fireGrid, smokeGrid) {
    const {w,h}=this;
    const ns=smokeGrid.slice();
    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const i=y*w+x, ct=cellGrid[i];
      if (ct===WALL||ct===OBSTACLE) continue;
      if (fireGrid[i]>=FIRE_BURNING) ns[i]=Math.min(1, smokeGrid[i]+0.3);
      for (const [dx,dy] of CARDINAL) {
        const nx=x+dx, ny=y+dy;
        if (nx<0||nx>=w||ny<0||ny>=h) continue;
        const ni=ny*w+nx, nct=cellGrid[ni];
        if (nct===WALL||nct===OBSTACLE) continue;
        if (smokeGrid[i]>smokeGrid[ni]) {
          const diff=smokeGrid[i]-smokeGrid[ni];
          let rate=SMOKE_SPREAD_RATE;
          if (nct===DOOR_CLOSED) rate*=SMOKE_DOOR_FACTOR;
          ns[ni]=Math.min(1, ns[ni]+Math.min(diff*rate, diff*0.5));
        }
      }
      const decay=this.ventMap?this.ventMap[i]:SMOKE_DECAY;
      ns[i]=Math.max(0, ns[i]-decay);
    }
    for (let i=0;i<smokeGrid.length;i++) smokeGrid[i]=ns[i];
  }
}

// ─── Floor plan templates (mirrors floor_plan.py) ───────────────────────────

function makeSmallOffice() {
  const W=16, H=16;
  const rows=[
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  ];
  const grid=rows.flat();
  return {
    name:'small_office', w:W, h:H, grid,
    exits:[[0,6],[15,8]],
    doors:[[2,4],[6,4],[10,4],[14,4],[2,10],[6,10],[10,10],[14,10]],
    agentSpawns: (() => {
      const s=[];
      for(let y=5;y<=9;y++) for(let x=1;x<=14;x++) if(grid[y*W+x]===0) s.push([x,y]);
      return s;
    })(),
    zoneMap: buildZoneMap_SO(grid,W,H)
  };
}

function buildZoneMap_SO(grid,W,H) {
  const zm={};
  for(let y=0;y<H;y++) for(let x=0;x<W;x++) {
    const ct=grid[y*W+x];
    if(ct===0) zm[`${x},${y}`]=y<=4?'north_offices':y>=5&&y<=9?'main_corridor':'south_offices';
    else if(ct===4) zm[`${x},${y}`]='exit';
  }
  return zm;
}

function makeOpenPlan() {
  const W=16, H=16;
  const rows=[
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,5,5,0,0,0,0,0,5,5,0,0,0,1],
    [1,0,0,5,5,0,0,0,0,0,5,5,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,5,5,0,0,0,0,0,5,5,0,0,0,1],
    [1,0,0,5,5,0,0,0,0,0,5,5,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],
  ];
  const grid=rows.flat();
  const agentSpawns=[];
  for(let y=1;y<H-1;y++) for(let x=1;x<W-1;x++) if(grid[y*W+x]===FLOOR) agentSpawns.push([x,y]);
  return {name:'open_plan',w:W,h:H,grid,exits:[[0,1],[15,15]],doors:[],agentSpawns};
}

function makeTCorridor() {
  const W=16, H=16;
  const rows=[
    [1,1,1,1,1,1,1,4,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
    [1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1],
    [1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1],
    [1,1,2,1,1,2,1,0,1,2,1,1,1,1,2,1],
    [1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1],
    [1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  ];
  const grid=rows.flat();
  const agentSpawns=[];
  for(let x=1;x<W-1;x++) if(grid[7*W+x]===FLOOR) agentSpawns.push([x,7]);
  for(let y=4;y<8;y++) if(grid[y*W+7]===FLOOR) agentSpawns.push([7,y]);
  return {name:'t_corridor',w:W,h:H,grid,exits:[[7,0],[0,7],[15,7]],doors:[[2,10],[5,10],[9,10],[14,10]],agentSpawns};
}

const TEMPLATES = [makeSmallOffice(), makeOpenPlan(), makeTCorridor()];

// ─── BFS exit distance ──────────────────────────────────────────────────────

function bfsExitDist(ax, ay, exits, cellGrid, w, h) {
  const blocked = exits.filter(([ex,ey])=>cellGrid[ey*w+ex]===OBSTACLE);
  const targets = exits.filter(([ex,ey])=>cellGrid[ey*w+ex]!==OBSTACLE);
  if (!targets.length) return 999;
  const visited = new Set([`${ax},${ay}`]);
  const queue = [[ax,ay,0]];
  while (queue.length) {
    const [x,y,d]=queue.shift();
    for (const [dx,dy] of CARDINAL) {
      const nx=x+dx, ny=y+dy;
      if (nx<0||nx>=w||ny<0||ny>=h) continue;
      const key=`${nx},${ny}`;
      if (visited.has(key)) continue;
      const ct=cellGrid[ny*w+nx];
      if (ct===WALL||ct===OBSTACLE||ct===DOOR_CLOSED) continue;
      if (targets.some(([ex,ey])=>ex===nx&&ey===ny)) return d+1;
      visited.add(key);
      queue.push([nx,ny,d+1]);
    }
  }
  return 999;
}

// ─── Fog of war (BFS, visibility radius shrinks under smoke) ────────────────

function computeVisible(ax, ay, cellGrid, smokeGrid, w, h) {
  const smokeDensity = smokeGrid[ay*w+ax];
  const radius = smokeDensity>0.5 ? 2 : smokeDensity>0.2 ? 3 : 5;
  const visible = new Set([`${ax},${ay}`]);
  const queue = [[ax,ay,0]];
  const visited = new Set([`${ax},${ay}`]);
  while (queue.length) {
    const [x,y,d]=queue.shift();
    if (d>=radius) continue;
    for (const [dx,dy] of CARDINAL) {
      const nx=x+dx, ny=y+dy;
      if (nx<0||nx>=w||ny<0||ny>=h) continue;
      const key=`${nx},${ny}`;
      if (visited.has(key)) continue;
      visited.add(key);
      const ct=cellGrid[ny*w+nx];
      if (ct===WALL) continue;
      visible.add(key);
      queue.push([nx,ny,d+1]);
    }
  }
  return visible;
}

// ─── Simple BFS demo agent ──────────────────────────────────────────────────

class DemoAgent {
  constructor() { this.lastDir = null; this.stuckCount = 0; }

  chooseAction(state) {
    const { agentX:ax, agentY:ay, cellGrid, fireGrid, smokeGrid, exits, w, h } = state;
    const dirs = ['north','south','east','west'];
    const deltas = {north:[0,-1],south:[0,1],east:[1,0],west:[-1,0]};

    // Find best direction toward nearest exit
    let bestDir = null, bestDist = Infinity;
    for (const dir of dirs) {
      const [dx,dy]=deltas[dir];
      const nx=ax+dx, ny=ay+dy;
      if (nx<0||nx>=w||ny<0||ny>=h) continue;
      const ct=cellGrid[ny*w+nx];
      if (ct===WALL||ct===OBSTACLE) continue;
      if (ct===DOOR_CLOSED) {
        // Try to open door if fire is not adjacent
        return {action:'door', target_id:this._findDoor(state,nx,ny), door_state:'open'};
      }
      const fire=fireGrid[ny*w+nx];
      if (fire>0.4) continue; // avoid fire
      const d = bfsExitDist(nx,ny,exits,cellGrid,w,h);
      if (d<bestDist) { bestDist=d; bestDir=dir; }
    }

    if (!bestDir || Math.random()<0.15) {
      // Random valid move
      const shuffled=[...dirs].sort(()=>Math.random()-0.5);
      for (const dir of shuffled) {
        const [dx,dy]=deltas[dir];
        const nx=ax+dx, ny=ay+dy;
        if (nx<0||nx>=w||ny<0||ny>=h) continue;
        const ct=cellGrid[ny*w+nx];
        if (ct!==WALL&&ct!==OBSTACLE&&ct!==DOOR_CLOSED&&fireGrid[ny*w+nx]<0.4) {
          return {action:'move',direction:dir};
        }
      }
    }
    return bestDir ? {action:'move',direction:bestDir} : {action:'wait'};
  }

  _findDoor(state, dx, dy) {
    for (const [id,[x,y]] of Object.entries(state.doorRegistry||{})) {
      if (x===dx&&y===dy) return id;
    }
    return `door_1`;
  }
}

// ─── Episode state machine ───────────────────────────────────────────────────

class PyreEpisode {
  constructor(templateIdx=0, seed=42, difficulty='medium') {
    const presets = {
      easy:   {nSources:[1,1],  pSpread:[0.10,0.20], humidity:[0.30,0.50], winds:['CALM'],          maxSteps:200},
      medium: {nSources:[2,4],  pSpread:[0.15,0.40], humidity:[0.10,0.45], winds:Object.keys(WIND_DIRS), maxSteps:150},
      hard:   {nSources:[3,5],  pSpread:[0.30,0.55], humidity:[0.05,0.20], winds:Object.keys(WIND_DIRS).filter(d=>d!=='CALM'), maxSteps:100},
    };
    const p = presets[difficulty]||presets.medium;
    const rng = makePRNG(seed);
    const tpl = TEMPLATES[templateIdx % TEMPLATES.length];

    // Deep-copy grid
    this.cellGrid = tpl.grid.slice();
    this.w = tpl.w; this.h = tpl.h;
    this.exits = tpl.exits.map(e=>[...e]);
    this.doors = (tpl.doors||[]).map(d=>[...d]);
    this.doorRegistry = {};
    for (let j=0;j<this.doors.length;j++) {
      this.doorRegistry[`door_${j+1}`]=this.doors[j];
      if (rng()<0.3) this.cellGrid[this.doors[j][1]*this.w+this.doors[j][0]]=DOOR_CLOSED;
    }

    // Fire params
    const [ns0,ns1]=p.nSources, [sp0,sp1]=p.pSpread, [hm0,hm1]=p.humidity;
    const nSrc = Math.round(ns0 + rng()*(ns1-ns0));
    const pSpread = sp0 + rng()*(sp1-sp0);
    const humidity = hm0 + rng()*(hm1-hm0);
    const windDir = p.winds[Math.floor(rng()*p.winds.length)];

    this.pSpread = pSpread;
    this.humidity = humidity;
    this.windDir = windDir;
    this.maxSteps = p.maxSteps;
    this.templateName = tpl.name;
    this.difficulty = difficulty;
    this.seed = seed;

    // Fire grids
    this.fireGrid = new Float32Array(this.w*this.h);
    this.smokeGrid = new Float32Array(this.w*this.h);
    this.burnTimers = new Int32Array(this.w*this.h);

    // Place fire sources
    const floorCells = [];
    for (let y=0;y<this.h;y++) for (let x=0;x<this.w;x++) {
      if (this.cellGrid[y*this.w+x]===FLOOR) {
        const farFromExit = this.exits.every(([ex,ey])=>Math.abs(x-ex)+Math.abs(y-ey)>=5);
        if (farFromExit) floorCells.push([x,y]);
      }
    }
    floorCells.sort(()=>rng()-0.5);

    // Agent spawn
    const spawn = tpl.agentSpawns[Math.floor(rng()*tpl.agentSpawns.length)];
    this.agentX = spawn[0]; this.agentY = spawn[1];
    this.agentHealth = 100.0;
    this.agentAlive = true;
    this.agentEvacuated = false;
    this.stepCount = 0;

    // Place fires far from agent
    const fireSources = floorCells.filter(([fx,fy])=>Math.abs(fx-this.agentX)+Math.abs(fy-this.agentY)>=4).slice(0,nSrc);
    for (const [fx,fy] of fireSources) this.fireGrid[fy*this.w+fx]=FIRE_IGNITION;

    // Fire sim
    this.sim = new FireSim(this.w, this.h, makePRNG(seed+1), pSpread, windDir, humidity);

    // Tracking
    this.visibleCells = new Set();
    this.exploreSet = new Set();
    this._updateVisibility();

    // Reward tracking
    this.totalReward = 0;
    this.rewardHistory = [];
    this.fireSizeHistory = [];
    this.actionCounts = {move:0, door:0, look:0, wait:0};
    this.eventLog = [];
    this.lastAction = null;
    this.lastReward = 0;
  }

  _updateVisibility() {
    this.visibleCells = computeVisible(this.agentX, this.agentY, this.cellGrid, this.smokeGrid, this.w, this.h);
    if (!this.exploreSet) this.exploreSet = new Set();
    for (const k of this.visibleCells) this.exploreSet.add(k);
  }

  _fireSize() {
    let count=0;
    for (let i=0;i<this.fireGrid.length;i++) if(this.fireGrid[i]>=FIRE_BURNING) count++;
    return count;
  }

  step(action) {
    if (!this.agentAlive || this.agentEvacuated || this.stepCount>=this.maxSteps) return false;

    const prevX=this.agentX, prevY=this.agentY;
    let feedback = '';

    // Execute action
    const deltas = {north:[0,-1],south:[0,1],east:[1,0],west:[-1,0]};
    if (action.action==='move') {
      const [dx,dy]=deltas[action.direction]||[0,0];
      const nx=this.agentX+dx, ny=this.agentY+dy;
      if (nx>=0&&nx<this.w&&ny>=0&&ny<this.h) {
        const ct=this.cellGrid[ny*this.w+nx];
        if (ct!==WALL&&ct!==OBSTACLE&&ct!==DOOR_CLOSED) {
          this.agentX=nx; this.agentY=ny;
          feedback=`Moved ${action.direction}`;
        } else feedback=`Blocked: ${ct===DOOR_CLOSED?'door closed':'wall'}`;
      }
      this.actionCounts.move++;
    } else if (action.action==='door') {
      const doorPos = this.doorRegistry[action.target_id];
      if (doorPos) {
        const [dx,dy]=doorPos;
        if (Math.abs(dx-this.agentX)+Math.abs(dy-this.agentY)<=2) {
          const cur=this.cellGrid[dy*this.w+dx];
          if (action.door_state==='open'&&cur===DOOR_CLOSED) {
            this.cellGrid[dy*this.w+dx]=DOOR_OPEN;
            feedback=`Opened ${action.target_id}`;
          } else if (action.door_state==='close'&&cur===DOOR_OPEN) {
            this.cellGrid[dy*this.w+dx]=DOOR_CLOSED;
            feedback=`Closed ${action.target_id} — slowing fire`;
          }
        }
      }
      this.actionCounts.door++;
    } else if (action.action==='wait') {
      feedback='Waiting...'; this.actionCounts.wait++;
    } else if (action.action==='look') {
      feedback=`Looking ${action.direction||''}`; this.actionCounts.look++;
    }

    // Check evacuation
    const agentCell=this.cellGrid[this.agentY*this.w+this.agentX];
    if (agentCell===EXIT && this.fireGrid[this.agentY*this.w+this.agentX]<EXIT_BLOCKED_FIRE_THRESHOLD) {
      this.agentEvacuated=true;
      feedback='EVACUATED — reached safety!';
    }

    // Advance fire
    const burned = this.sim.step(this.cellGrid, this.fireGrid, this.smokeGrid, this.burnTimers);

    // Health damage
    const ai=this.agentY*this.w+this.agentX;
    const smoke=this.smokeGrid[ai], fire=this.fireGrid[ai];
    let dmg=0;
    if (smoke>=0.8) dmg+=5; else if (smoke>=0.5) dmg+=2; else if (smoke>=0.2) dmg+=0.5;
    if (fire>=FIRE_BURNING) dmg+=10;
    this.agentHealth=Math.max(0, this.agentHealth-dmg);
    if (this.agentHealth<=0) { this.agentAlive=false; feedback='Incapacitated by fire and smoke!'; }

    this.stepCount++;
    this._updateVisibility();

    // Reward (simplified rubric)
    const prevDist=bfsExitDist(prevX,prevY,this.exits,this.cellGrid,this.w,this.h);
    const curDist=bfsExitDist(this.agentX,this.agentY,this.exits,this.cellGrid,this.w,this.h);
    let reward=-0.01;
    if (curDist<prevDist) reward+=0.1;
    if (smoke>=0.5||fire>=FIRE_BURNING) reward-=0.5;
    reward-=0.02*dmg;
    if (this.agentEvacuated) reward+=5.0+(0.05*Math.max(0,this.maxSteps-this.stepCount));
    if (!this.agentAlive) reward-=10.0;
    if (action.action==='door'&&action.door_state==='close') reward+=0.5;

    this.lastReward=Math.round(reward*1000)/1000;
    this.totalReward=Math.round((this.totalReward+reward)*1000)/1000;
    this.rewardHistory.push(this.totalReward);
    this.fireSizeHistory.push(this._fireSize());

    if (feedback) this.eventLog.unshift({step:this.stepCount, text:feedback, reward:this.lastReward});
    if (this.eventLog.length>50) this.eventLog.pop();
    if (burned.length) this.eventLog.unshift({step:this.stepCount, text:`${burned.length} cell(s) burned out`, reward:0, isAlert:true});

    this.lastAction = action;
    return !this.agentEvacuated && this.agentAlive && this.stepCount<this.maxSteps;
  }

  get done() { return !this.agentAlive||this.agentEvacuated||this.stepCount>=this.maxSteps; }
  get percentBurned() { return Math.round(100*Array.from(this.fireGrid).filter(f=>f>0).length/(this.w*this.h)); }
  get exitBlocked() { return this.exits.filter(([ex,ey])=>this.fireGrid[ey*this.w+ex]>=EXIT_BLOCKED_FIRE_THRESHOLD); }
  get smokeLevel() {
    const s=this.smokeGrid[this.agentY*this.w+this.agentX];
    return s>=0.8?'heavy':s>=0.5?'moderate':s>=0.2?'light':'clear';
  }
  get healthStatus() {
    if(this.agentHealth>75) return 'Good';
    if(this.agentHealth>50) return 'Moderate';
    if(this.agentHealth>25) return 'Low';
    return 'Critical';
  }
}
