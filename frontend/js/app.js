/**
 * app.js — Pyre visualization controller
 * Handles episode lifecycle, HUD updates, charts, live/demo modes.
 */

// ─── Mini chart renderer ─────────────────────────────────────────────────────

class MiniChart {
  constructor(canvas, opts={}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.color = opts.color || '#00e5ff';
    this.fill  = opts.fill  || 'rgba(0,229,255,0.12)';
    this.label = opts.label || '';
    this.maxPoints = opts.maxPoints || 200;
    this.yMin = opts.yMin !== undefined ? opts.yMin : null;
    this.yMax = opts.yMax !== undefined ? opts.yMax : null;
    this.data = [];
  }

  push(v) {
    this.data.push(v);
    if (this.data.length > this.maxPoints) this.data.shift();
  }

  draw() {
    const {canvas, ctx, data, color, fill} = this;
    const w=canvas.width, h=canvas.height;
    ctx.fillStyle='#faf9f6'; ctx.fillRect(0,0,w,h);
    if (data.length < 2) return;

    const yMin = this.yMin !== null ? this.yMin : Math.min(...data);
    const yMax = this.yMax !== null ? this.yMax : Math.max(...data);
    const yRange = (yMax-yMin) || 1;
    const pad = 4;

    // Grid lines
    ctx.strokeStyle='rgba(0,0,0,0.06)';
    ctx.lineWidth=0.5;
    for (let i=0;i<=3;i++) {
      const gy=pad+(h-2*pad)*(i/3);
      ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(w,gy); ctx.stroke();
    }

    const toX = i => (i/(data.length-1))*w;
    const toY = v => h-pad - ((v-yMin)/yRange)*(h-2*pad);

    // Fill area
    ctx.beginPath();
    ctx.moveTo(toX(0), h);
    for (let i=0;i<data.length;i++) ctx.lineTo(toX(i), toY(data[i]));
    ctx.lineTo(toX(data.length-1), h);
    ctx.closePath();
    ctx.fillStyle=fill;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i=1;i<data.length;i++) ctx.lineTo(toX(i), toY(data[i]));
    ctx.strokeStyle=color;
    ctx.lineWidth=1.8;
    ctx.stroke();

    // Current value dot
    const lastX=toX(data.length-1), lastY=toY(data[data.length-1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI*2);
    ctx.fillStyle=color;
    ctx.fill();

    // Current value label
    ctx.fillStyle=color;
    ctx.font='10px monospace';
    ctx.textAlign='right';
    ctx.fillText(data[data.length-1].toFixed(1), w-2, lastY-4);
  }
}

// ─── Action histogram ────────────────────────────────────────────────────────

function drawActionHistogram(canvas, counts) {
  const ctx = canvas.getContext('2d');
  const w=canvas.width, h=canvas.height;
  ctx.fillStyle='#faf9f6'; ctx.fillRect(0,0,w,h);

  const entries = Object.entries(counts);
  const total = entries.reduce((s,[,v])=>s+v,0)||1;
  const colors = {move:'#1d4ed8', door:'#c2410c', wait:'#7c3aed', look:'#166534'};
  const barH = Math.floor((h-8)/entries.length)-2;

  entries.forEach(([k,v],i) => {
    const ratio = v/total;
    const bw = Math.max(2, ratio*(w-60));
    const by = 4+i*(barH+2);
    // Bar bg
    ctx.fillStyle='rgba(0,0,0,0.05)';
    ctx.fillRect(60,by,w-62,barH);
    // Bar fill
    ctx.fillStyle=colors[k]||'#888';
    ctx.fillRect(60,by,bw,barH);
    // Label
    ctx.fillStyle='rgba(80,70,65,0.85)';
    ctx.font=`${Math.min(11,barH-1)}px monospace`;
    ctx.textAlign='right';
    ctx.fillText(k, 54, by+barH-2);
    // Count
    ctx.fillStyle=colors[k]||'#888';
    ctx.textAlign='left';
    ctx.fillText(v, 64+bw, by+barH-2);
  });
}

// ─── Live mode client ────────────────────────────────────────────────────────

class LiveClient {
  constructor(baseUrl='https://krooz-pyre-env.hf.space') {
    this.baseUrl=baseUrl;
    this.connected=false;
  }
  async checkHealth() {
    try {
      const r=await fetch(`${this.baseUrl}/web/health`,{signal:AbortSignal.timeout(3000)});
      this.connected=r.ok;
    } catch { this.connected=false; }
    return this.connected;
  }
  async reset(difficulty='medium', seed=null) {
    const body={difficulty};
    if(seed!==null) body.seed=parseInt(seed)||null;
    const r=await fetch(`${this.baseUrl}/web/reset`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    return r.json();
  }
  async step(action) {
    const r=await fetch(`${this.baseUrl}/web/step`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)});
    return r.json();
  }
  async getScene() {
    const r=await fetch(`${this.baseUrl}/web/scene`);
    return r.json();
  }
}

// ─── App controller ───────────────────────────────────────────────────────────

class PyreApp {
  constructor() {
    this.canvas = document.getElementById('mainCanvas');
    this.windCanvas = document.getElementById('windCanvas');
    this.renderer = new PyreRenderer(this.canvas);
    this.episode = null;
    this.agent = new DemoAgent();
    this.mode = 'demo'; // 'demo' | 'live'
    this.playing = false;
    this.speed = 1;     // steps per second
    this.stepInterval = null;
    this.animFrame = null;
    this.templateIdx = 0;
    this.difficulty = 'medium';
    this.seed = 42;
    this.liveClient = new LiveClient();
    this.liveScene = null;

    // Charts
    this.rewardChart = new MiniChart(document.getElementById('rewardChart'),{
      color:'#1d4ed8', fill:'rgba(29,78,216,0.08)', yMin:null, yMax:null
    });
    this.fireSizeChart = new MiniChart(document.getElementById('fireSizeChart'),{
      color:'#c2410c', fill:'rgba(194,65,12,0.08)', yMin:0
    });

    this._bindControls();
    this.newEpisode();
    this._startRenderLoop();
  }

  newEpisode() {
    this.playing = false;
    this._clearStepInterval();
    const seed = parseInt(document.getElementById('seedInput')?.value) || this.seed;
    this.seed = seed;
    this.episode = new PyreEpisode(this.templateIdx, seed, this.difficulty);
    this.renderer.init(this.episode);
    this.agent = new DemoAgent();
    this._updateHUD();
    this._clearCharts();
    this._updateEventLog();
    this._updatePlayBtn();
    // Auto-play
    setTimeout(()=>{this.playing=true;this._startStepInterval();this._updatePlayBtn();},300);
    this.templateIdx = (this.templateIdx+1) % 3; // cycle templates
  }

  _clearCharts() {
    this.rewardChart.data = [];
    this.fireSizeChart.data = [];
    this.rewardChart.draw();
    this.fireSizeChart.draw();
    const ac=document.getElementById('actionChart');
    if(ac){const c=ac.getContext('2d');c.clearRect(0,0,ac.width,ac.height);}
  }

  step() {
    if (!this.episode || this.episode.done) { this.playing=false; this._clearStepInterval(); this._updatePlayBtn(); return; }
    const action = this.agent.chooseAction({
      agentX:this.episode.agentX, agentY:this.episode.agentY,
      cellGrid:this.episode.cellGrid, fireGrid:this.episode.fireGrid,
      smokeGrid:this.episode.smokeGrid, exits:this.episode.exits,
      w:this.episode.w, h:this.episode.h, doorRegistry:this.episode.doorRegistry
    });
    this.episode.step(action);
    this._updateHUD();
    this._updateCharts();
    this._updateEventLog();

    if (this.episode.done) {
      this.playing=false;
      this._clearStepInterval();
      this._updatePlayBtn();
      this._showEpisodeEnd();
    }
  }

  togglePlay() {
    this.playing = !this.playing;
    if (this.playing) this._startStepInterval();
    else this._clearStepInterval();
    this._updatePlayBtn();
  }

  setSpeed(s) {
    this.speed = s;
    document.querySelectorAll('.speed-btn').forEach(b=>{
      b.classList.toggle('on', parseFloat(b.dataset.speed)===s);
    });
    if (this.playing) { this._clearStepInterval(); this._startStepInterval(); }
  }

  _startStepInterval() {
    this._clearStepInterval();
    const ms = Math.max(50, 1000/this.speed);
    this.stepInterval = setInterval(()=>this.step(), ms);
  }

  _clearStepInterval() {
    if (this.stepInterval) { clearInterval(this.stepInterval); this.stepInterval=null; }
  }

  // ── HUD updates ────────────────────────────────────────────────────────────

  _updateHUD() {
    const ep = this.episode;
    if (!ep) return;

    // Health bar
    const hPct = ep.agentHealth/100;
    const hBar = document.getElementById('healthBar');
    if (hBar) {
      hBar.style.width=`${ep.agentHealth}%`;
      hBar.className='hbar-fill ' + (hPct>0.6?'g':hPct>0.3?'m':'c');
    }
    const hVal = document.getElementById('healthVal');
    if(hVal) hVal.textContent=`${Math.round(ep.agentHealth)}`;
    const hStatus = document.getElementById('healthStatus');
    if(hStatus) { hStatus.textContent=ep.healthStatus; hStatus.className='hstatus '+ep.healthStatus.toLowerCase(); }

    // Step counter
    const stepEl=document.getElementById('stepCount');
    if(stepEl) stepEl.textContent=`${ep.stepCount} / ${ep.maxSteps}`;
    const progBar=document.getElementById('stepBar');
    if(progBar) progBar.style.width=`${100*ep.stepCount/ep.maxSteps}%`;

    // Stats
    setText('rewardVal',  ep.totalReward.toFixed(2));
    setText('burnPct',    `${ep.percentBurned}%`);
    setText('smokeVal', ep.smokeLevel);
    const smokeTag=document.getElementById('smokeTag');
    if(smokeTag) smokeTag.textContent=`smoke: ${ep.smokeLevel}`;
    setText('windVal',    ep.windDir);
    setText('humidVal',   `${Math.round(ep.humidity*100)}%`);
    setText('spreadVal',  ep.pSpread.toFixed(2));
    const diffTag=document.getElementById('diffTag');
    if(diffTag) {
      let dt=`${ep.difficulty} · ${ep.templateName}`;
      if(ep._fireDirection) dt+=` · fire ${ep._fireDirection}`;
      diffTag.textContent=dt;
    }

    // Episode status banner
    const banner=document.getElementById('epBanner');
    if(banner){
      if(ep.agentEvacuated){ banner.textContent='✓ EVACUATED'; banner.className='ok'; banner.style.display='flex'; }
      else if(!ep.agentAlive){ banner.textContent='✗ INCAPACITATED'; banner.className='bad'; banner.style.display='flex'; }
      else if(ep.stepCount>=ep.maxSteps){ banner.textContent='⏱ TIMEOUT'; banner.className='tmo'; banner.style.display='flex'; }
      else banner.style.display='none';
    }

    // Smoke level indicator
    const smokeInd=document.getElementById('smokeIndicator');
    if(smokeInd){ smokeInd.className='smoke-indicator '+ep.smokeLevel; smokeInd.title=`Smoke: ${ep.smokeLevel}`; }

    // Wind rose
    if(this.windCanvas) this.renderer.drawWindRose(this.windCanvas, ep.windDir);
  }

  _updateCharts() {
    const ep=this.episode;
    if(!ep) return;
    this.rewardChart.push(ep.totalReward);
    setText('rewardLast', ep.lastReward>0?'+'+ep.lastReward.toFixed(2):ep.lastReward.toFixed(2));
    this.rewardChart.draw();
    const fs=ep.fireSizeHistory[ep.fireSizeHistory.length-1]||0;
    this.fireSizeChart.push(fs);
    setText('fireLast', fs);
    this.fireSizeChart.draw();
    const ac=document.getElementById('actionChart');
    if(ac) drawActionHistogram(ac, ep.actionCounts);
  }

  _updateEventLog() {
    const logEl=document.getElementById('eventLog');
    if(!logEl||!this.episode) return;
    const entries=this.episode.eventLog.slice(0,18);
    logEl.innerHTML=entries.map(e=>`
      <div class="erow${e.isAlert?' alarm':''}">
        <span class="estep">${String(e.step).padStart(3,'0')}</span>
        <span class="etext">${e.text}</span>
        ${e.reward?`<span class="erwd ${e.reward>0?'p':'n'}">${e.reward>0?'+':''}${e.reward.toFixed(2)}</span>`:''}
      </div>`).join('');
  }

  _showEpisodeEnd() {
    const ep=this.episode;
    const banner=document.getElementById('statusBanner');
    if(!banner) return;
    if(ep.agentEvacuated) {
      banner.innerHTML=`✓ EVACUATED — ${ep.totalReward.toFixed(1)} pts in ${ep.stepCount} steps`;
      banner.className='ok';
    } else if(!ep.agentAlive) {
      banner.innerHTML=`✗ INCAPACITATED — health depleted`;
      banner.className='bad';
    } else {
      banner.innerHTML=`⏱ TIMEOUT — ${ep.stepCount} steps`;
      banner.className='tmo';
    }
    banner.style.display='flex';
  }

  _updatePlayBtn() {
    const btn=document.getElementById('playBtn');
    if(btn){btn.textContent=this.playing?'⏸':'▶';btn.classList.toggle('play',!this.playing);}
  }

  // ── Render loop ────────────────────────────────────────────────────────────

  _startRenderLoop() {
    const loop=()=>{ this._render(); this.animFrame=requestAnimationFrame(loop); };
    this.animFrame=requestAnimationFrame(loop);
  }

  _render() {
    if (!this.episode) return;
    this.renderer.frame(this.episode, 1);
    // Periodic chart updates when paused (to keep time display fresh)
  }

  // ── Mode switching ─────────────────────────────────────────────────────────

  async switchToLive() {
    const ok=await this.liveClient.checkHealth();
    const indicator=document.getElementById('liveIndicator');
    if(!ok) {
      if(indicator) { indicator.textContent='⚠ Server offline'; indicator.className='live-indicator offline'; }
      showToast('Cannot reach server at localhost:8000 — staying in Demo mode');
      return;
    }
    this.mode='live';
    if(indicator){ indicator.textContent='● LIVE'; indicator.className='live-indicator online'; }
    showToast('Connected to live server');
    await this._resetLive();
    this._startLiveLoop();
  }

  switchToDemo() {
    this.mode='demo';
    this._clearStepInterval();
    const indicator=document.getElementById('liveIndicator');
    if(indicator){ indicator.textContent='◉ DEMO'; indicator.className='live-indicator demo'; }
    this.newEpisode();
  }

  async _resetLive() {
    try {
      const seed=parseInt(document.getElementById('seedInput')?.value)||null;
      await this.liveClient.reset(this.difficulty, seed);
    } catch(e){ console.warn('Live reset failed',e); }
  }

  async _startLiveLoop() {
    const poll=async()=>{
      if(this.mode!=='live') return;
      try {
        const scene=await this.liveClient.getScene();
        this._applyLiveScene(scene);
      } catch(e){ console.warn('Live poll failed',e); }
      if(this.mode==='live') setTimeout(poll, 800);
    };
    poll();
  }

  _applyLiveScene(scene) {
    if(!scene||!scene.graph) return;
    const g=scene.graph, lb=scene.labels;
    const w=g.width, h=g.height;
    const n=w*h;
    const cellGrid=new Array(n), fireGrid=new Float32Array(n), smokeGrid=new Float32Array(n);
    const visibleCells=new Set();
    for(let y=0;y<h;y++) for(let x=0;x<w;x++){
      const cell=g.grid[y][x];
      const i=y*w+x;
      cellGrid[i]=cell[0]; fireGrid[i]=cell[1]; smokeGrid[i]=cell[2];
      if(cell[4]>0.5) visibleCells.add(`${x},${y}`);
    }
    if(!this.episode||this.episode.w!==w){
      this.episode=new PyreEpisode(0,42,'medium');
      this.renderer.init(this.episode);
    }
    this.episode.cellGrid=cellGrid;
    this.episode.fireGrid=fireGrid;
    this.episode.smokeGrid=smokeGrid;
    this.episode.visibleCells=visibleCells;
    // Agent state
    this.episode.agentX=lb.agent.x; this.episode.agentY=lb.agent.y;
    this.episode.agentHealth=lb.agent.health;
    this.episode.agentAlive=lb.agent.alive;
    this.episode.agentEvacuated=lb.agent.evacuated;
    // Episode metadata
    this.episode.stepCount=lb.episode.step;
    this.episode.maxSteps=lb.episode.max_steps;
    this.episode.windDir=lb.episode.wind_dir;
    this.episode.pSpread=lb.episode.fire_spread_rate;
    this.episode.humidity=lb.episode.humidity;
    this.episode.difficulty=lb.episode.difficulty||'medium';
    this.episode.templateName=lb.episode.template||'—';
    // Map
    this.episode.exits=lb.map.exit_positions||this.episode.exits;
    this.episode.doorRegistry=lb.map.door_registry||{};
    // Narrative fields (agent perception)
    this.episode._smokeLevel=lb.agent.smoke_level||'clear';
    this.episode._fireDirection=lb.agent.fire_direction||null;
    this.episode._locationLabel=lb.agent.location||'';
    this.episode._healthStatus=lb.agent.health_status||'Good';
    // Surroundings → inject into event log
    if(lb.surroundings) {
      const sur=lb.surroundings;
      if(sur.audible_signals&&sur.audible_signals.length) {
        const step=lb.episode.step;
        const prev=this.episode.eventLog[0];
        if(!prev||prev.step!==step||!prev._fromSurroundings) {
          for(const sig of sur.audible_signals.slice(0,2)) {
            this.episode.eventLog.unshift({step,text:`🔊 ${sig}`,reward:0,isAlert:true,_fromSurroundings:true});
          }
          if(this.episode.eventLog.length>50) this.episode.eventLog.length=50;
        }
      }
    }
    this.episode.w=w; this.episode.h=h;
    this._updateHUD();
  }

  // ── Control binding ────────────────────────────────────────────────────────

  _bindControls() {
    document.getElementById('playBtn')?.addEventListener('click',()=>this.togglePlay());
    document.getElementById('stepBtn')?.addEventListener('click',()=>this.step());
    document.getElementById('resetBtn')?.addEventListener('click',()=>this.newEpisode());
    document.getElementById('prevBtn')?.addEventListener('click',()=>{
      this.templateIdx=((this.templateIdx-2)+3)%3;
      this.newEpisode();
    });

    document.querySelectorAll('.speed-btn').forEach(b=>{
      b.addEventListener('click',()=>this.setSpeed(parseFloat(b.dataset.speed)));
    });

    document.querySelectorAll('.diff-btn').forEach(b=>{
      b.addEventListener('click',()=>{
        this.difficulty=b.dataset.diff;
        document.querySelectorAll('.diff-btn').forEach(x=>x.classList.toggle('active',x===b));
        this.newEpisode();
      });
    });

    document.querySelectorAll('.mode-btn').forEach(b=>{
      b.addEventListener('click',()=>{
        document.querySelectorAll('.mode-btn').forEach(x=>x.classList.remove('active','fire'));
        b.classList.add('active');
        if(b.dataset.mode==='live') this.switchToLive();
        else this.switchToDemo();
      });
    });

    document.getElementById('seedInput')?.addEventListener('change',()=>this.newEpisode());

    // Keyboard shortcuts
    window.addEventListener('keydown',e=>{
      if(e.target.tagName==='INPUT') return;
      if(e.code==='Space'){e.preventDefault();this.togglePlay();}
      else if(e.code==='ArrowRight') this.step();
      else if(e.code==='KeyR') this.newEpisode();
      else if(e.code==='Digit1') this.setSpeed(0.5);
      else if(e.code==='Digit2') this.setSpeed(1);
      else if(e.code==='Digit3') this.setSpeed(2);
      else if(e.code==='Digit4') this.setSpeed(4);
      else if(e.code==='Digit5') this.setSpeed(8);
    });
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────

function setText(id, val) {
  const el=document.getElementById(id);
  if(el) el.textContent=val;
}

function showToast(msg) {
  const t=document.getElementById('toast');
  if(!t) return;
  t.textContent=msg; t.classList.add('up');
  setTimeout(()=>t.classList.remove('up'),3000);
}

// Boot
window.addEventListener('DOMContentLoaded',()=>{ window.app=new PyreApp(); });
