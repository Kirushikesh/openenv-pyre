import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import type { Observation, Door, ApiReport, SceneResponse } from './types';
import Map2D from './components/Map2D';
import HUD from './components/HUD';
import ControlPanel from './components/ControlPanel';
import APIReport from './components/APIReport';

const DOOR_CLOSED = 3;
const OBSTACLE    = 5;

interface EventEntry {
  step: number;
  text: string;
  reward: number;
  isAlarm?: boolean;
}

function App() {
  const [observation,    setObservation]    = useState<Observation | null>(null);
  const [sceneData,      setSceneData]      = useState<SceneResponse | null>(null);
  const [isAutoWait,     setIsAutoWait]     = useState(false);
  const [isPolling,      setIsPolling]      = useState(true);
  const [status,         setStatus]         = useState('Idle — waiting for connection');
  const [isError,        setIsError]        = useState(false);
  const [apiReport,      setApiReport]      = useState<ApiReport | null>(null);
  const [agentMoveCount, setAgentMoveCount] = useState(0);
  const [agentMoveFlash, setAgentMoveFlash] = useState(0);
  const [eventLog,       setEventLog]       = useState<EventEntry[]>([]);

  const prevAgentPos  = useRef({ x: -1, y: -1 });
  const autoWaitTimer = useRef<number | null>(null);
  const logEndRef     = useRef<HTMLDivElement | null>(null);

  /* scroll event log to bottom */
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [eventLog]);

  const setStatusMsg = (msg: string, error = false) => {
    setStatus(msg);
    setIsError(error);
  };

  const pushLog = (text: string, step: number, reward: number, isAlarm = false) => {
    setEventLog(prev => [...prev.slice(-49), { step, text, reward, isAlarm }]);
  };

  const applyObservation = useCallback((obs: Observation) => {
    const newX = obs.map_state.agent_x;
    const newY = obs.map_state.agent_y;
    if (prevAgentPos.current.x !== -1 &&
       (newX !== prevAgentPos.current.x || newY !== prevAgentPos.current.y)) {
      setAgentMoveFlash(18);
      setAgentMoveCount(c => c + 1);
    }
    prevAgentPos.current = { x: newX, y: newY };
    setObservation(obs);
  }, []);

  const updateReport = (kind: string, request: unknown, response: any) => {
    const mapState = response?.observation?.map_state || response?.map_state || response?.graph;
    const template = mapState?.template_name || response?.labels?.episode?.template || 'unknown';
    const step     = mapState?.step_count ?? response?.observation?.elapsed_steps ?? response?.labels?.episode?.step ?? '-';
    const reward   = Number(response?.reward ?? 0).toFixed(3);
    const done     = Boolean(response?.done);
    setApiReport({
      call_type: kind,
      request,
      response,
      meta: `${kind.toUpperCase()} | template=${template} | step=${step} | reward=${reward} | done=${done}`,
    });
  };

  const apiCall = async (path: string, payload: unknown) => {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {}),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return res.json();
  };

  const resetLive = async (difficulty = 'medium') => {
    try {
      setStatusMsg('Initiating Reset...');
      const payload = { difficulty };
      const data = await apiCall('/reset', payload);
      const obs: Observation = data.observation;
      if (data.observation?.map_state) {
        obs.metadata = {
          fire_sources:    data.observation.fire_sources_count ?? 0,
          fire_spread_rate:data.observation.fire_spread_rate   ?? 0,
          humidity:        data.observation.humidity           ?? 0,
          difficulty,
        };
      }
      applyObservation(obs);
      updateReport('reset', payload, data);
      setStatusMsg(`Ready. Reward: ${Number(data.reward || 0).toFixed(2)}`);
      pushLog('Episode reset. Assess surroundings.', obs.map_state.step_count, data.reward ?? 0);
    } catch (err: any) {
      setStatusMsg(`Reset Failed: ${err.message}`, true);
    }
  };

  const resetUntilDoors = async () => {
    try {
      setStatusMsg('Searching for layout with doors...');
      for (let i = 1; i <= 8; i++) {
        const payload = { difficulty: 'medium' };
        const data    = await apiCall('/reset', payload);
        const doorCount = Object.keys(data?.observation?.map_state?.door_registry || {}).length;
        if (doorCount > 0) {
          const obs: Observation = data.observation;
          obs.metadata = {
            fire_sources:    data.observation.fire_sources_count ?? 0,
            fire_spread_rate:data.observation.fire_spread_rate   ?? 0,
            humidity:        data.observation.humidity           ?? 0,
            difficulty:      'medium',
          };
          applyObservation(obs);
          updateReport('reset', payload, data);
          setStatusMsg(`System Ready — ${doorCount} door(s) detected.`);
          pushLog(`Layout found (${doorCount} doors). Doors detected.`, obs.map_state.step_count, 0);
          return;
        }
      }
      setStatusMsg('Optimal layout not found after 8 attempts.', true);
    } catch (err: any) {
      setStatusMsg(`Search Failed: ${err.message}`, true);
    }
  };

  const runAction = async (actionObj: unknown, label: string) => {
    try {
      setStatusMsg(`Action: ${label}`);
      const payload = actionObj;
      const data    = await apiCall('/step', payload);
      const obs: Observation = data.observation;
      obs.metadata = observation?.metadata;
      applyObservation(obs);
      updateReport('step', payload, data);
      const rwd = Number(data.reward || 0);
      setStatusMsg(`Executed. Reward: ${rwd.toFixed(2)}`);
      pushLog(
        obs.last_action_feedback || label,
        obs.map_state.step_count,
        rwd,
        rwd < -0.5,
      );
      if (data.done) setIsAutoWait(false);
    } catch (err: any) {
      setStatusMsg(`Error: ${err.message}`, true);
    }
  };

  useEffect(() => {
    if (isAutoWait) {
      autoWaitTimer.current = window.setInterval(() => runAction({ action: 'wait' }, 'AUTO WAIT'), 900);
    } else {
      if (autoWaitTimer.current) clearInterval(autoWaitTimer.current);
    }
    return () => { if (autoWaitTimer.current) clearInterval(autoWaitTimer.current); };
  }, [isAutoWait]);

  useEffect(() => {
    if (isPolling) {
      const es = new EventSource('/live-movements');
      
      es.onmessage = (event) => {
        try {
          const scene: SceneResponse = JSON.parse(event.data);
          if (scene.error) {
            console.error('SSE data error:', scene.error);
            return;
          }
          setSceneData(scene);
          updateReport('scene', {}, scene);

          const { labels, graph } = scene;
          const cell_grid:  number[] = [];
          const fire_grid:  number[] = [];
          const smoke_grid: number[] = [];

          for (let y = 0; y < graph.height; y++) {
            for (let x = 0; x < graph.width; x++) {
              const [type, fire, smoke] = graph.grid[y][x];
              cell_grid.push(type);
              fire_grid.push(fire);
              smoke_grid.push(smoke);
            }
          }

          const visible_cells: [number, number][] = [];
          for (let y = 0; y < graph.height; y++) {
            for (let x = 0; x < graph.width; x++) {
              if (graph.grid[y][x][4] === 1.0) visible_cells.push([x, y]);
            }
          }

          const pseudoObs: Observation = {
            map_state: {
              cell_grid, fire_grid, smoke_grid,
              agent_x:      labels.agent.x,
              agent_y:      labels.agent.y,
              visible_cells,
              door_registry:  labels.map.door_registry,
              exit_positions: labels.map.exit_positions,
              step_count:     labels.episode.step,
              max_steps:      labels.episode.max_steps,
              grid_w:         graph.width,
              grid_h:         graph.height,
              template_name:  labels.episode.template,
            },
            agent_health:         labels.agent.health,
            location_label:       labels.agent.location,
            smoke_level:          labels.agent.smoke_level,
            wind_dir:             labels.episode.wind_dir,
            fire_visible:         labels.agent.fire_visible,
            fire_direction:       labels.agent.fire_direction,
            last_action_feedback: labels.agent.last_action_feedback,
            narrative: '',
            metadata: {
              fire_sources:    labels.episode.fire_sources,
              fire_spread_rate:labels.episode.fire_spread_rate,
              humidity:        labels.episode.humidity,
              difficulty:      labels.episode.difficulty,
            },
          };
          applyObservation(pseudoObs);
        } catch (err: any) {
          setStatusMsg(`SSE Parse Error: ${err.message}`, true);
        }
      };

      es.onerror = (err) => {
        console.error('SSE Error:', err);
        setStatusMsg('SSE Connection lost. Retrying...', true);
      };

      return () => es.close();
    }
  }, [isPolling, applyObservation, updateReport, applyObservation]); // Note: dependencies updated to include needed setters

  useEffect(() => {
    if (agentMoveFlash > 0) {
      const timer = setTimeout(() => setAgentMoveFlash(f => f - 1), 50);
      return () => clearTimeout(timer);
    }
  }, [agentMoveFlash]);

  const setup = async () => {
    setIsPolling(false);
    setAgentMoveCount(0);
    setAgentMoveFlash(0);
    setEventLog([]);
    prevAgentPos.current = { x: -1, y: -1 };
    await resetLive();
    setIsPolling(true);
  };

  /* Derived state */
  const doors: Door[] = Object.entries(observation?.map_state.door_registry || {})
    .map(([id, [x, y]]) => {
      const ct = observation?.map_state.cell_grid[y * (observation?.map_state.grid_w ?? 16) + x];
      let state: 'open' | 'closed' | 'failed' = 'open';
      if (ct === DOOR_CLOSED) state = 'closed';
      if (ct === OBSTACLE)   state = 'failed';
      return { id, x, y, state };
    })
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  const fireCells   = observation?.map_state.fire_grid.filter(v => v > 0.05).length ?? 0;
  const exploredPct = observation
    ? Math.round((new Set(observation.map_state.visible_cells.map(([vx, vy]) => `${vx},${vy}`)).size
        / observation.map_state.cell_grid.length) * 100)
    : 0;

  const hp        = Math.round(observation?.agent_health ?? 0);
  const hpColor   = hp >= 60 ? 'var(--green)' : hp >= 30 ? 'var(--amber)' : 'var(--red)';
  const isOnline  = isPolling && !isError;
  const epId      = sceneData?.labels.episode.id?.slice(0, 8) ?? '—';

  return (
    <div className="shell">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">🔥</div>
          <span className="brand-name">Pyre</span>
          <span className="brand-sep">/</span>
          <span className="brand-sub">Crisis Navigation</span>
        </div>

        <div className="topbar-right">
          <div className="topbar-ep">
            Episode <span className="ep-id">{epId}</span>
          </div>
          <span className={`live-chip ${isError ? 'error' : isOnline ? 'online' : 'offline'}`}>
            {isError ? 'Error' : isOnline ? 'Live' : 'Idle'}
          </span>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="content">

        {/* ── Left: Canvas Zone ── */}
        <div className="canvas-zone">
          <div className="canvas-frame">
            <Map2D observation={observation} agentMoveFlash={agentMoveFlash} />
            <HUD observation={observation} agentMoveCount={agentMoveCount} />
          </div>

          {/* Legend */}
          <div className="legend">
            {[
              { color:'#5e5850', label:'Wall'      },
              { color:'#3a3530', label:'Obstacle'  },
              { color:'#e6f4ec', label:'Exit'      },
              { color:'#7c5c3c', label:'Door'      },
              { color:'#f97316', label:'Fire'      },
              { color:'rgba(72,82,96,0.7)', label:'Smoke' },
              { color:'#3b82f6', label:'Agent'     },
              { color:'rgba(2,132,199,0.6)', label:'Trail'},
            ].map(({ color, label }) => (
              <div key={label} className="legend-item">
                <div className="leg-swatch" style={{ background: color, border:'1px solid rgba(0,0,0,0.1)' }} />
                {label}
              </div>
            ))}
          </div>

          {/* Field Report */}
          <div className="dialog">
            <span className="dialog-who">Field Report</span>
            {observation?.last_action_feedback || 'Establishing link to field systems...'}
          </div>
        </div>

        {/* ── Right: Side Panel ── */}
        <aside className="side">

          {/* Controls */}
          <ControlPanel
            onAction={runAction}
            onReset={resetLive}
            onResetDoors={resetUntilDoors}
            onSetup={setup}
            doors={doors}
            isAutoWait={isAutoWait}
            toggleAutoWait={() => setIsAutoWait(!isAutoWait)}
            isPolling={isPolling}
            togglePolling={() => setIsPolling(!isPolling)}
            status={status}
            isError={isError}
          />

          {/* Agent Biometrics */}
          <div className="side-sec">
            <div className="sec-hd">Agent Biometrics</div>
            <div className="sg">
              <div className="sc">
                <div className="sc-l">Health</div>
                <div className="sc-v" style={{ color: hpColor }}>{hp}%</div>
              </div>
              <div className="sc">
                <div className="sc-l">Status</div>
                <div className="sc-v" style={{ color: hpColor }}>
                  {sceneData?.labels.agent.health_status ?? 'NOMINAL'}
                </div>
              </div>
              <div className="sc">
                <div className="sc-l">Position</div>
                <div className="sc-v blue">
                  ({observation?.map_state.agent_x ?? '—'},{observation?.map_state.agent_y ?? '—'})
                </div>
              </div>
              <div className="sc">
                <div className="sc-l">Sector</div>
                <div className="sc-v">{observation?.location_label ?? 'Unknown'}</div>
              </div>
            </div>
            <div className="bar-w">
              <div className="bar-lbl">
                <span>System Integrity</span>
                <span>{hp}%</span>
              </div>
              <div className="bar-bg">
                <div className="bar-fill" style={{ width: `${hp}%`, background: hpColor }} />
              </div>
            </div>
          </div>

          {/* Environment */}
          <div className="side-sec">
            <div className="sec-hd">Environment</div>
            <div className="sg">
              <div className="sc">
                <div className="sc-l">Hazard Cells</div>
                <div className="sc-v fire">{fireCells}</div>
              </div>
              <div className="sc">
                <div className="sc-l">Explored</div>
                <div className="sc-v blue">{exploredPct}%</div>
              </div>
              <div className="sc">
                <div className="sc-l">Wind</div>
                <div className="sc-v">{observation?.wind_dir ?? 'CALM'}</div>
              </div>
              <div className="sc">
                <div className="sc-l">Humidity</div>
                <div className="sc-v amber">
                  {Math.round((observation?.metadata?.humidity ?? 0) * 100)}%
                </div>
              </div>
            </div>
          </div>

          {/* Event Log */}
          <div className="side-sec" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="sec-hd">
              Event Log
              <span style={{ fontFamily: 'var(--mono)', fontSize: '9px', color: 'var(--t3)' }}>
                {eventLog.length} events
              </span>
            </div>
            <div className="elog">
              {eventLog.length === 0 && (
                <div style={{ color: 'var(--t3)', fontFamily: 'var(--mono)', fontSize: '10px', padding: '4px' }}>
                  No events yet…
                </div>
              )}
              {eventLog.map((e, i) => (
                <div key={i} className={`erow ${e.isAlarm ? 'alarm' : ''}`}>
                  <span className="estep">T{e.step}</span>
                  <span className="etext">{e.text}</span>
                  <span className={`erwd ${e.reward >= 0 ? 'p' : 'n'}`}>
                    {e.reward >= 0 ? '+' : ''}{e.reward.toFixed(2)}
                  </span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {/* API Report */}
          <div className="side-sec">
            <div className="sec-hd">Network Activity</div>
            <APIReport report={apiReport} onCopyReset={() => {}} onCopyStep={() => {}} onCopyScene={() => {}} />
          </div>

        </aside>
      </div>
    </div>
  );
}

export default App;
