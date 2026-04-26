import React, { useEffect, useRef, useState } from 'react';
import type { Observation } from '../types';

interface HUDProps {
  observation: Observation | null;
  agentMoveCount?: number;
}

/* Unicode compass arrows for each wind direction */
const WIND_ARROW: Record<string, string> = {
  N: '↑', S: '↓', E: '→', W: '←',
  NE: '↗', NW: '↖', SE: '↘', SW: '↙',
  CALM: '·',
};

/* Rotation degrees so the arrow visually points in the right direction */
const WIND_DEG: Record<string, number> = {
  N: 0, NE: 45, E: 90, SE: 135,
  S: 180, SW: 225, W: 270, NW: 315, CALM: 0,
};

const HUD: React.FC<HUDProps> = ({ observation, agentMoveCount = 0 }) => {
  const [firePulse, setFirePulse] = useState(0);
  const pulseRef = useRef(0);

  useEffect(() => {
    let raf: number;
    const tick = () => {
      pulseRef.current += 0.05;
      setFirePulse(Math.sin(pulseRef.current * 4) * 0.5 + 0.5);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  if (!observation) return null;

  const { map_state, agent_health, smoke_level, wind_dir, fire_visible, metadata } = observation;
  const hPct = Math.round(agent_health);
  const sPct = Math.round((map_state.step_count / map_state.max_steps) * 100);
  const totalFireCells = map_state.fire_grid.filter(v => v > 0.05).length;

  let hBarClass    = 'g';
  let hStatusLabel = 'Nominal';
  let hStatusClass = 'good';

  if (hPct < 30)      { hBarClass = 'c'; hStatusLabel = 'Critical'; hStatusClass = 'critical'; }
  else if (hPct < 60) { hBarClass = 'm'; hStatusLabel = 'Moderate'; hStatusClass = 'moderate'; }

  const windDir    = wind_dir || 'CALM';
  const windArrow  = WIND_ARROW[windDir] ?? '?';
  const windDeg    = WIND_DEG[windDir]   ?? 0;
  const spreadRate = metadata?.fire_spread_rate ?? 0;
  const humidity   = metadata?.humidity ?? 0;

  return (
    <div className="hud-overlay">
      {/* ── Left: Health ── */}
      <div className="hud-card">
        <div className="hud-r">
          <span className="hlbl">HP</span>
          <div className="hbar-bg">
            <div className={`hbar-fill ${hBarClass}`} style={{ width: `${hPct}%` }} />
          </div>
          <span className="hval">{hPct}</span>
        </div>
        <div className="hud-r" style={{ gap: '8px', marginTop: '2px' }}>
          <span className={`hstatus ${hStatusClass}`}>{hStatusLabel}</span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '9px', color: 'rgba(168,162,158,.55)', marginLeft: 'auto' }}>
            💨 {smoke_level || 'clear'}
          </span>
        </div>
        {agentMoveCount > 0 && (
          <div style={{ fontFamily: 'var(--mono)', fontSize: '8px', color: 'rgba(168,162,158,.4)', marginTop: '2px' }}>
            moves: {agentMoveCount}
          </div>
        )}
      </div>

      {/* ── Center: Wind & Hazard ── */}
      <div className="hud-card hud-card-center">
        {/* Compass rose */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ position: 'relative', width: 32, height: 32, flexShrink: 0 }}>
            {/* compass ring */}
            <svg width="32" height="32" viewBox="0 0 32 32" style={{ position: 'absolute', top: 0, left: 0 }}>
              <circle cx="16" cy="16" r="14" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1.5" />
              <text x="16" y="6"  textAnchor="middle" fontSize="5" fill="rgba(255,255,255,0.35)" dominantBaseline="middle">N</text>
              <text x="16" y="28" textAnchor="middle" fontSize="5" fill="rgba(255,255,255,0.35)" dominantBaseline="middle">S</text>
              <text x="4"  y="17" textAnchor="middle" fontSize="5" fill="rgba(255,255,255,0.35)" dominantBaseline="middle">W</text>
              <text x="28" y="17" textAnchor="middle" fontSize="5" fill="rgba(255,255,255,0.35)" dominantBaseline="middle">E</text>
            </svg>
            {/* direction arrow */}
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: `translate(-50%,-50%) rotate(${windDeg}deg)`,
              fontSize: windDir === 'CALM' ? '10px' : '14px',
              color: windDir === 'CALM' ? 'rgba(168,162,158,0.6)' : '#fbbf24',
              lineHeight: 1,
              transition: 'transform 0.6s ease',
            }}>
              {windArrow}
            </div>
          </div>

          <div>
            <div className="hlbl">Wind</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '13px', fontWeight: 500, color: '#f0e8e0', lineHeight: 1.2 }}>
              {windDir}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: '4px' }}>
          <div>
            <div className="hlbl">Spread</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: spreadRate > 0.5 ? '#f87171' : '#fbbf24' }}>
              {(spreadRate * 100).toFixed(0)}%
            </div>
          </div>
          <div>
            <div className="hlbl">Humidity</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: humidity > 0.6 ? '#60a5fa' : '#a8a29e' }}>
              {(humidity * 100).toFixed(0)}%
            </div>
          </div>
          {totalFireCells > 0 && (
            <div style={{ alignSelf: 'center' }}>
              <span style={{
                fontFamily: 'var(--mono)', fontSize: '9px', fontWeight: 700,
                color: fire_visible ? '#fff' : '#fbbf24',
                background: fire_visible
                  ? `rgba(239,${Math.floor(30 + firePulse * 40)},0,${0.75 + firePulse * 0.25})`
                  : `rgba(180,60,0,${0.55 + firePulse * 0.3})`,
                border: fire_visible
                  ? `1px solid rgba(255,${Math.floor(60 + firePulse * 80)},0,0.8)`
                  : '1px solid rgba(251,191,36,0.5)',
                padding: '2px 6px', borderRadius: '3px', letterSpacing: '0.06em',
                boxShadow: fire_visible
                  ? `0 0 ${6 + firePulse * 8}px rgba(255,80,0,0.7)`
                  : `0 0 ${3 + firePulse * 4}px rgba(200,80,0,0.4)`,
                display: 'flex', alignItems: 'center', gap: '3px',
                transition: 'box-shadow 0.1s',
              }}>
                🔥 {fire_visible ? 'IN RANGE' : 'ACTIVE'} · {totalFireCells}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ── Right: Steps ── */}
      <div className="hud-card" style={{ textAlign: 'right', minWidth: 'auto' }}>
        <div className="hud-r" style={{ justifyContent: 'flex-end' }}>
          <span className="step-val">{map_state.step_count} / {map_state.max_steps}</span>
        </div>
        <div className="sbar-bg">
          <div className="sbar-fill" style={{ width: `${sPct}%` }} />
        </div>
        <div className="step-meta">
          {metadata?.difficulty ?? 'medium'} · {map_state.template_name}
        </div>
      </div>
    </div>
  );
};

export default HUD;
