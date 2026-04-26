import React from 'react';
import type { Door } from '../types';

interface ControlPanelProps {
  onAction: (action: unknown, label: string) => void;
  onReset: (difficulty: string) => void;
  onResetDoors: () => void;
  onSetup: () => void;
  doors: Door[];
  isAutoWait: boolean;
  toggleAutoWait: () => void;
  isPolling: boolean;
  togglePolling: () => void;
  status: string;
  isError: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  onAction, onSetup, doors,
  isAutoWait, toggleAutoWait,
  isPolling, togglePolling,
  status, isError,
}) => {
  const mv = (dir: string) => onAction({ action: 'move', direction: dir.toLowerCase() }, `MOVE ${dir.toUpperCase()}`);

  return (
    <div className="side-sec">
      <div className="sec-hd">Tactical Controls</div>

      {/* D-Pad */}
      <div className="ctrl-grid" style={{ maxWidth: 160, margin: '0 auto', marginBottom: 8 }}>
        {/* Row 1 */}
        <div />
        <button className="ctrl-btn" onClick={() => mv('north')}>▲</button>
        <div />
        {/* Row 2 */}
        <button className="ctrl-btn" onClick={() => mv('west')}>◀</button>
        <button className="ctrl-btn accent" onClick={() => onAction({ action: 'wait' }, 'WAIT')} title="Wait in place">●</button>
        <button className="ctrl-btn" onClick={() => mv('east')}>▶</button>
        {/* Row 3 */}
        <div />
        <button className="ctrl-btn" onClick={() => mv('south')}>▼</button>
        <div />
      </div>

      {/* Secondary actions */}
      <div className="ctrl-row">
        <button className="ctrl-btn" onClick={() => onAction({ action: 'look' }, 'LOOK')}>SCAN</button>
        <button className={`ctrl-btn ${isAutoWait ? 'active' : ''}`} onClick={toggleAutoWait}>
          {isAutoWait ? 'STOP AUTO' : 'AUTO'}
        </button>
        <button className={`ctrl-btn ${isPolling ? 'active' : ''}`} onClick={togglePolling}>
          {isPolling ? 'LIVE ●' : 'LIVE ○'}
        </button>
      </div>

      {/* Reboot */}
      <div className="ctrl-row" style={{ marginTop: 6 }}>
        <button className="ctrl-btn play" onClick={onSetup} style={{ flex: 'none', width: '100%' }}>
          ↺ REBOOT EPISODE
        </button>
      </div>

      {/* Status line */}
      <div className={`ctrl-status ${isError ? 'error' : ''}`}>
        {status}
      </div>

      {/* Doors */}
      {doors.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="sec-hd" style={{ marginBottom: 6 }}>Proximity Doors</div>
          <div className="door-grid">
            {doors.map(d => (
              <button
                key={d.id}
                className={`door-btn ${d.state}`}
                onClick={() => onAction({ action: 'door', target_id: d.id, door_state: d.state === 'closed' ? 'open' : 'close' }, `DOOR ${d.id}`)}
              >
                {d.id} [{d.state}]
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
