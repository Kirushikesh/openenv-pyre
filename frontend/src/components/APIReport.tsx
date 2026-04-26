import React from 'react';
import type { ApiReport } from '../types';

interface APIReportProps {
  report: ApiReport | null;
  onCopyReset: () => void;
  onCopyStep: () => void;
  onCopyScene: () => void;
}

const APIReport: React.FC<APIReportProps> = ({ report, onCopyReset, onCopyStep, onCopyScene }) => {
  return (
    <div>
      <div className="report-meta" style={{ marginBottom: '8px' }}>
        {report ? report.meta : 'Awaiting telemetry...'}
      </div>
      <div className="report-box">
        {report ? JSON.stringify({
          call_type: report.call_type,
          request: report.request,
          response: report.response
        }, null, 2) : '{}'}
      </div>
      <div className="ctrl-grid" style={{ marginTop: '12px' }}>
        <button className="ctrl-btn" onClick={onCopyScene}>Scene</button>
        <button className="ctrl-btn" onClick={onCopyReset}>Reset</button>
        <button className="ctrl-btn" onClick={onCopyStep}>Step</button>
      </div>
    </div>
  );
};

export default APIReport;
