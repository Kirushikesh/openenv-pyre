import React from 'react';

interface StatusRowProps {
  label: string;
  value: string | number;
  className?: string;
}

export const StatusRow: React.FC<StatusRowProps> = ({ label, value, className = '' }) => (
  <div className="srow">
    <span>{label}</span>
    <span className={`sv ${className}`}>{value}</span>
  </div>
);

interface StatusCardProps {
  title: string;
  children: React.ReactNode;
}

export const StatusCard: React.FC<StatusCardProps> = ({ title, children }) => (
  <div className="card">
    <div className="card-title">{title}</div>
    {children}
  </div>
);
