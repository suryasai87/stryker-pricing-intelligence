import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const COMPETITORS = [
  { key: 'stryker', name: 'Stryker', color: '#0057B8' },
  { key: 'medtronic', name: 'Medtronic', color: '#8b5cf6' },
  { key: 'zimmer', name: 'Zimmer Biomet', color: '#f59e0b' },
  { key: 'smith_nephew', name: 'Smith & Nephew', color: '#ec4899' },
  { key: 'depuy', name: 'DePuy Synthes', color: '#06b6d4' },
];

/**
 * Custom tooltip for market share trend.
 */
function ShareTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const sorted = [...payload].sort((a, b) => (b.value || 0) - (a.value || 0));

  return (
    <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl min-w-[200px]">
      <p className="text-white/60 text-xs mb-2">{label}</p>
      <div className="space-y-1.5">
        {sorted.map((entry) => (
          <div key={entry.name} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-white/70 text-xs">{entry.name}</span>
            </div>
            <span className="text-white font-mono text-xs font-medium">{entry.value?.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Custom interactive legend.
 */
function InteractiveLegend({ competitors, hidden, onToggle }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3 mt-4">
      {competitors.map((comp) => {
        const isHidden = hidden.has(comp.key);
        return (
          <motion.button
            key={comp.key}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
              isHidden
                ? 'bg-white/5 text-white/30 border border-white/5'
                : 'bg-white/5 text-white/70 border border-white/10'
            }`}
            onClick={() => onToggle(comp.key)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={springTransition}
          >
            <div
              className="w-2.5 h-2.5 rounded-full transition-opacity"
              style={{
                backgroundColor: comp.color,
                opacity: isHidden ? 0.3 : 1,
              }}
            />
            <span style={{ textDecoration: isHidden ? 'line-through' : 'none' }}>{comp.name}</span>
          </motion.button>
        );
      })}
    </div>
  );
}

/**
 * MarketShareTrend - LineChart showing market share over time.
 *
 * Props:
 *   data    - array of time-series data points
 *   loading - boolean
 *   title   - string (default: 'Market Share Trends')
 */
export default function MarketShareTrend({
  data = null,
  loading = false,
  title = 'Market Share Trends',
}) {
  const [hiddenLines, setHiddenLines] = useState(new Set());

  const toggleLine = useCallback((key) => {
    setHiddenLines((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  // Default demo data
  const chartData = data || [
    { quarter: 'Q1 2023', stryker: 28.5, medtronic: 24.2, zimmer: 22.1, smith_nephew: 14.8, depuy: 10.4 },
    { quarter: 'Q2 2023', stryker: 29.1, medtronic: 23.8, zimmer: 21.9, smith_nephew: 14.5, depuy: 10.7 },
    { quarter: 'Q3 2023', stryker: 29.8, medtronic: 23.5, zimmer: 21.5, smith_nephew: 14.2, depuy: 11.0 },
    { quarter: 'Q4 2023', stryker: 30.2, medtronic: 23.1, zimmer: 21.3, smith_nephew: 14.0, depuy: 11.4 },
    { quarter: 'Q1 2024', stryker: 30.8, medtronic: 22.9, zimmer: 21.0, smith_nephew: 13.8, depuy: 11.5 },
    { quarter: 'Q2 2024', stryker: 31.2, medtronic: 22.5, zimmer: 20.8, smith_nephew: 13.5, depuy: 12.0 },
    { quarter: 'Q3 2024', stryker: 31.5, medtronic: 22.2, zimmer: 20.5, smith_nephew: 13.3, depuy: 12.5 },
    { quarter: 'Q4 2024', stryker: 32.0, medtronic: 21.8, zimmer: 20.2, smith_nephew: 13.0, depuy: 13.0 },
  ];

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-white font-semibold text-sm">{title}</h3>
        {chartData.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-[#0057B8] font-mono text-sm font-bold">
              {chartData[chartData.length - 1]?.stryker?.toFixed(1)}%
            </span>
            <span className="text-white/30 text-xs">current</span>
          </div>
        )}
      </div>
      <p className="text-white/40 text-xs mb-6">
        Orthopedic device market share by manufacturer (% of revenue)
      </p>

      {loading ? (
        <div className="h-72 animate-pulse bg-white/5 rounded-xl" />
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              {COMPETITORS.map((comp) => (
                <linearGradient key={comp.key} id={`line-gradient-${comp.key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={comp.color} stopOpacity={0.8} />
                  <stop offset="100%" stopColor={comp.color} stopOpacity={0.2} />
                </linearGradient>
              ))}
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />

            <XAxis
              dataKey="quarter"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={false}
            />

            <YAxis
              domain={[0, 40]}
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={false}
              tickFormatter={(val) => `${val}%`}
            />

            <Tooltip content={<ShareTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)' }} />

            {COMPETITORS.map((comp) => (
              <Line
                key={comp.key}
                type="monotone"
                dataKey={comp.key}
                name={comp.name}
                stroke={comp.color}
                strokeWidth={comp.key === 'stryker' ? 3 : 2}
                dot={comp.key === 'stryker' ? { r: 4, fill: comp.color, stroke: '#0f172a', strokeWidth: 2 } : false}
                activeDot={{ r: 6, fill: comp.color, stroke: 'rgba(255,255,255,0.3)', strokeWidth: 2 }}
                hide={hiddenLines.has(comp.key)}
                strokeDasharray={comp.key === 'stryker' ? undefined : undefined}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}

      <InteractiveLegend competitors={COMPETITORS} hidden={hiddenLines} onToggle={toggleLine} />
    </motion.div>
  );
}
