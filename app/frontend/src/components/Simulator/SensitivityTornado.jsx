import React from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * Custom tooltip for the tornado chart.
 */
function TornadoTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;
  return (
    <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
      <p className="text-white text-sm font-medium mb-1">{data.factor}</p>
      <div className="space-y-1">
        {data.negative !== 0 && (
          <p className="text-[#f43f5e] text-xs font-mono">
            Negative: {data.negative.toFixed(1)}%
          </p>
        )}
        {data.positive !== 0 && (
          <p className="text-[#10b981] text-xs font-mono">
            Positive: +{data.positive.toFixed(1)}%
          </p>
        )}
      </div>
      <p className="text-white/40 text-[10px] mt-1">SHAP importance: {data.importance?.toFixed(3) ?? 'N/A'}</p>
    </div>
  );
}

/**
 * SensitivityTornado - Horizontal tornado/butterfly chart showing top sensitivity factors.
 *
 * Props:
 *   data    - array of { factor, negative, positive, importance }
 *   loading - boolean
 *   title   - string (default: 'Sensitivity Analysis (SHAP)')
 */
export default function SensitivityTornado({
  data = null,
  loading = false,
  title = 'Sensitivity Analysis (SHAP)',
}) {
  // Default demo data
  const chartData = data || [
    { factor: 'Competitor Price Gap', negative: -8.2, positive: 12.5, importance: 0.342 },
    { factor: 'Contract Volume', negative: -6.1, positive: 9.3, importance: 0.281 },
    { factor: 'GPO Tier Level', negative: -5.4, positive: 7.8, importance: 0.215 },
    { factor: 'Market Share Trend', negative: -4.2, positive: 6.1, importance: 0.178 },
    { factor: 'Titanium Cost Index', negative: -3.8, positive: 2.4, importance: 0.134 },
    { factor: 'Hospital CapEx Budget', negative: -2.9, positive: 4.5, importance: 0.112 },
    { factor: 'Patent Runway (Years)', negative: -1.5, positive: 5.2, importance: 0.098 },
    { factor: 'Tariff Rate', negative: -3.1, positive: 1.2, importance: 0.087 },
  ];

  // Sort by total absolute impact
  const sortedData = [...chartData].sort(
    (a, b) => (Math.abs(b.negative) + Math.abs(b.positive)) - (Math.abs(a.negative) + Math.abs(a.positive))
  );

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <h3 className="text-white font-semibold text-sm mb-1">{title}</h3>
      <p className="text-white/40 text-xs mb-6">Impact of each factor on predicted volume change</p>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="animate-pulse flex items-center gap-3">
              <div className="w-28 h-4 bg-white/10 rounded" />
              <div className="flex-1 h-6 bg-white/10 rounded" />
            </div>
          ))}
        </div>
      ) : (
        <div className="w-full">
          {sortedData.map((item, index) => (
            <motion.div
              key={item.factor}
              className="flex items-center gap-3 mb-3"
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ ...springTransition, delay: index * 0.06 }}
            >
              {/* Factor label */}
              <div className="w-36 shrink-0 text-right">
                <span className="text-white/70 text-xs">{item.factor}</span>
              </div>

              {/* Bar container */}
              <div className="flex-1 flex items-center h-7 relative">
                {/* Center line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/20" />

                {/* Negative bar (extends left from center) */}
                <div className="flex-1 flex justify-end pr-0">
                  <motion.div
                    className="h-6 rounded-l-md"
                    style={{ backgroundColor: '#f43f5e' }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.abs(item.negative) * 4}%` }}
                    transition={{ ...springTransition, delay: index * 0.06 + 0.1 }}
                  />
                </div>

                {/* Positive bar (extends right from center) */}
                <div className="flex-1 flex justify-start pl-0">
                  <motion.div
                    className="h-6 rounded-r-md"
                    style={{ backgroundColor: '#10b981' }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.abs(item.positive) * 4}%` }}
                    transition={{ ...springTransition, delay: index * 0.06 + 0.1 }}
                  />
                </div>
              </div>

              {/* Values */}
              <div className="w-24 shrink-0 flex items-center gap-1">
                <span className="text-[#f43f5e] text-xs font-mono">{item.negative.toFixed(1)}</span>
                <span className="text-white/20">/</span>
                <span className="text-[#10b981] text-xs font-mono">+{item.positive.toFixed(1)}</span>
              </div>
            </motion.div>
          ))}

          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-white/10">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#f43f5e' }} />
              <span className="text-white/50 text-xs">Negative Impact</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#10b981' }} />
              <span className="text-white/50 text-xs">Positive Impact</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
