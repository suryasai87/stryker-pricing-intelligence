import React from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * Sparkline - Minimal trend line for inline display.
 */
function Sparkline({ data, dataKey, color, height = 40 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            return (
              <div className="bg-slate-800/95 border border-white/10 rounded-lg px-2 py-1 text-xs">
                <span className="text-white font-mono">{payload[0]?.value}%</span>
              </div>
            );
          }}
          cursor={false}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 3, fill: color }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/**
 * TariffCard - Individual tariff material card.
 */
function TariffCard({ material, rate, trend, impact, trendData, index }) {
  const isIncreasing = trend === 'up';
  const trendColor = isIncreasing ? '#f43f5e' : '#10b981';

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 relative overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...springTransition, delay: index * 0.08 }}
      whileHover={{ scale: 1.01, borderColor: 'rgba(255,255,255,0.2)' }}
    >
      {/* Background accent */}
      <div
        className="absolute -top-8 -right-8 w-24 h-24 rounded-full blur-3xl pointer-events-none"
        style={{ backgroundColor: `${trendColor}15` }}
      />

      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-white font-medium text-sm">{material}</h4>
          <p className="text-white/40 text-xs mt-0.5">Current tariff rate</p>
        </div>
        <motion.div
          className="flex items-center gap-1 px-2 py-1 rounded-lg"
          style={{
            backgroundColor: `${trendColor}15`,
            border: `1px solid ${trendColor}30`,
          }}
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={springTransition}
        >
          <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none">
            <path
              d={isIncreasing ? 'M6 9V3M6 3L3 6M6 3L9 6' : 'M6 3V9M6 9L3 6M6 9L9 6'}
              stroke={trendColor}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-xs font-mono" style={{ color: trendColor }}>
            {isIncreasing ? '+' : '-'}{Math.abs(impact)}%
          </span>
        </motion.div>
      </div>

      {/* Rate Display */}
      <div className="mb-4">
        <span className="font-mono text-3xl font-bold text-white">{rate}%</span>
      </div>

      {/* Sparkline */}
      <div className="mb-3">
        <Sparkline data={trendData} dataKey="value" color={trendColor} />
      </div>

      {/* Impact indicator */}
      <div className="flex items-center justify-between pt-3 border-t border-white/10">
        <span className="text-white/40 text-xs">COGS Impact</span>
        <span className="font-mono text-sm font-medium" style={{ color: trendColor }}>
          {isIncreasing ? '+' : '-'}${Math.abs(impact * 120).toLocaleString()}K/yr
        </span>
      </div>
    </motion.div>
  );
}

/**
 * TariffDashboard - Cards showing current tariff rates for key materials.
 *
 * Props:
 *   data    - array of tariff data objects
 *   loading - boolean
 *   title   - string (default: 'Tariff Monitor')
 */
export default function TariffDashboard({
  data = null,
  loading = false,
  title = 'Tariff Monitor',
}) {
  // Generate trend sparkline data
  const generateTrend = (base, direction, points = 12) => {
    const result = [];
    let val = base - (direction === 'up' ? 3 : -3);
    for (let i = 0; i < points; i++) {
      val += direction === 'up'
        ? (Math.random() * 0.8 - 0.2)
        : -(Math.random() * 0.8 - 0.2);
      result.push({ month: i, value: parseFloat(val.toFixed(1)) });
    }
    return result;
  };

  const tariffs = data || [
    {
      material: 'Steel (Section 232)',
      rate: 25.0,
      trend: 'up',
      impact: 2.3,
      trendData: generateTrend(22, 'up'),
    },
    {
      material: 'Titanium (HTS 8108)',
      rate: 15.0,
      trend: 'up',
      impact: 1.8,
      trendData: generateTrend(12, 'up'),
    },
    {
      material: 'Cobalt-Chrome Alloy',
      rate: 7.5,
      trend: 'down',
      impact: -0.5,
      trendData: generateTrend(9, 'down'),
    },
    {
      material: 'UHMWPE Polymer',
      rate: 6.5,
      trend: 'up',
      impact: 0.8,
      trendData: generateTrend(5.5, 'up'),
    },
  ];

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white font-semibold text-sm">{title}</h3>
          <p className="text-white/40 text-xs mt-0.5">Import tariff rates on key raw materials</p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 rounded-lg border border-white/10">
          <div className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse" />
          <span className="text-white/50 text-xs">Live rates</span>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="animate-pulse bg-white/5 rounded-2xl h-48" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {tariffs.map((tariff, index) => (
            <TariffCard key={tariff.material} {...tariff} index={index} />
          ))}
        </div>
      )}
    </div>
  );
}
