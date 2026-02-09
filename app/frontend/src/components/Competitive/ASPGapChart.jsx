import React, { useState } from 'react';
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
  LabelList,
} from 'recharts';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const CATEGORIES = [
  'Joint Replacement',
  'Trauma & Extremities',
  'Spine',
  'Instruments',
  'Medical/Surgical',
];

/**
 * Custom tooltip for ASP gap chart.
 */
function GapTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
      <p className="text-white text-sm font-medium mb-2">{label}</p>
      <div className="space-y-1.5">
        {payload.map((entry) => (
          <div key={entry.name} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: entry.fill || entry.color }} />
              <span className="text-white/60 text-xs">{entry.name}</span>
            </div>
            <span className="text-white font-mono text-xs">${entry.value?.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * ASPGapChart - Recharts grouped BarChart comparing Stryker ASP vs competitors.
 *
 * Props:
 *   data     - object keyed by category, each with competitor comparisons
 *   loading  - boolean
 *   title    - string (default: 'ASP Competitive Gap Analysis')
 */
export default function ASPGapChart({
  data = null,
  loading = false,
  title = 'ASP Competitive Gap Analysis',
}) {
  const [selectedCategory, setSelectedCategory] = useState('Joint Replacement');

  // Default demo data by category
  const demoData = {
    'Joint Replacement': [
      { product: 'Hip System', stryker: 45200, medtronic: 42800, zimmer: 44100, smith_nephew: 41500 },
      { product: 'Knee System', stryker: 12800, medtronic: 11900, zimmer: 13200, smith_nephew: 11200 },
      { product: 'Shoulder System', stryker: 18500, medtronic: 17200, zimmer: 19100, smith_nephew: 16800 },
    ],
    'Trauma & Extremities': [
      { product: 'Tibial Nail', stryker: 3400, medtronic: 3100, zimmer: 3250, smith_nephew: 2900 },
      { product: 'Plating System', stryker: 2800, medtronic: 2600, zimmer: 2750, smith_nephew: 2400 },
    ],
    'Spine': [
      { product: 'Spinal Implant', stryker: 8900, medtronic: 9200, zimmer: 8100, smith_nephew: 7800 },
      { product: 'Disc Replacement', stryker: 15200, medtronic: 16100, zimmer: 14500, smith_nephew: 13800 },
    ],
    'Instruments': [
      { product: 'Power Tools', stryker: 22100, medtronic: 19800, zimmer: 20500, smith_nephew: 18200 },
      { product: 'Waste Mgmt', stryker: 18500, medtronic: 16200, zimmer: 17100, smith_nephew: 15500 },
    ],
    'Medical/Surgical': [
      { product: 'Bed System', stryker: 35600, medtronic: 32100, zimmer: 33800, smith_nephew: 30500 },
      { product: 'Positioning', stryker: 4200, medtronic: 3800, zimmer: 3950, smith_nephew: 3600 },
    ],
  };

  const chartData = data?.[selectedCategory] || demoData[selectedCategory] || [];

  const competitors = [
    { key: 'stryker', name: 'Stryker', color: '#0057B8' },
    { key: 'medtronic', name: 'Medtronic', color: '#6b7280' },
    { key: 'zimmer', name: 'Zimmer Biomet', color: '#9ca3af' },
    { key: 'smith_nephew', name: 'Smith & Nephew', color: '#d1d5db' },
  ];

  // Calculate % gaps
  const dataWithGaps = chartData.map((item) => {
    const gaps = {};
    competitors.forEach((comp) => {
      if (comp.key !== 'stryker' && item.stryker && item[comp.key]) {
        gaps[`${comp.key}_gap`] = (((item.stryker - item[comp.key]) / item[comp.key]) * 100).toFixed(1);
      }
    });
    return { ...item, ...gaps };
  });

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-white font-semibold text-sm">{title}</h3>
      </div>
      <p className="text-white/40 text-xs mb-4">
        Stryker average selling price vs key competitors by category
      </p>

      {/* Category Selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {CATEGORIES.map((cat) => (
          <motion.button
            key={cat}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              selectedCategory === cat
                ? 'bg-[#0057B8]/20 text-[#0057B8] border border-[#0057B8]/40'
                : 'bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:text-white/80'
            }`}
            onClick={() => setSelectedCategory(cat)}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={springTransition}
          >
            {cat}
          </motion.button>
        ))}
      </div>

      {loading ? (
        <div className="h-72 animate-pulse bg-white/5 rounded-xl" />
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={dataWithGaps} margin={{ top: 20, right: 20, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="product"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={false}
              tickFormatter={(val) => `$${(val / 1000).toFixed(0)}K`}
            />
            <Tooltip content={<GapTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />

            {competitors.map((comp) => (
              <Bar
                key={comp.key}
                dataKey={comp.key}
                name={comp.name}
                fill={comp.color}
                radius={[4, 4, 0, 0]}
                maxBarSize={40}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      )}

      {/* Gap Labels */}
      {!loading && dataWithGaps.length > 0 && (
        <motion.div
          className="mt-4 pt-4 border-t border-white/10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ ...springTransition, delay: 0.3 }}
        >
          <p className="text-white/40 text-xs mb-3">Stryker Premium/Discount vs Competitors</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {dataWithGaps.map((item) => (
              <div key={item.product} className="bg-white/5 rounded-lg p-3">
                <p className="text-white text-xs font-medium mb-2">{item.product}</p>
                <div className="space-y-1">
                  {competitors.filter((c) => c.key !== 'stryker').map((comp) => {
                    const gap = item[`${comp.key}_gap`];
                    if (!gap) return null;
                    const gapNum = parseFloat(gap);
                    return (
                      <div key={comp.key} className="flex items-center justify-between text-xs">
                        <span className="text-white/40">vs {comp.name}</span>
                        <span
                          className="font-mono font-medium"
                          style={{ color: gapNum > 0 ? '#FFB81C' : '#10b981' }}
                        >
                          {gapNum > 0 ? '+' : ''}{gap}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-4">
        {competitors.map((comp) => (
          <div key={comp.key} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: comp.color }} />
            <span className="text-white/50 text-xs">{comp.name}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
