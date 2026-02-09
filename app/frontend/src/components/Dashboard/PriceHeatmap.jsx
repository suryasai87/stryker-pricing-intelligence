import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../shared/GlassCard';
import Tooltip from '../shared/Tooltip';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const DEFAULT_CATEGORIES = [
  'Hip Implants',
  'Knee Implants',
  'Spine Devices',
  'Trauma Fixation',
  'Surgical Instruments',
  'Endoscopy',
  'Power Tools',
  'Navigation Systems',
];

/**
 * Generate sample heatmap data with realistic YoY price change percentages.
 */
function generateDefaultData(categories) {
  return categories.map((category) => ({
    category,
    values: MONTHS.map(() => parseFloat((Math.random() * 10 - 3).toFixed(1))),
  }));
}

/**
 * Map a value to a color on the green-to-red scale.
 * Positive = green (price increase), Negative = red (price decrease).
 */
function valueToColor(value) {
  const clamped = Math.max(-8, Math.min(8, value));
  const t = (clamped + 8) / 16; // normalize to 0..1

  if (t >= 0.5) {
    // Green side
    const intensity = (t - 0.5) * 2; // 0..1
    const r = Math.round(16 * (1 - intensity));
    const g = Math.round(185 * intensity + 60 * (1 - intensity));
    const b = Math.round(129 * intensity + 40 * (1 - intensity));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Red side
    const intensity = (0.5 - t) * 2; // 0..1
    const r = Math.round(244 * intensity + 60 * (1 - intensity));
    const g = Math.round(63 * intensity + 60 * (1 - intensity));
    const b = Math.round(94 * intensity + 60 * (1 - intensity));
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/**
 * PriceHeatmap - YoY price change heatmap grid.
 *
 * Props:
 *   data       - array of { category: string, values: number[] (12 months) }
 *   categories - string[], category names (used if data not provided)
 *   className  - string, additional CSS classes
 */
export default function PriceHeatmap({
  data,
  categories = DEFAULT_CATEGORIES,
  className = '',
}) {
  const heatmapData = useMemo(() => {
    if (data && data.length > 0) return data;
    return generateDefaultData(categories);
  }, [data, categories]);

  return (
    <GlassCard className={className} animate={false}>
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-white font-semibold text-sm">YoY Price Change Heatmap</h3>
        <div className="flex items-center gap-3 text-xs text-white/40">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#f43f5e' }} />
            <span>Decrease</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#10b981' }} />
            <span>Increase</span>
          </div>
        </div>
      </div>

      {/* Grid */}
      <div className="overflow-x-auto">
        <div className="min-w-[700px]">
          {/* Month header row */}
          <div className="grid gap-1 mb-1" style={{ gridTemplateColumns: '140px repeat(12, 1fr)' }}>
            <div /> {/* Empty corner cell */}
            {MONTHS.map((month) => (
              <div
                key={month}
                className="text-center text-white/40 text-xs font-medium py-1"
              >
                {month}
              </div>
            ))}
          </div>

          {/* Data rows */}
          {heatmapData.map((row, rowIdx) => (
            <motion.div
              key={row.category}
              className="grid gap-1 mb-1"
              style={{ gridTemplateColumns: '140px repeat(12, 1fr)' }}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ ...springTransition, delay: rowIdx * 0.04 }}
            >
              {/* Category label */}
              <div className="text-white/60 text-xs font-medium truncate py-2 pr-2 flex items-center">
                {row.category}
              </div>

              {/* Value cells */}
              {row.values.map((value, colIdx) => (
                <Tooltip
                  key={`${row.category}-${colIdx}`}
                  content={
                    <div>
                      <p className="font-semibold">{row.category}</p>
                      <p>
                        {MONTHS[colIdx]}:{' '}
                        <span className="font-mono">
                          {value >= 0 ? '+' : ''}
                          {value}%
                        </span>
                      </p>
                    </div>
                  }
                  position="top"
                >
                  <motion.div
                    className="rounded-md w-full aspect-square flex items-center justify-center cursor-pointer"
                    style={{ backgroundColor: valueToColor(value) }}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 0.85, scale: 1 }}
                    transition={{
                      ...springTransition,
                      delay: rowIdx * 0.03 + colIdx * 0.015,
                    }}
                    whileHover={{ opacity: 1, scale: 1.15 }}
                  >
                    <span className="text-white font-mono text-[9px] font-bold drop-shadow-md">
                      {value > 0 ? '+' : ''}
                      {value}
                    </span>
                  </motion.div>
                </Tooltip>
              ))}
            </motion.div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}
