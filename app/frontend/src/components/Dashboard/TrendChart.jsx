import React from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import GlassCard from '../shared/GlassCard';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/** Default trend data (24 months) */
const DEFAULT_DATA = [
  { month: 'Jan 24', revenue: 1450, margin: 62 },
  { month: 'Feb 24', revenue: 1520, margin: 63 },
  { month: 'Mar 24', revenue: 1480, margin: 61 },
  { month: 'Apr 24', revenue: 1600, margin: 64 },
  { month: 'May 24', revenue: 1650, margin: 63 },
  { month: 'Jun 24', revenue: 1580, margin: 62 },
  { month: 'Jul 24', revenue: 1700, margin: 65 },
  { month: 'Aug 24', revenue: 1750, margin: 64 },
  { month: 'Sep 24', revenue: 1680, margin: 63 },
  { month: 'Oct 24', revenue: 1820, margin: 66 },
  { month: 'Nov 24', revenue: 1900, margin: 67 },
  { month: 'Dec 24', revenue: 1850, margin: 65 },
  { month: 'Jan 25', revenue: 1920, margin: 66 },
  { month: 'Feb 25', revenue: 1980, margin: 67 },
  { month: 'Mar 25', revenue: 1950, margin: 66 },
  { month: 'Apr 25', revenue: 2050, margin: 68 },
  { month: 'May 25', revenue: 2100, margin: 67 },
  { month: 'Jun 25', revenue: 2020, margin: 66 },
  { month: 'Jul 25', revenue: 2150, margin: 69 },
  { month: 'Aug 25', revenue: 2200, margin: 68 },
  { month: 'Sep 25', revenue: 2180, margin: 67 },
  { month: 'Oct 25', revenue: 2300, margin: 70 },
  { month: 'Nov 25', revenue: 2380, margin: 71 },
  { month: 'Dec 25', revenue: 2350, margin: 69 },
];

/**
 * Custom glass-card styled tooltip for Recharts.
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
      <p className="text-white/50 text-xs mb-2 font-medium">{label}</p>
      {payload.map((entry, idx) => (
        <div key={idx} className="flex items-center gap-2 mb-1 last:mb-0">
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-white/60 text-xs">{entry.name}:</span>
          <span className="text-white font-mono text-xs font-bold">
            {entry.name === 'Revenue' ? `$${entry.value}M` : `${entry.value}%`}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * TrendChart - Recharts AreaChart showing revenue and margin trends over time.
 *
 * Props:
 *   data      - array of { month: string, revenue: number, margin: number }
 *   className - string, additional CSS classes
 */
export default function TrendChart({ data = DEFAULT_DATA, className = '' }) {
  return (
    <GlassCard className={className} animate={false}>
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-white font-semibold text-sm">Revenue & Margin Trends</h3>
        <div className="flex items-center gap-4 text-xs text-white/40">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-1 rounded-full" style={{ backgroundColor: '#0057B8' }} />
            <span>Revenue</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-1 rounded-full" style={{ backgroundColor: '#FFB81C' }} />
            <span>Margin %</span>
          </div>
        </div>
      </div>

      <motion.div
        className="w-full"
        style={{ height: 320 }}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.15 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <defs>
              {/* Revenue gradient */}
              <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0057B8" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#0057B8" stopOpacity={0} />
              </linearGradient>
              {/* Margin gradient */}
              <linearGradient id="marginGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FFB81C" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#FFB81C" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
              vertical={false}
            />

            <XAxis
              dataKey="month"
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={false}
              interval="preserveStartEnd"
            />

            <YAxis
              yAxisId="revenue"
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `$${v}`}
            />

            <YAxis
              yAxisId="margin"
              orientation="right"
              domain={[50, 80]}
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />

            <RechartsTooltip content={<CustomTooltip />} />

            <Area
              yAxisId="revenue"
              type="monotone"
              dataKey="revenue"
              name="Revenue"
              stroke="#0057B8"
              strokeWidth={2}
              fill="url(#revenueGradient)"
              dot={false}
              activeDot={{
                r: 5,
                fill: '#0057B8',
                stroke: 'rgba(255,255,255,0.3)',
                strokeWidth: 2,
              }}
              animationDuration={1500}
            />

            <Area
              yAxisId="margin"
              type="monotone"
              dataKey="margin"
              name="Margin"
              stroke="#FFB81C"
              strokeWidth={2}
              fill="url(#marginGradient)"
              dot={false}
              activeDot={{
                r: 5,
                fill: '#FFB81C',
                stroke: 'rgba(255,255,255,0.3)',
                strokeWidth: 2,
              }}
              animationDuration={1500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>
    </GlassCard>
  );
}
