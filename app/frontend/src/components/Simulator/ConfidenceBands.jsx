import React from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * Custom tooltip for confidence bands chart.
 */
function BandTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
      <p className="text-white/60 text-xs mb-2">Price Change: {data.priceChange > 0 ? '+' : ''}{data.priceChange}%</p>
      <div className="space-y-1">
        <div className="flex items-center justify-between gap-4">
          <span className="text-white/40 text-xs">Upper (95%)</span>
          <span className="text-[#10b981] text-xs font-mono">{data.upper?.toFixed(1)}%</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-white/40 text-xs">Prediction</span>
          <span className="text-[#0057B8] text-xs font-mono font-bold">{data.prediction?.toFixed(1)}%</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-white/40 text-xs">Lower (95%)</span>
          <span className="text-[#f43f5e] text-xs font-mono">{data.lower?.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

/**
 * ConfidenceBands - Visualization of confidence intervals around ML predictions.
 *
 * Props:
 *   data       - array of { priceChange, prediction, upper, lower }
 *   loading    - boolean
 *   title      - string (default: 'Prediction Confidence Intervals')
 *   currentVal - number, current price change value to highlight
 */
export default function ConfidenceBands({
  data = null,
  loading = false,
  title = 'Prediction Confidence Intervals',
  currentVal = 0,
}) {
  // Default demo data: volume change predictions across price changes
  const chartData = data || [
    { priceChange: -30, prediction: 18.5, upper: 25.2, lower: 12.1 },
    { priceChange: -25, prediction: 15.8, upper: 21.6, lower: 10.3 },
    { priceChange: -20, prediction: 12.4, upper: 17.8, lower: 7.5 },
    { priceChange: -15, prediction: 9.1, upper: 13.5, lower: 5.2 },
    { priceChange: -10, prediction: 6.2, upper: 9.8, lower: 3.1 },
    { priceChange: -5, prediction: 3.1, upper: 5.4, lower: 1.2 },
    { priceChange: 0, prediction: 0, upper: 1.8, lower: -1.5 },
    { priceChange: 5, prediction: -2.8, upper: -0.5, lower: -5.2 },
    { priceChange: 10, prediction: -5.9, upper: -2.8, lower: -9.4 },
    { priceChange: 15, prediction: -9.3, upper: -5.6, lower: -13.8 },
    { priceChange: 20, prediction: -13.1, upper: -8.9, lower: -18.2 },
    { priceChange: 25, prediction: -17.2, upper: -12.1, lower: -23.5 },
    { priceChange: 30, prediction: -21.8, upper: -15.4, lower: -29.1 },
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
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-[#0057B8] rounded" />
            <span className="text-white/40 text-[10px]">Point Estimate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-[#0057B8]/20 border border-[#0057B8]/30" />
            <span className="text-white/40 text-[10px]">95% CI</span>
          </div>
        </div>
      </div>
      <p className="text-white/40 text-xs mb-6">Predicted volume change across price adjustments</p>

      {loading ? (
        <div className="h-64 flex items-center justify-center">
          <div className="animate-pulse space-y-4 w-full">
            <div className="h-48 bg-white/5 rounded-xl" />
          </div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0057B8" stopOpacity={0.3} />
                <stop offset="50%" stopColor="#0057B8" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#0057B8" stopOpacity={0.3} />
              </linearGradient>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="50%" stopColor="#0057B8" />
                <stop offset="100%" stopColor="#f43f5e" />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />

            <XAxis
              dataKey="priceChange"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickFormatter={(val) => `${val > 0 ? '+' : ''}${val}%`}
              label={{
                value: 'Price Change %',
                position: 'insideBottom',
                offset: -5,
                fill: 'rgba(255,255,255,0.3)',
                fontSize: 10,
              }}
            />

            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              tickFormatter={(val) => `${val > 0 ? '+' : ''}${val}%`}
              label={{
                value: 'Volume Change %',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                fill: 'rgba(255,255,255,0.3)',
                fontSize: 10,
              }}
            />

            <Tooltip content={<BandTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)' }} />

            {/* Upper confidence band */}
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="url(#bandGradient)"
              fillOpacity={1}
            />

            {/* Lower confidence band */}
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="#0f172a"
              fillOpacity={1}
            />

            {/* Point estimate line */}
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#0057B8"
              strokeWidth={2.5}
              dot={false}
              activeDot={{
                r: 5,
                fill: '#0057B8',
                stroke: 'rgba(0, 87, 184, 0.4)',
                strokeWidth: 8,
              }}
            />

            {/* Zero reference line */}
            <Line
              type="monotone"
              dataKey={() => 0}
              stroke="rgba(255,255,255,0.15)"
              strokeDasharray="4 4"
              strokeWidth={1}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </motion.div>
  );
}
