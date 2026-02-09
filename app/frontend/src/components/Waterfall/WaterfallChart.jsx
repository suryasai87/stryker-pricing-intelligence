import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * WaterfallChart - Animated waterfall bar chart showing price decomposition.
 *
 * Props:
 *   data    - array of { label, value, type: 'start'|'deduction'|'final' }
 *   loading - boolean
 *   title   - string (default: 'Price Waterfall Analysis')
 *   height  - number (default: 400)
 */
export default function WaterfallChart({
  data = null,
  loading = false,
  title = 'Price Waterfall Analysis',
  height = 400,
}) {
  // Default demo data
  const waterfallData = data || [
    { label: 'List Price', value: 45200, type: 'start' },
    { label: 'Contract Discount', value: -5420, type: 'deduction' },
    { label: 'GPO Rebate', value: -3164, type: 'deduction' },
    { label: 'Volume Bonus', value: -1808, type: 'deduction' },
    { label: 'Freight/Shipping', value: -904, type: 'deduction' },
    { label: 'Cash Discount', value: -452, type: 'deduction' },
    { label: 'Pocket Price', value: 33452, type: 'final' },
  ];

  // Calculate cumulative positions
  const bars = useMemo(() => {
    let running = 0;
    return waterfallData.map((item, index) => {
      if (item.type === 'start') {
        const bar = { ...item, y: 0, height: item.value, bottom: 0 };
        running = item.value;
        return bar;
      }
      if (item.type === 'final') {
        return { ...item, y: 0, height: item.value, bottom: 0 };
      }
      // Deduction
      const newRunning = running + item.value;
      const bar = {
        ...item,
        y: Math.min(running, newRunning),
        height: Math.abs(item.value),
        bottom: newRunning,
      };
      running = newRunning;
      return bar;
    });
  }, [waterfallData]);

  const maxValue = Math.max(...bars.map((b) => b.type === 'start' ? b.value : b.y + b.height));
  const chartPadding = { top: 40, right: 30, bottom: 80, left: 30 };
  const chartWidth = 700;
  const chartHeight = height;
  const plotWidth = chartWidth - chartPadding.left - chartPadding.right;
  const plotHeight = chartHeight - chartPadding.top - chartPadding.bottom;
  const barWidth = Math.min(80, (plotWidth / bars.length) * 0.6);
  const barGap = (plotWidth - barWidth * bars.length) / (bars.length + 1);

  const scaleY = (val) => chartPadding.top + (1 - val / maxValue) * plotHeight;
  const getBarX = (i) => chartPadding.left + barGap * (i + 1) + barWidth * i;

  const getBarColor = (type) => {
    if (type === 'start') return '#0057B8';
    if (type === 'final') return '#10b981';
    return '#f43f5e';
  };

  const formatCurrency = (val) => {
    const abs = Math.abs(val);
    if (abs >= 1000) return `$${(abs / 1000).toFixed(1)}K`;
    return `$${abs.toFixed(0)}`;
  };

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <h3 className="text-white font-semibold text-sm mb-1">{title}</h3>
      <p className="text-white/40 text-xs mb-6">
        Decomposition from list price to pocket price
      </p>

      {loading ? (
        <div className="animate-pulse" style={{ height }}>
          <div className="h-full bg-white/5 rounded-xl" />
        </div>
      ) : (
        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            className="w-full"
            style={{ minWidth: 500 }}
          >
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
              <g key={pct}>
                <line
                  x1={chartPadding.left}
                  y1={scaleY(maxValue * pct)}
                  x2={chartWidth - chartPadding.right}
                  y2={scaleY(maxValue * pct)}
                  stroke="rgba(255,255,255,0.05)"
                  strokeDasharray="4 4"
                />
                <text
                  x={chartPadding.left - 5}
                  y={scaleY(maxValue * pct)}
                  textAnchor="end"
                  alignmentBaseline="middle"
                  fill="rgba(255,255,255,0.25)"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  {formatCurrency(maxValue * pct)}
                </text>
              </g>
            ))}

            {/* Connecting lines between bars */}
            {bars.map((bar, i) => {
              if (i === 0 || bar.type === 'final') return null;
              const prevBar = bars[i - 1];
              const prevBottom = prevBar.type === 'start'
                ? prevBar.value
                : prevBar.bottom;
              const prevX = getBarX(i - 1) + barWidth;
              const currX = getBarX(i);
              const yPos = scaleY(bar.y + bar.height);

              return (
                <motion.line
                  key={`line-${i}`}
                  x1={prevX}
                  y1={scaleY(prevBottom)}
                  x2={currX}
                  y2={scaleY(prevBottom)}
                  stroke="rgba(255,255,255,0.15)"
                  strokeDasharray="4 2"
                  strokeWidth="1"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ ...springTransition, delay: i * 0.1 }}
                />
              );
            })}

            {/* Bars */}
            {bars.map((bar, i) => {
              const x = getBarX(i);
              const barTop = bar.type === 'deduction' ? scaleY(bar.y + bar.height) : scaleY(bar.value);
              const barH = (bar.height / maxValue) * plotHeight;
              const color = getBarColor(bar.type);

              return (
                <g key={`bar-${i}`}>
                  {/* Bar */}
                  <motion.rect
                    x={x}
                    y={bar.type === 'deduction' ? scaleY(bar.y + bar.height) - barH + barH : barTop}
                    width={barWidth}
                    rx={4}
                    fill={color}
                    fillOpacity={0.85}
                    initial={{ height: 0, y: scaleY(0) }}
                    animate={{
                      height: barH,
                      y: bar.type === 'deduction' ? scaleY(bar.y + bar.height) : barTop,
                    }}
                    transition={{ ...springTransition, delay: i * 0.12 }}
                  />

                  {/* Glow effect */}
                  <motion.rect
                    x={x}
                    y={bar.type === 'deduction' ? scaleY(bar.y + bar.height) : barTop}
                    width={barWidth}
                    rx={4}
                    fill={color}
                    fillOpacity={0.15}
                    filter="url(#glow)"
                    initial={{ height: 0 }}
                    animate={{ height: barH }}
                    transition={{ ...springTransition, delay: i * 0.12 }}
                  />

                  {/* Value label */}
                  <motion.text
                    x={x + barWidth / 2}
                    y={(bar.type === 'deduction' ? scaleY(bar.y + bar.height) : barTop) - 8}
                    textAnchor="middle"
                    fill="white"
                    fontSize="11"
                    fontFamily="monospace"
                    fontWeight="600"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.12 + 0.3 }}
                  >
                    {bar.type === 'deduction' ? '-' : ''}{formatCurrency(bar.value)}
                  </motion.text>

                  {/* Label */}
                  <motion.text
                    x={x + barWidth / 2}
                    y={chartHeight - chartPadding.bottom + 20}
                    textAnchor="middle"
                    fill="rgba(255,255,255,0.5)"
                    fontSize="10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.12 + 0.2 }}
                  >
                    {bar.label}
                  </motion.text>

                  {/* Percentage label for deductions */}
                  {bar.type === 'deduction' && (
                    <motion.text
                      x={x + barWidth / 2}
                      y={chartHeight - chartPadding.bottom + 35}
                      textAnchor="middle"
                      fill="rgba(244, 63, 94, 0.6)"
                      fontSize="9"
                      fontFamily="monospace"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.12 + 0.4 }}
                    >
                      {((bar.value / waterfallData[0].value) * 100).toFixed(1)}%
                    </motion.text>
                  )}
                </g>
              );
            })}

            {/* SVG filter for glow */}
            <defs>
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
          </svg>
        </div>
      )}

      {/* Summary */}
      {!loading && (
        <motion.div
          className="flex items-center justify-center gap-8 mt-4 pt-4 border-t border-white/10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ ...springTransition, delay: 0.8 }}
        >
          <div className="text-center">
            <p className="text-white/40 text-xs">List Price</p>
            <p className="text-white font-mono text-lg font-bold">
              ${waterfallData[0]?.value?.toLocaleString()}
            </p>
          </div>
          <div className="text-2xl text-white/20">&rarr;</div>
          <div className="text-center">
            <p className="text-white/40 text-xs">Total Leakage</p>
            <p className="text-[#f43f5e] font-mono text-lg font-bold">
              -${Math.abs(waterfallData[0]?.value - waterfallData[waterfallData.length - 1]?.value).toLocaleString()}
            </p>
          </div>
          <div className="text-2xl text-white/20">&rarr;</div>
          <div className="text-center">
            <p className="text-white/40 text-xs">Pocket Price</p>
            <p className="text-[#10b981] font-mono text-lg font-bold">
              ${waterfallData[waterfallData.length - 1]?.value?.toLocaleString()}
            </p>
          </div>
          <div className="text-center">
            <p className="text-white/40 text-xs">Realization %</p>
            <p className="text-[#FFB81C] font-mono text-lg font-bold">
              {((waterfallData[waterfallData.length - 1]?.value / waterfallData[0]?.value) * 100).toFixed(1)}%
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
