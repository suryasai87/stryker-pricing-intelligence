import React from 'react';
import { motion } from 'framer-motion';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/solid';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import AnimatedNumber from '../shared/AnimatedNumber';
import GlassCard from '../shared/GlassCard';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * KPICard - Animated KPI card with icon, animated value, delta, and mini sparkline.
 *
 * Props:
 *   icon       - React component (Heroicon), the card icon
 *   title      - string, the KPI label
 *   value      - number, the KPI numeric value
 *   format     - "currency" | "percent" | "number" | "compact" (default: "number")
 *   delta      - number, percentage change (positive = up/green, negative = down/red)
 *   deltaLabel - string, label for the delta (e.g., "vs last quarter")
 *   sparkline  - array of { value: number }, data for the mini sparkline
 *   index      - number, used for staggered entrance delay
 *   onClick    - function, optional click handler
 */
export default function KPICard({
  icon: Icon,
  title,
  value = 0,
  format = 'number',
  delta = 0,
  deltaLabel = 'vs last period',
  sparkline = [],
  index = 0,
  onClick,
}) {
  const isPositive = delta >= 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...springTransition, delay: index * 0.08 }}
    >
      <GlassCard
        className="relative overflow-hidden cursor-default"
        onClick={onClick}
        animate
      >
        {/* Header row: icon + title */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {Icon && (
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ backgroundColor: 'rgba(0, 87, 184, 0.15)' }}
              >
                <Icon className="w-5 h-5" style={{ color: '#0057B8' }} />
              </div>
            )}
            <span className="text-white/50 text-sm font-medium">{title}</span>
          </div>
        </div>

        {/* Value */}
        <div className="mb-2">
          <AnimatedNumber
            value={value}
            format={format}
            className="text-2xl font-bold text-white"
          />
        </div>

        {/* Delta indicator */}
        <div className="flex items-center gap-2 mb-3">
          <div
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${
              isPositive
                ? 'bg-emerald-500/15 text-emerald-400'
                : 'bg-rose-500/15 text-rose-400'
            }`}
          >
            {isPositive ? (
              <ArrowTrendingUpIcon className="w-3.5 h-3.5" />
            ) : (
              <ArrowTrendingDownIcon className="w-3.5 h-3.5" />
            )}
            <span className="font-mono">
              {isPositive ? '+' : ''}
              {delta.toFixed(1)}%
            </span>
          </div>
          <span className="text-white/30 text-xs">{deltaLabel}</span>
        </div>

        {/* Mini Sparkline */}
        {sparkline.length > 0 && (
          <div className="h-12 -mx-2 -mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sparkline}>
                <defs>
                  <linearGradient id={`sparkGrad-${index}`} x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor={isPositive ? '#10b981' : '#f43f5e'}
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="100%"
                      stopColor={isPositive ? '#10b981' : '#f43f5e'}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={isPositive ? '#10b981' : '#f43f5e'}
                  strokeWidth={1.5}
                  fill={`url(#sparkGrad-${index})`}
                  dot={false}
                  animationDuration={1200}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </GlassCard>
    </motion.div>
  );
}
