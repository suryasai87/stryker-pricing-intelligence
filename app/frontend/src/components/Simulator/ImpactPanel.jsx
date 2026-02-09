import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence, useSpring, useMotionValue } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * AnimatedNumber - Smoothly animates between numeric values.
 */
function AnimatedNumber({ value, prefix = '', suffix = '', decimals = 1, className = '' }) {
  const [displayValue, setDisplayValue] = useState(value);
  const motionVal = useMotionValue(value);
  const spring = useSpring(motionVal, { stiffness: 300, damping: 30 });

  useEffect(() => {
    motionVal.set(value);
  }, [value, motionVal]);

  useEffect(() => {
    const unsubscribe = spring.on('change', (latest) => {
      setDisplayValue(latest);
    });
    return unsubscribe;
  }, [spring]);

  const formatted = typeof value === 'number'
    ? `${prefix}${Math.abs(displayValue).toFixed(decimals)}${suffix}`
    : `${prefix}${value}${suffix}`;

  const isNeg = typeof value === 'number' && displayValue < 0;

  return (
    <span className={`font-mono ${className}`}>
      {isNeg ? '-' : ''}{formatted}
    </span>
  );
}

/**
 * ShimmerCard - Loading placeholder with shimmer effect.
 */
function ShimmerCard() {
  return (
    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 animate-pulse">
      <div className="h-3 w-24 bg-white/10 rounded mb-3" />
      <div className="h-8 w-32 bg-white/10 rounded mb-2" />
      <div className="h-2 w-16 bg-white/10 rounded" />
    </div>
  );
}

/**
 * MetricCard - Individual impact metric display.
 */
function MetricCard({ label, value, prefix, suffix, decimals, icon, positive, loading, index }) {
  const isPositive = positive !== undefined ? positive : value >= 0;
  const color = isPositive ? '#10b981' : '#f43f5e';
  const bgColor = isPositive ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)';
  const borderColor = isPositive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)';

  if (loading) return <ShimmerCard />;

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 relative overflow-hidden"
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ ...springTransition, delay: index * 0.08 }}
      layout
      whileHover={{ scale: 1.02, borderColor: 'rgba(255, 255, 255, 0.2)' }}
    >
      {/* Background glow */}
      <div
        className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl pointer-events-none"
        style={{ backgroundColor: bgColor }}
      />

      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        {icon && <span className="text-lg">{icon}</span>}
        <span className="text-white/50 text-xs font-medium uppercase tracking-wider">{label}</span>
      </div>

      {/* Value */}
      <motion.div layout transition={springTransition}>
        <AnimatedNumber
          value={value}
          prefix={prefix}
          suffix={suffix}
          decimals={decimals}
          className="text-2xl font-bold"
          style={{ color }}
        />
        <span className="text-2xl font-bold font-mono" style={{ color }}>
          {value < 0 ? '' : ''}
        </span>
      </motion.div>

      {/* Direction indicator */}
      <div className="flex items-center gap-1 mt-2">
        <motion.div
          className="w-5 h-5 rounded-full flex items-center justify-center"
          style={{ backgroundColor: bgColor, border: `1px solid ${borderColor}` }}
          animate={{ rotate: isPositive ? 0 : 180 }}
          transition={springTransition}
        >
          <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none">
            <path d="M6 9V3M6 3L3 6M6 3L9 6" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </motion.div>
        <span className="text-xs" style={{ color: `${color}99` }}>
          {isPositive ? 'Favorable' : 'Unfavorable'}
        </span>
      </div>
    </motion.div>
  );
}

/**
 * ImpactPanel - Real-time ML prediction display panel.
 *
 * Props:
 *   predictions - object { volumeChange, revenueImpact, marginImpact, competitiveRisk }
 *   loading     - boolean, whether predictions are loading
 *   error       - string|null, error message if prediction failed
 */
export default function ImpactPanel({ predictions = null, loading = false, error = null }) {
  const metrics = predictions
    ? [
        {
          label: 'Volume Change',
          value: predictions.volumeChange ?? 0,
          prefix: '',
          suffix: '%',
          decimals: 1,
          icon: '\u{1F4E6}',
          positive: (predictions.volumeChange ?? 0) >= 0,
        },
        {
          label: 'Revenue Impact',
          value: predictions.revenueImpact ?? 0,
          prefix: '$',
          suffix: '',
          decimals: 0,
          icon: '\u{1F4B0}',
          positive: (predictions.revenueImpact ?? 0) >= 0,
        },
        {
          label: 'Margin Impact',
          value: predictions.marginImpact ?? 0,
          prefix: '$',
          suffix: '',
          decimals: 0,
          icon: '\u{1F4CA}',
          positive: (predictions.marginImpact ?? 0) >= 0,
        },
        {
          label: 'Competitive Risk',
          value: predictions.competitiveRisk ?? 50,
          prefix: '',
          suffix: '/100',
          decimals: 0,
          icon: '\u{26A0}\u{FE0F}',
          positive: (predictions.competitiveRisk ?? 50) < 50,
        },
      ]
    : [];

  if (error) {
    return (
      <motion.div
        className="bg-white/5 backdrop-blur-xl border border-[#f43f5e]/30 rounded-2xl p-6 text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={springTransition}
      >
        <p className="text-[#f43f5e] text-sm font-medium mb-1">Prediction Error</p>
        <p className="text-white/40 text-xs">{error}</p>
      </motion.div>
    );
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold text-sm">ML Impact Predictions</h3>
        {loading && (
          <motion.div
            className="flex items-center gap-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="w-2 h-2 rounded-full bg-[#FFB81C] animate-pulse" />
            <span className="text-[#FFB81C] text-xs">Predicting...</span>
          </motion.div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <AnimatePresence mode="wait">
          {loading
            ? [0, 1, 2, 3].map((i) => <ShimmerCard key={`shimmer-${i}`} />)
            : metrics.map((metric, index) => (
                <MetricCard key={metric.label} {...metric} index={index} loading={false} />
              ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
