import React, { useMemo } from 'react';
import { motion, useSpring, useMotionValue } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * MacroGauge - Radial gauge component showing a single metric (0-100 scale).
 *
 * Props:
 *   value    - number, current gauge value
 *   label    - string, metric label
 *   subtitle - string, optional subtitle
 *   min      - number, minimum scale value (default: 0)
 *   max      - number, maximum scale value (default: 100)
 *   size     - number, SVG size in pixels (default: 200)
 *   unit     - string, unit suffix (default: '')
 */
export default function MacroGauge({
  value = 50,
  label = 'Metric',
  subtitle = '',
  min = 0,
  max = 100,
  size = 200,
  unit = '',
}) {
  const normalizedValue = Math.max(min, Math.min(max, value));
  const percentage = ((normalizedValue - min) / (max - min)) * 100;

  // Gauge geometry
  const centerX = size / 2;
  const centerY = size / 2 + 10;
  const radius = size * 0.38;
  const strokeWidth = 12;
  const startAngle = 225; // degrees (bottom-left)
  const endAngle = -45;   // degrees (bottom-right)
  const totalAngle = 270;  // total sweep

  // Calculate arc path
  const polarToCartesian = (cx, cy, r, angleDeg) => {
    const rad = ((angleDeg - 90) * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  };

  const describeArc = (cx, cy, r, startDeg, endDeg) => {
    const start = polarToCartesian(cx, cy, r, endDeg);
    const end = polarToCartesian(cx, cy, r, startDeg);
    const sweepAngle = startDeg - endDeg;
    const largeArc = sweepAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  };

  // Background arc (full track)
  const bgArcPath = describeArc(centerX, centerY, radius, startAngle, endAngle);

  // Value arc
  const valueAngle = startAngle - (percentage / 100) * totalAngle;
  const valueArcPath = describeArc(centerX, centerY, radius, startAngle, valueAngle);

  // Needle position
  const needleAngle = startAngle - (percentage / 100) * totalAngle;
  const needleTip = polarToCartesian(centerX, centerY, radius - 20, needleAngle);

  // Color based on value
  const getColor = (pct) => {
    if (pct <= 33) return '#10b981';
    if (pct <= 66) return '#FFB81C';
    return '#f43f5e';
  };

  const gaugeColor = getColor(percentage);

  // Gradient ID unique per instance
  const gradientId = useMemo(() => `gauge-gradient-${label.replace(/\s/g, '-')}-${Math.random().toString(36).slice(2, 8)}`, [label]);

  // Tick marks
  const ticks = useMemo(() => {
    const result = [];
    const numTicks = 10;
    for (let i = 0; i <= numTicks; i++) {
      const pct = i / numTicks;
      const angle = startAngle - pct * totalAngle;
      const outerPoint = polarToCartesian(centerX, centerY, radius + 8, angle);
      const innerPoint = polarToCartesian(centerX, centerY, radius + 3, angle);
      const labelPoint = polarToCartesian(centerX, centerY, radius + 18, angle);
      const tickValue = Math.round(min + pct * (max - min));
      result.push({
        outer: outerPoint,
        inner: innerPoint,
        label: labelPoint,
        value: tickValue,
        isMajor: i % 5 === 0,
      });
    }
    return result;
  }, [centerX, centerY, radius, min, max]);

  // Arc circumference for dash animation
  const arcCircumference = (totalAngle / 360) * 2 * Math.PI * radius;
  const valueDash = (percentage / 100) * arcCircumference;

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 flex flex-col items-center"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={springTransition}
      whileHover={{ borderColor: 'rgba(255,255,255,0.2)' }}
    >
      <svg width={size} height={size * 0.7} viewBox={`0 0 ${size} ${size * 0.75}`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#FFB81C" />
            <stop offset="100%" stopColor="#f43f5e" />
          </linearGradient>
          <filter id={`${gradientId}-glow`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Tick marks */}
        {ticks.map((tick, i) => (
          <g key={i}>
            <line
              x1={tick.inner.x}
              y1={tick.inner.y}
              x2={tick.outer.x}
              y2={tick.outer.y}
              stroke={tick.isMajor ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.08)'}
              strokeWidth={tick.isMajor ? 2 : 1}
            />
            {tick.isMajor && (
              <text
                x={tick.label.x}
                y={tick.label.y}
                textAnchor="middle"
                alignmentBaseline="middle"
                fill="rgba(255,255,255,0.3)"
                fontSize="9"
                fontFamily="monospace"
              >
                {tick.value}
              </text>
            )}
          </g>
        ))}

        {/* Background arc */}
        <path
          d={bgArcPath}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Value arc */}
        <motion.path
          d={valueArcPath}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          filter={`url(#${gradientId}-glow)`}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ ...springTransition, duration: 1.5 }}
        />

        {/* Needle */}
        <motion.g
          initial={{ rotate: startAngle - 90 }}
          animate={{ rotate: needleAngle - 90 }}
          transition={springTransition}
          style={{ transformOrigin: `${centerX}px ${centerY}px` }}
        >
          {/* Needle body */}
          <line
            x1={centerX}
            y1={centerY}
            x2={centerX}
            y2={centerY - radius + 25}
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
          />
          {/* Needle shadow */}
          <line
            x1={centerX + 1}
            y1={centerY + 1}
            x2={centerX + 1}
            y2={centerY - radius + 26}
            stroke="rgba(0,0,0,0.3)"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </motion.g>

        {/* Center dot */}
        <circle cx={centerX} cy={centerY} r={6} fill={gaugeColor} />
        <circle cx={centerX} cy={centerY} r={3} fill="white" />

        {/* Value text */}
        <text
          x={centerX}
          y={centerY + 30}
          textAnchor="middle"
          fill="white"
          fontSize="24"
          fontFamily="monospace"
          fontWeight="bold"
        >
          {normalizedValue.toFixed(0)}{unit}
        </text>
      </svg>

      {/* Label */}
      <div className="text-center mt-1">
        <p className="text-white text-sm font-medium">{label}</p>
        {subtitle && <p className="text-white/40 text-xs mt-0.5">{subtitle}</p>}
      </div>

      {/* Status indicator */}
      <div className="flex items-center gap-1.5 mt-2">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: gaugeColor }} />
        <span className="text-xs" style={{ color: gaugeColor }}>
          {percentage <= 33 ? 'Low' : percentage <= 66 ? 'Moderate' : 'High'}
        </span>
      </div>
    </motion.div>
  );
}
