import React, { useEffect, useRef } from 'react';
import { useMotionValue, useTransform, animate, motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * Format a number according to the given format type.
 *
 * @param {number} num
 * @param {"currency"|"percent"|"number"|"compact"} format
 * @returns {string}
 */
function formatNumber(num, format) {
  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      }).format(num);

    case 'percent':
      return `${num.toFixed(1)}%`;

    case 'compact':
      if (Math.abs(num) >= 1_000_000_000) {
        return `$${(num / 1_000_000_000).toFixed(1)}B`;
      }
      if (Math.abs(num) >= 1_000_000) {
        return `$${(num / 1_000_000).toFixed(1)}M`;
      }
      if (Math.abs(num) >= 1_000) {
        return `$${(num / 1_000).toFixed(1)}K`;
      }
      return `$${num.toFixed(0)}`;

    case 'number':
    default:
      return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      }).format(num);
  }
}

/**
 * AnimatedNumber - Smoothly animates from the previous numeric value to a new one.
 *
 * Props:
 *   value     - number, the target numeric value
 *   format    - "currency" | "percent" | "number" | "compact" (default: "number")
 *   duration  - number, animation duration in seconds (default: 1)
 *   className - string, additional CSS classes
 */
export default function AnimatedNumber({
  value = 0,
  format = 'number',
  duration = 1,
  className = '',
}) {
  const motionValue = useMotionValue(0);
  const displayRef = useRef(null);
  const prevValue = useRef(0);

  useEffect(() => {
    const controls = animate(motionValue, value, {
      duration,
      ease: 'easeOut',
      onUpdate: (latest) => {
        if (displayRef.current) {
          displayRef.current.textContent = formatNumber(latest, format);
        }
      },
    });

    prevValue.current = value;

    return () => controls.stop();
  }, [value, format, duration, motionValue]);

  return (
    <motion.span
      ref={displayRef}
      className={`font-mono tabular-nums ${className}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      {formatNumber(0, format)}
    </motion.span>
  );
}
