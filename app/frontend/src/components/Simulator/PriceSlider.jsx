import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, useMotionValue, useTransform, useSpring } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * PriceSlider - Custom animated price change slider from -30% to +30%.
 *
 * Props:
 *   value          - number, current slider value (-30 to 30)
 *   onChange       - function(value), called on value change (debounced)
 *   debounceMs     - number, debounce delay in ms (default: 300)
 *   disabled       - boolean, whether the slider is disabled
 *   label          - string, label above the slider (default: 'Price Adjustment')
 */
export default function PriceSlider({
  value = 0,
  onChange,
  debounceMs = 300,
  disabled = false,
  label = 'Price Adjustment',
}) {
  const [internalValue, setInternalValue] = useState(value);
  const [isDragging, setIsDragging] = useState(false);
  const trackRef = useRef(null);
  const debounceRef = useRef(null);

  const MIN = -30;
  const MAX = 30;

  // Sync external value
  useEffect(() => {
    if (!isDragging) {
      setInternalValue(value);
    }
  }, [value, isDragging]);

  // Spring-animated display value
  const motionVal = useMotionValue(internalValue);
  const springVal = useSpring(motionVal, { stiffness: 300, damping: 30 });

  useEffect(() => {
    motionVal.set(internalValue);
  }, [internalValue, motionVal]);

  // Debounced onChange
  const debouncedOnChange = useCallback(
    (val) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onChange?.(val);
      }, debounceMs);
    },
    [onChange, debounceMs]
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const getPercentFromValue = (val) => ((val - MIN) / (MAX - MIN)) * 100;
  const getValueFromPercent = (pct) => Math.round(MIN + (pct / 100) * (MAX - MIN));

  const handleTrackInteraction = useCallback(
    (clientX) => {
      if (disabled || !trackRef.current) return;
      const rect = trackRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      const newVal = getValueFromPercent(pct);
      setInternalValue(newVal);
      debouncedOnChange(newVal);
    },
    [disabled, debouncedOnChange]
  );

  const handleMouseDown = (e) => {
    if (disabled) return;
    setIsDragging(true);
    handleTrackInteraction(e.clientX);

    const handleMouseMove = (e) => handleTrackInteraction(e.clientX);
    const handleMouseUp = () => {
      setIsDragging(false);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleTouchStart = (e) => {
    if (disabled) return;
    setIsDragging(true);
    handleTrackInteraction(e.touches[0].clientX);

    const handleTouchMove = (e) => handleTrackInteraction(e.touches[0].clientX);
    const handleTouchEnd = () => {
      setIsDragging(false);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
    };

    window.addEventListener('touchmove', handleTouchMove);
    window.addEventListener('touchend', handleTouchEnd);
  };

  const percent = getPercentFromValue(internalValue);
  const centerPercent = getPercentFromValue(0);
  const isPositive = internalValue > 0;
  const isNegative = internalValue < 0;

  // Track fill from center to thumb
  const fillLeft = isNegative ? percent : centerPercent;
  const fillWidth = Math.abs(percent - centerPercent);

  const trackColor = isPositive ? '#10b981' : isNegative ? '#f43f5e' : '#6b7280';
  const glowColor = isPositive ? 'rgba(16, 185, 129, 0.4)' : isNegative ? 'rgba(244, 63, 94, 0.4)' : 'rgba(107, 114, 128, 0.3)';

  return (
    <div className={`w-full ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-white/60 text-sm font-medium">{label}</span>
        <motion.div
          className="flex items-center gap-2"
          layout
          transition={springTransition}
        >
          <motion.span
            className="font-mono text-2xl font-bold"
            style={{ color: trackColor }}
            key={internalValue}
            initial={{ scale: 1.2, opacity: 0.7 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={springTransition}
          >
            {internalValue > 0 ? '+' : ''}{internalValue}%
          </motion.span>
        </motion.div>
      </div>

      {/* Slider Track */}
      <div className="relative py-3">
        <div
          ref={trackRef}
          className="relative h-2 rounded-full bg-white/10 cursor-pointer"
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
        >
          {/* Active fill */}
          <motion.div
            className="absolute top-0 h-full rounded-full"
            style={{
              left: `${fillLeft}%`,
              width: `${fillWidth}%`,
              backgroundColor: trackColor,
            }}
            animate={{
              left: `${fillLeft}%`,
              width: `${fillWidth}%`,
            }}
            transition={springTransition}
          />

          {/* Center marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-white/30 rounded-full"
            style={{ left: `${centerPercent}%` }}
          />

          {/* Tick marks */}
          {[-30, -20, -10, 0, 10, 20, 30].map((tick) => (
            <div
              key={tick}
              className="absolute top-full mt-2 -translate-x-1/2"
              style={{ left: `${getPercentFromValue(tick)}%` }}
            >
              <div className="w-px h-1.5 bg-white/20 mx-auto mb-1" />
              <span className="text-[10px] text-white/30 font-mono">
                {tick > 0 ? `+${tick}` : tick}
              </span>
            </div>
          ))}

          {/* Thumb */}
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 z-10"
            style={{ left: `${percent}%` }}
            animate={{ left: `${percent}%` }}
            transition={springTransition}
          >
            <motion.div
              className="w-6 h-6 rounded-full border-2 cursor-grab active:cursor-grabbing"
              style={{
                backgroundColor: trackColor,
                borderColor: 'rgba(255, 255, 255, 0.3)',
                boxShadow: `0 0 ${isDragging ? 20 : 12}px ${glowColor}, 0 0 ${isDragging ? 40 : 24}px ${glowColor}`,
              }}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
              animate={{
                scale: isDragging ? 1.15 : 1,
              }}
              transition={springTransition}
            />
          </motion.div>
        </div>
      </div>

      {/* Quick adjustment buttons */}
      <div className="flex items-center justify-center gap-2 mt-6">
        {[-10, -5, 0, 5, 10].map((preset) => (
          <motion.button
            key={preset}
            className={`
              px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors
              ${internalValue === preset
                ? 'bg-[#0057B8]/30 text-[#0057B8] border border-[#0057B8]/50'
                : 'bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:text-white/80'
              }
            `}
            onClick={() => {
              setInternalValue(preset);
              onChange?.(preset);
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={springTransition}
          >
            {preset > 0 ? `+${preset}%` : preset === 0 ? '0%' : `${preset}%`}
          </motion.button>
        ))}
      </div>
    </div>
  );
}
