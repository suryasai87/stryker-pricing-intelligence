import React from 'react';
import { motion } from 'framer-motion';

/**
 * LoadingShimmer - Skeleton loading placeholder with animated gradient shimmer.
 *
 * Props:
 *   width     - string, CSS width (default: '100%')
 *   height    - string, CSS height for a single line (default: '20px')
 *   className - string, additional CSS classes
 *   count     - number, how many shimmer lines to render (default: 1)
 *   rounded   - string, Tailwind border-radius class (default: 'rounded-lg')
 */
export default function LoadingShimmer({
  width = '100%',
  height = '20px',
  className = '',
  count = 1,
  rounded = 'rounded-lg',
}) {
  const lines = Array.from({ length: count }, (_, i) => i);

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {lines.map((i) => (
        <motion.div
          key={i}
          className={`relative overflow-hidden ${rounded}`}
          style={{
            width: i === lines.length - 1 && count > 1 ? '75%' : width,
            height,
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: i * 0.05 }}
        >
          <motion.div
            className="absolute inset-0"
            style={{
              background:
                'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%)',
            }}
            animate={{ x: ['-100%', '100%'] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
              delay: i * 0.1,
            }}
          />
        </motion.div>
      ))}
    </div>
  );
}
