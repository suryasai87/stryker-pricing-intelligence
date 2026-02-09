import React from 'react';
import { motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * GlassCard - Reusable glass morphism card with optional hover animation.
 *
 * Props:
 *   children   - ReactNode, card content
 *   className  - string, additional CSS classes
 *   onClick    - function, click handler
 *   animate    - boolean, whether to apply whileHover scale animation (default: true)
 *   padding    - string, Tailwind padding class (default: 'p-6')
 */
export default function GlassCard({
  children,
  className = '',
  onClick,
  animate: shouldAnimate = true,
  padding = 'p-6',
}) {
  const baseClasses = `bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl ${padding}`;

  if (!shouldAnimate) {
    return (
      <div
        className={`${baseClasses} ${className}`}
        onClick={onClick}
        role={onClick ? 'button' : undefined}
        tabIndex={onClick ? 0 : undefined}
        onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick(e) : undefined}
      >
        {children}
      </div>
    );
  }

  return (
    <motion.div
      className={`${baseClasses} ${className}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick(e) : undefined}
      whileHover={{ scale: 1.02 }}
      transition={springTransition}
    >
      {children}
    </motion.div>
  );
}
