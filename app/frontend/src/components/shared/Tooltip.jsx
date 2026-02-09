import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * Compute position classes and transform origin for the tooltip arrow.
 */
function getPositionStyles(position) {
  switch (position) {
    case 'top':
      return {
        wrapper: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
        initial: { opacity: 0, y: 6, scale: 0.95 },
        animate: { opacity: 1, y: 0, scale: 1 },
        exit: { opacity: 0, y: 6, scale: 0.95 },
        arrow: 'top-full left-1/2 -translate-x-1/2 border-t-white/20 border-l-transparent border-r-transparent border-b-transparent',
      };
    case 'bottom':
      return {
        wrapper: 'top-full left-1/2 -translate-x-1/2 mt-2',
        initial: { opacity: 0, y: -6, scale: 0.95 },
        animate: { opacity: 1, y: 0, scale: 1 },
        exit: { opacity: 0, y: -6, scale: 0.95 },
        arrow: 'bottom-full left-1/2 -translate-x-1/2 border-b-white/20 border-l-transparent border-r-transparent border-t-transparent',
      };
    case 'left':
      return {
        wrapper: 'right-full top-1/2 -translate-y-1/2 mr-2',
        initial: { opacity: 0, x: 6, scale: 0.95 },
        animate: { opacity: 1, x: 0, scale: 1 },
        exit: { opacity: 0, x: 6, scale: 0.95 },
        arrow: 'left-full top-1/2 -translate-y-1/2 border-l-white/20 border-t-transparent border-b-transparent border-r-transparent',
      };
    case 'right':
      return {
        wrapper: 'left-full top-1/2 -translate-y-1/2 ml-2',
        initial: { opacity: 0, x: -6, scale: 0.95 },
        animate: { opacity: 1, x: 0, scale: 1 },
        exit: { opacity: 0, x: -6, scale: 0.95 },
        arrow: 'right-full top-1/2 -translate-y-1/2 border-r-white/20 border-t-transparent border-b-transparent border-l-transparent',
      };
    default:
      return getPositionStyles('top');
  }
}

/**
 * Tooltip - Hover tooltip using Framer Motion AnimatePresence.
 *
 * Props:
 *   content   - ReactNode | string, the tooltip content
 *   children  - ReactNode, the trigger element
 *   position  - "top" | "bottom" | "left" | "right" (default: "top")
 */
export default function Tooltip({ content, children, position = 'top' }) {
  const [visible, setVisible] = useState(false);
  const pos = getPositionStyles(position);

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}

      <AnimatePresence>
        {visible && content && (
          <motion.div
            className={`absolute z-50 ${pos.wrapper} pointer-events-none`}
            initial={pos.initial}
            animate={pos.animate}
            exit={pos.exit}
            transition={springTransition}
          >
            <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-lg px-3 py-2 text-xs text-white/90 shadow-xl whitespace-nowrap">
              {content}
            </div>
            {/* Arrow nub */}
            <div
              className={`absolute w-0 h-0 border-[5px] ${pos.arrow}`}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
