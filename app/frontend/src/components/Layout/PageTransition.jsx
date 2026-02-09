import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * PageTransition - Wrapper component for animated page transitions.
 *
 * Uses AnimatePresence + motion.div to slide and fade pages in and out.
 *
 * Props:
 *   children   - ReactNode, the page content to animate
 *   pageKey    - string, unique key for the current page (triggers re-animation on change)
 *   className  - string, additional CSS classes for the wrapper
 */
export default function PageTransition({ children, pageKey, className = '' }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={pageKey}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={springTransition}
        className={`w-full ${className}`}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
