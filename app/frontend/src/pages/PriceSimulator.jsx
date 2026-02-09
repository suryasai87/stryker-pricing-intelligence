import React from 'react';
import { motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

export default function PriceSimulator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
      className="space-y-6"
    >
      <div>
        <h1 className="text-2xl font-bold text-white">Price Simulator</h1>
        <p className="text-sm text-white/50 mt-1">Model pricing scenarios and project revenue impact</p>
      </div>
      <div className="glass-card p-8 flex items-center justify-center min-h-[400px]">
        <p className="text-white/40 text-sm">Price Simulator content will be implemented here.</p>
      </div>
    </motion.div>
  );
}
