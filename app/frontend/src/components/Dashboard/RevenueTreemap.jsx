import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as d3 from 'd3';
import GlassCard from '../shared/GlassCard';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/** Default segment data if none is provided */
const DEFAULT_SEGMENTS = [
  { name: 'Orthopaedics', value: 6200, growth: 5.2, color: '#0057B8' },
  { name: 'MedSurg', value: 5100, growth: 3.8, color: '#10b981' },
  { name: 'Neurotechnology', value: 3800, growth: 7.1, color: '#FFB81C' },
  { name: 'Capital Equipment', value: 2400, growth: -1.2, color: '#8b5cf6' },
  { name: 'Consumables', value: 1900, growth: 4.5, color: '#f43f5e' },
];

/**
 * RevenueTreemap - D3-based treemap visualization of revenue by segment.
 *
 * Props:
 *   data      - array of { name: string, value: number, growth: number, color: string }
 *   className - string, additional CSS classes
 */
export default function RevenueTreemap({ data = DEFAULT_SEGMENTS, className = '' }) {
  const containerRef = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [tooltip, setTooltip] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  /** Compute treemap layout */
  const computeLayout = useCallback(() => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height || 320;
    setDimensions({ width, height });

    const root = d3
      .hierarchy({ children: data })
      .sum((d) => d.value || 0)
      .sort((a, b) => b.value - a.value);

    d3.treemap()
      .size([width, height])
      .paddingInner(4)
      .paddingOuter(2)
      .round(true)(root);

    setNodes(root.leaves());
  }, [data]);

  useEffect(() => {
    computeLayout();

    const observer = new ResizeObserver(() => computeLayout());
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [computeLayout]);

  const formatValue = (v) => {
    if (v >= 1000) return `$${(v / 1000).toFixed(1)}B`;
    return `$${v}M`;
  };

  return (
    <GlassCard className={className} animate={false} padding="p-0">
      {/* Header */}
      <div className="px-6 pt-5 pb-3 flex items-center justify-between">
        <h3 className="text-white font-semibold text-sm">Revenue by Segment</h3>
        <span className="text-white/30 text-xs">Click for detail</span>
      </div>

      {/* Treemap Container */}
      <motion.div
        ref={containerRef}
        className="relative w-full px-2 pb-2"
        style={{ height: 320 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ ...springTransition, delay: 0.2 }}
      >
        {nodes.map((node, i) => {
          const d = node.data;
          const w = node.x1 - node.x0;
          const h = node.y1 - node.y0;
          const showLabel = w > 80 && h > 50;

          return (
            <motion.div
              key={d.name}
              className="absolute rounded-xl overflow-hidden cursor-pointer"
              style={{
                left: node.x0,
                top: node.y0,
                width: w,
                height: h,
                backgroundColor: d.color || '#0057B8',
              }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ ...springTransition, delay: 0.1 + i * 0.06 }}
              whileHover={{ scale: 1.03, zIndex: 10 }}
              onMouseEnter={(e) => {
                const rect = containerRef.current.getBoundingClientRect();
                setTooltip({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  data: d,
                });
              }}
              onMouseMove={(e) => {
                if (!containerRef.current) return;
                const rect = containerRef.current.getBoundingClientRect();
                setTooltip((prev) =>
                  prev ? { ...prev, x: e.clientX - rect.left, y: e.clientY - rect.top } : null
                );
              }}
              onMouseLeave={() => setTooltip(null)}
            >
              {/* Subtle gradient overlay */}
              <div
                className="absolute inset-0"
                style={{
                  background: 'linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 60%)',
                }}
              />

              {showLabel && (
                <div className="relative z-10 p-3 flex flex-col justify-between h-full">
                  <span className="text-white font-semibold text-xs leading-tight truncate">
                    {d.name}
                  </span>
                  <div>
                    <span className="text-white/90 font-mono text-sm font-bold block">
                      {formatValue(d.value)}
                    </span>
                    <span
                      className={`text-xs font-mono ${
                        d.growth >= 0 ? 'text-emerald-200' : 'text-rose-200'
                      }`}
                    >
                      {d.growth >= 0 ? '+' : ''}
                      {d.growth}%
                    </span>
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}

        {/* Tooltip */}
        <AnimatePresence>
          {tooltip && (
            <motion.div
              className="absolute z-50 pointer-events-none"
              style={{
                left: tooltip.x + 12,
                top: tooltip.y - 10,
              }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={springTransition}
            >
              <div className="bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-lg px-4 py-3 shadow-2xl">
                <p className="text-white font-semibold text-sm mb-1">{tooltip.data.name}</p>
                <p className="text-white/70 text-xs">
                  Revenue:{' '}
                  <span className="text-white font-mono font-bold">
                    {formatValue(tooltip.data.value)}
                  </span>
                </p>
                <p className="text-white/70 text-xs">
                  Growth:{' '}
                  <span
                    className={`font-mono font-bold ${
                      tooltip.data.growth >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {tooltip.data.growth >= 0 ? '+' : ''}
                    {tooltip.data.growth}%
                  </span>
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </GlassCard>
  );
}
