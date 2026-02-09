import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ASPGapChart from './ASPGapChart';
import MarketShareTrend from './MarketShareTrend';
import PatentTimeline from './PatentTimeline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const CATEGORIES = [
  { id: 'all', label: 'All Categories' },
  { id: 'joint_replacement', label: 'Joint Replacement' },
  { id: 'trauma', label: 'Trauma & Extremities' },
  { id: 'spine', label: 'Spine' },
  { id: 'instruments', label: 'Instruments' },
  { id: 'medsurg', label: 'Medical/Surgical' },
];

/**
 * CompetitivePage - Page combining ASPGapChart, MarketShareTrend, PatentTimeline.
 *
 * Props:
 *   apiEndpoint - string, base API endpoint (default: '/api/competitive')
 */
export default function CompetitivePage({ apiEndpoint = '/api/competitive' }) {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [aspData, setAspData] = useState(null);
  const [shareData, setShareData] = useState(null);
  const [patentData, setPatentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchCompetitiveLandscape = useCallback(
    async (category) => {
      setLoading(true);
      setError(null);
      try {
        const url = category && category !== 'all'
          ? `${apiEndpoint}?category=${category}`
          : apiEndpoint;
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch competitive data');
        const data = await res.json();
        setAspData(data.asp_comparison || null);
        setShareData(data.market_share || null);
        setPatentData(data.patents || null);
      } catch (err) {
        console.error('Competitive fetch error:', err);
        // Use default demo data on error
        setAspData(null);
        setShareData(null);
        setPatentData(null);
      } finally {
        setLoading(false);
      }
    },
    [apiEndpoint]
  );

  useEffect(() => {
    fetchCompetitiveLandscape(selectedCategory);
  }, [selectedCategory, fetchCompetitiveLandscape]);

  // Summary metrics
  const summaryMetrics = [
    { label: 'Market Position', value: '#1', subtext: 'Overall Ortho', color: '#0057B8' },
    { label: 'Market Share', value: '32.0%', subtext: '+3.5pp YoY', color: '#10b981' },
    { label: 'Avg ASP Premium', value: '+8.2%', subtext: 'vs competitors', color: '#FFB81C' },
    { label: 'Patents Expiring', value: '3', subtext: 'within 3 years', color: '#f43f5e' },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springTransition}
      >
        <h2 className="text-white text-xl font-bold">Competitive Landscape</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Market intelligence, pricing gaps, and patent protection analysis
        </p>
      </motion.div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryMetrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springTransition, delay: index * 0.05 }}
          >
            <p className="text-white/40 text-xs font-medium uppercase tracking-wider">{metric.label}</p>
            <p className="font-mono text-2xl font-bold mt-1" style={{ color: metric.color }}>
              {metric.value}
            </p>
            <p className="text-white/30 text-xs mt-0.5">{metric.subtext}</p>
          </motion.div>
        ))}
      </div>

      {/* Category Filter */}
      <motion.div
        className="flex flex-wrap gap-2"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.1 }}
      >
        {CATEGORIES.map((cat) => (
          <motion.button
            key={cat.id}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
              selectedCategory === cat.id
                ? 'bg-[#0057B8]/20 text-[#0057B8] border border-[#0057B8]/40'
                : 'bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:text-white/80'
            }`}
            onClick={() => setSelectedCategory(cat.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={springTransition}
          >
            {cat.label}
          </motion.button>
        ))}
      </motion.div>

      {/* Error State */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="bg-white/5 backdrop-blur-xl border border-[#f43f5e]/30 rounded-2xl p-6 text-center"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={springTransition}
          >
            <p className="text-[#f43f5e] text-sm font-medium">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ASP Gap Chart */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.15 }}
      >
        <ASPGapChart data={aspData} loading={loading} />
      </motion.div>

      {/* Market Share + Patent Timeline side by side on large screens */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...springTransition, delay: 0.2 }}
        >
          <MarketShareTrend data={shareData} loading={loading} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...springTransition, delay: 0.25 }}
        >
          <PatentTimeline data={patentData} loading={loading} />
        </motion.div>
      </div>
    </div>
  );
}
