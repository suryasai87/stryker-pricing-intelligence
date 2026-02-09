import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import WaterfallChart from './WaterfallChart';
import MarginLeakTable from './MarginLeakTable';
import ProductSelector from '../Simulator/ProductSelector';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * WaterfallPage - Page combining WaterfallChart and MarginLeakTable.
 *
 * Props:
 *   apiEndpoint - string, base API endpoint (default: '/api/waterfall')
 */
export default function WaterfallPage({ apiEndpoint = '/api/waterfall' }) {
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [waterfallData, setWaterfallData] = useState(null);
  const [marginData, setMarginData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchWaterfall = useCallback(
    async (productId) => {
      setLoading(true);
      setError(null);
      try {
        const url = productId
          ? `${apiEndpoint}?product_id=${productId}`
          : apiEndpoint;
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch waterfall data');
        const data = await res.json();
        setWaterfallData(data.waterfall || null);
        setMarginData(data.margin_table || null);
      } catch (err) {
        console.error('Waterfall fetch error:', err);
        // Use default demo data on error
        setWaterfallData(null);
        setMarginData(null);
      } finally {
        setLoading(false);
      }
    },
    [apiEndpoint]
  );

  // Fetch data when product changes
  useEffect(() => {
    fetchWaterfall(selectedProduct?.id);
  }, [selectedProduct, fetchWaterfall]);

  const handleProductSelect = (product) => {
    setSelectedProduct(product);
  };

  // Summary stats
  const summaryStats = [
    { label: 'Avg Realization', value: '72.4%', color: '#FFB81C' },
    { label: 'Products Below 50%', value: '3', color: '#f43f5e' },
    { label: 'Total Leakage (Q)', value: '$4.2M', color: '#f43f5e' },
    { label: 'Highest Margin', value: '74.0%', color: '#10b981' },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springTransition}
      >
        <h2 className="text-white text-xl font-bold">Price Waterfall</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Analyze price decomposition and identify margin leakage across the portfolio
        </p>
      </motion.div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryStats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springTransition, delay: index * 0.05 }}
          >
            <p className="text-white/40 text-xs font-medium uppercase tracking-wider">{stat.label}</p>
            <p className="font-mono text-xl font-bold mt-1" style={{ color: stat.color }}>
              {stat.value}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Product Selector */}
      <motion.div
        className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.1 }}
      >
        <label className="text-white/50 text-xs font-medium uppercase tracking-wider mb-3 block">
          Select Product for Waterfall
        </label>
        <ProductSelector selectedProduct={selectedProduct} onSelect={handleProductSelect} />
      </motion.div>

      {/* Waterfall Chart */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.15 }}
      >
        <WaterfallChart
          data={waterfallData}
          loading={loading}
          title={selectedProduct ? `${selectedProduct.name} - Price Waterfall` : 'Price Waterfall Analysis'}
        />
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

      {/* Margin Leak Table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.2 }}
      >
        <MarginLeakTable data={marginData} loading={loading} />
      </motion.div>
    </div>
  );
}
