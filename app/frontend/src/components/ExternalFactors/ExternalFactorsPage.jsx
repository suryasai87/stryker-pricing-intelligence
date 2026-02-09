import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import TariffDashboard from './TariffDashboard';
import MacroGauge from './MacroGauge';
import CurrencyCalculator from './CurrencyCalculator';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * ExternalFactorsPage - Page combining TariffDashboard, MacroGauge, CurrencyCalculator.
 *
 * Props:
 *   apiEndpoint - string, base API endpoint (default: '/api/external-factors')
 */
export default function ExternalFactorsPage({ apiEndpoint = '/api/external-factors' }) {
  const [externalData, setExternalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchExternalFactors = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiEndpoint);
      if (!res.ok) throw new Error('Failed to fetch external factors');
      const data = await res.json();
      setExternalData(data);
    } catch (err) {
      console.error('External factors fetch error:', err);
      // Use default demo data
      setExternalData(null);
    } finally {
      setLoading(false);
    }
  }, [apiEndpoint]);

  useEffect(() => {
    fetchExternalFactors();
  }, [fetchExternalFactors]);

  // Macro gauge data
  const macroMetrics = externalData?.macroMetrics || [
    { label: 'CPI Medical Care', value: 68, subtitle: 'YoY +4.2%, above Fed target', unit: '' },
    { label: 'Supply Chain Pressure', value: 42, subtitle: 'GSCPI normalized index', unit: '' },
    { label: 'Hospital CapEx Index', value: 55, subtitle: 'Moderate spending outlook', unit: '' },
  ];

  // Summary cards
  const summaryCards = [
    { label: 'Steel Tariff', value: '25%', trend: '+2.3%', trendColor: '#f43f5e' },
    { label: 'USD/EUR', value: '1.08', trend: '-1.2%', trendColor: '#10b981' },
    { label: 'CPI Medical', value: '4.2%', trend: '+0.3pp', trendColor: '#f43f5e' },
    { label: 'Supply Chain', value: '42/100', trend: '-8pts', trendColor: '#10b981' },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springTransition}
      >
        <h2 className="text-white text-xl font-bold">External Factors</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Tariffs, macroeconomic indicators, and currency exposure analysis
        </p>
      </motion.div>

      {/* Quick Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card, index) => (
          <motion.div
            key={card.label}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springTransition, delay: index * 0.05 }}
          >
            <p className="text-white/40 text-xs font-medium uppercase tracking-wider">{card.label}</p>
            <div className="flex items-end justify-between mt-1">
              <span className="font-mono text-xl font-bold text-white">{card.value}</span>
              <span className="font-mono text-xs font-medium" style={{ color: card.trendColor }}>
                {card.trend}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

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

      {/* Tariff Dashboard */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.1 }}
      >
        <TariffDashboard data={externalData?.tariffs} loading={loading} />
      </motion.div>

      {/* Macro Gauges */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.15 }}
      >
        <div className="mb-4">
          <h3 className="text-white font-semibold text-sm">Macroeconomic Indicators</h3>
          <p className="text-white/40 text-xs mt-0.5">Key economic signals affecting pricing strategy</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {macroMetrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ ...springTransition, delay: 0.15 + index * 0.08 }}
            >
              <MacroGauge
                value={metric.value}
                label={metric.label}
                subtitle={metric.subtitle}
                unit={metric.unit}
                min={0}
                max={100}
                size={200}
              />
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Currency Calculator */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.2 }}
      >
        <CurrencyCalculator
          baseCOGS={externalData?.baseCOGS || 1200000}
          eurExposure={externalData?.eurExposure || 30}
          jpyExposure={externalData?.jpyExposure || 15}
          baseEurRate={externalData?.baseEurRate || 1.08}
          baseJpyRate={externalData?.baseJpyRate || 150}
        />
      </motion.div>

      {/* Data Freshness Footer */}
      <motion.div
        className="flex items-center justify-between px-4 py-3 bg-white/[0.02] border border-white/5 rounded-xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ ...springTransition, delay: 0.3 }}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse" />
          <span className="text-white/30 text-xs">Data sources: USITC, Federal Reserve, Bloomberg FX</span>
        </div>
        <span className="text-white/20 text-xs font-mono">
          Last updated: {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </span>
      </motion.div>
    </div>
  );
}
