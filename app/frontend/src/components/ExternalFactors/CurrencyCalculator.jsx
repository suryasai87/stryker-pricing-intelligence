import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * CurrencyCalculator - Interactive FX impact calculator.
 *
 * Props:
 *   baseCOGS         - number, base COGS in USD (default: 1200000)
 *   eurExposure      - number, percentage of COGS in EUR (default: 30)
 *   jpyExposure      - number, percentage of COGS in JPY (default: 15)
 *   baseEurRate      - number, baseline USD/EUR rate (default: 1.08)
 *   baseJpyRate      - number, baseline USD/JPY rate (default: 150)
 *   onRatesChange    - function({ eurRate, jpyRate, impacts }), called on rate changes
 */
export default function CurrencyCalculator({
  baseCOGS = 1200000,
  eurExposure = 30,
  jpyExposure = 15,
  baseEurRate = 1.08,
  baseJpyRate = 150,
  onRatesChange,
}) {
  const [eurRate, setEurRate] = useState(baseEurRate);
  const [jpyRate, setJpyRate] = useState(baseJpyRate);

  // Calculate impacts
  const impacts = useMemo(() => {
    // EUR impact: if EUR strengthens (rate goes up), our EUR-denominated costs increase in USD
    const eurChange = ((eurRate - baseEurRate) / baseEurRate) * 100;
    const eurCOGSExposure = baseCOGS * (eurExposure / 100);
    const eurImpact = eurCOGSExposure * (eurChange / 100);

    // JPY impact: if JPY weakens (rate goes up), our JPY-denominated costs decrease in USD
    const jpyChange = ((jpyRate - baseJpyRate) / baseJpyRate) * 100;
    const jpyCOGSExposure = baseCOGS * (jpyExposure / 100);
    const jpyImpact = -jpyCOGSExposure * (jpyChange / 100); // inverse because higher JPY/USD means weaker JPY

    const totalImpact = eurImpact + jpyImpact;
    const totalPricingImpact = totalImpact * 0.6; // 60% pass-through

    return {
      eurChange,
      jpyChange,
      eurImpact,
      jpyImpact,
      totalImpact,
      totalPricingImpact,
      eurCOGSExposure,
      jpyCOGSExposure,
    };
  }, [eurRate, jpyRate, baseCOGS, eurExposure, jpyExposure, baseEurRate, baseJpyRate]);

  // Notify parent of changes
  const handleEurChange = (e) => {
    const val = parseFloat(e.target.value) || 0;
    setEurRate(val);
    onRatesChange?.({ eurRate: val, jpyRate, impacts });
  };

  const handleJpyChange = (e) => {
    const val = parseFloat(e.target.value) || 0;
    setJpyRate(val);
    onRatesChange?.({ eurRate, jpyRate: val, impacts });
  };

  const formatCurrency = (val) => {
    const abs = Math.abs(val);
    const formatted = abs >= 1000000
      ? `$${(abs / 1000000).toFixed(2)}M`
      : abs >= 1000
        ? `$${(abs / 1000).toFixed(1)}K`
        : `$${abs.toFixed(0)}`;
    return val >= 0 ? `+${formatted}` : `-${formatted}`;
  };

  const getColor = (val) => (val >= 0 ? '#f43f5e' : '#10b981');

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <h3 className="text-white font-semibold text-sm mb-1">FX Impact Calculator</h3>
      <p className="text-white/40 text-xs mb-6">
        Model currency fluctuation impact on COGS and pricing
      </p>

      {/* Rate Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        {/* EUR/USD */}
        <div className="bg-white/5 rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-white text-sm font-medium">USD/EUR</p>
              <p className="text-white/40 text-[10px]">
                Base: {baseEurRate.toFixed(2)} | Exposure: {eurExposure}%
              </p>
            </div>
            <div className="flex items-center gap-1 text-xs">
              <span className="font-mono" style={{ color: getColor(impacts.eurChange) }}>
                {impacts.eurChange >= 0 ? '+' : ''}{impacts.eurChange.toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={baseEurRate * 0.85}
              max={baseEurRate * 1.15}
              step={0.01}
              value={eurRate}
              onChange={handleEurChange}
              className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #10b981, #FFB81C, #f43f5e)`,
              }}
            />
            <input
              type="number"
              value={eurRate}
              onChange={handleEurChange}
              step={0.01}
              className="w-20 bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-white text-sm font-mono text-center outline-none focus:border-[#0057B8]/50"
            />
          </div>
        </div>

        {/* USD/JPY */}
        <div className="bg-white/5 rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-white text-sm font-medium">USD/JPY</p>
              <p className="text-white/40 text-[10px]">
                Base: {baseJpyRate.toFixed(0)} | Exposure: {jpyExposure}%
              </p>
            </div>
            <div className="flex items-center gap-1 text-xs">
              <span className="font-mono" style={{ color: getColor(-impacts.jpyChange) }}>
                {impacts.jpyChange >= 0 ? '+' : ''}{impacts.jpyChange.toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={baseJpyRate * 0.85}
              max={baseJpyRate * 1.15}
              step={0.5}
              value={jpyRate}
              onChange={handleJpyChange}
              className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #10b981, #FFB81C, #f43f5e)`,
              }}
            />
            <input
              type="number"
              value={jpyRate}
              onChange={handleJpyChange}
              step={0.5}
              className="w-20 bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-white text-sm font-mono text-center outline-none focus:border-[#0057B8]/50"
            />
          </div>
        </div>
      </div>

      {/* Impact Results */}
      <div className="space-y-3">
        <p className="text-white/50 text-xs font-medium uppercase tracking-wider">Impact Analysis</p>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {/* EUR COGS Impact */}
          <motion.div
            className="bg-white/5 rounded-xl p-3 border border-white/10"
            layout
            transition={springTransition}
          >
            <p className="text-white/40 text-[10px] uppercase tracking-wider">EUR COGS Impact</p>
            <p className="font-mono text-lg font-bold mt-1" style={{ color: getColor(impacts.eurImpact) }}>
              {formatCurrency(impacts.eurImpact)}
            </p>
          </motion.div>

          {/* JPY COGS Impact */}
          <motion.div
            className="bg-white/5 rounded-xl p-3 border border-white/10"
            layout
            transition={springTransition}
          >
            <p className="text-white/40 text-[10px] uppercase tracking-wider">JPY COGS Impact</p>
            <p className="font-mono text-lg font-bold mt-1" style={{ color: getColor(impacts.jpyImpact) }}>
              {formatCurrency(impacts.jpyImpact)}
            </p>
          </motion.div>

          {/* Total COGS Impact */}
          <motion.div
            className="bg-white/5 rounded-xl p-3 border border-white/10"
            layout
            transition={springTransition}
          >
            <p className="text-white/40 text-[10px] uppercase tracking-wider">Total COGS Impact</p>
            <p className="font-mono text-lg font-bold mt-1" style={{ color: getColor(impacts.totalImpact) }}>
              {formatCurrency(impacts.totalImpact)}
            </p>
          </motion.div>

          {/* Pricing Impact (60% pass-through) */}
          <motion.div
            className="bg-white/5 rounded-xl p-3 border border-[#0057B8]/30"
            layout
            transition={springTransition}
          >
            <p className="text-white/40 text-[10px] uppercase tracking-wider">Pricing Impact</p>
            <p className="font-mono text-lg font-bold mt-1 text-[#0057B8]">
              {formatCurrency(impacts.totalPricingImpact)}
            </p>
            <p className="text-white/30 text-[10px] mt-0.5">60% pass-through</p>
          </motion.div>
        </div>
      </div>

      {/* Reset button */}
      <div className="mt-4 flex justify-end">
        <motion.button
          className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white/50 text-xs hover:bg-white/10 hover:text-white/80 transition-colors"
          onClick={() => {
            setEurRate(baseEurRate);
            setJpyRate(baseJpyRate);
          }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          transition={springTransition}
        >
          Reset to Baseline
        </motion.button>
      </div>
    </motion.div>
  );
}
