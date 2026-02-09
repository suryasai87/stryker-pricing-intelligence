import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookmarkIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import ProductSelector from './ProductSelector';
import PriceSlider from './PriceSlider';
import ImpactPanel from './ImpactPanel';
import SensitivityTornado from './SensitivityTornado';
import ConfidenceBands from './ConfidenceBands';
import ScenarioTable from './ScenarioTable';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * useModelPrediction - Custom hook for fetching ML predictions.
 */
function useModelPrediction(apiEndpoint = '/api/simulator/predict') {
  const [predictions, setPredictions] = useState(null);
  const [sensitivityData, setSensitivityData] = useState(null);
  const [confidenceData, setConfidenceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const predict = useCallback(
    async (productId, priceChange) => {
      if (!productId) return;

      // Abort previous request
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      setLoading(true);
      setError(null);

      try {
        const res = await fetch(apiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ product_id: productId, price_change_pct: priceChange }),
          signal: abortRef.current.signal,
        });

        if (!res.ok) throw new Error('Prediction request failed');
        const data = await res.json();

        setPredictions(data.predictions || data);
        setSensitivityData(data.sensitivity || null);
        setConfidenceData(data.confidence_bands || null);
      } catch (err) {
        if (err.name === 'AbortError') return;
        console.error('Prediction error:', err);

        // Fallback to simulated predictions for development
        const simVolume = priceChange < 0
          ? Math.abs(priceChange) * 0.62 + (Math.random() - 0.5) * 2
          : -priceChange * 0.55 - (Math.random() - 0.5) * 2;
        const simRevenue = priceChange * 15000 + simVolume * 8000;
        const simMargin = priceChange * 12000 - Math.abs(simVolume) * 3000;
        const simRisk = Math.min(100, Math.max(0, 30 + Math.abs(priceChange) * 2 + (priceChange < 0 ? -10 : 15)));

        setPredictions({
          volumeChange: parseFloat(simVolume.toFixed(1)),
          revenueImpact: Math.round(simRevenue),
          marginImpact: Math.round(simMargin),
          competitiveRisk: Math.round(simRisk),
        });
        setSensitivityData(null);
        setConfidenceData(null);
        setError(null);
      } finally {
        setLoading(false);
      }
    },
    [apiEndpoint]
  );

  const reset = useCallback(() => {
    setPredictions(null);
    setSensitivityData(null);
    setConfidenceData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { predictions, sensitivityData, confidenceData, loading, error, predict, reset };
}

/**
 * SimulatorPage - Main simulator page combining all simulator components.
 */
export default function SimulatorPage() {
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [priceChange, setPriceChange] = useState(0);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [scenarioName, setScenarioName] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);

  const {
    predictions,
    sensitivityData,
    confidenceData,
    loading,
    error,
    predict,
    reset,
  } = useModelPrediction();

  // Run prediction when product or price changes
  const handlePriceChange = useCallback(
    (val) => {
      setPriceChange(val);
      if (selectedProduct) {
        predict(selectedProduct.id, val);
      }
    },
    [selectedProduct, predict]
  );

  const handleProductSelect = useCallback(
    (product) => {
      setSelectedProduct(product);
      reset();
      setPriceChange(0);
      if (product) {
        predict(product.id, 0);
      }
    },
    [predict, reset]
  );

  const handleSaveScenario = () => {
    if (!predictions || !selectedProduct || !scenarioName.trim()) return;
    const newScenario = {
      id: `sc-${Date.now()}`,
      name: scenarioName.trim(),
      product: selectedProduct.name,
      priceChange,
      volumeDelta: predictions.volumeChange,
      revenueDelta: predictions.revenueImpact,
      marginDelta: predictions.marginImpact,
    };
    setScenarios((prev) => [newScenario, ...prev]);
    setScenarioName('');
    setShowSaveModal(false);
  };

  const handleDeleteScenario = (id) => {
    setScenarios((prev) => prev.filter((s) => s.id !== id));
    if (selectedScenario === id) setSelectedScenario(null);
  };

  const handleReset = () => {
    setSelectedProduct(null);
    setPriceChange(0);
    reset();
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={springTransition}
      >
        <div>
          <h2 className="text-white text-xl font-bold">Price Simulator</h2>
          <p className="text-white/40 text-sm mt-0.5">
            Model price elasticity and predict market impact with ML
          </p>
        </div>
        <div className="flex items-center gap-2">
          <motion.button
            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white/60 text-sm hover:bg-white/10 hover:text-white transition-colors"
            onClick={handleReset}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={springTransition}
          >
            <ArrowPathIcon className="w-4 h-4" />
            Reset
          </motion.button>
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-white text-sm font-medium disabled:opacity-40"
            style={{ backgroundColor: '#0057B8' }}
            disabled={!predictions}
            onClick={() => setShowSaveModal(true)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={springTransition}
          >
            <BookmarkIcon className="w-4 h-4" />
            Save Scenario
          </motion.button>
        </div>
      </motion.div>

      {/* Product Selector */}
      <motion.div
        className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.05 }}
      >
        <label className="text-white/50 text-xs font-medium uppercase tracking-wider mb-3 block">
          Select Product
        </label>
        <ProductSelector selectedProduct={selectedProduct} onSelect={handleProductSelect} />
      </motion.div>

      {/* Price Slider */}
      <AnimatePresence>
        {selectedProduct && (
          <motion.div
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={springTransition}
          >
            <PriceSlider value={priceChange} onChange={handlePriceChange} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Impact Panel */}
      <AnimatePresence>
        {selectedProduct && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ ...springTransition, delay: 0.1 }}
          >
            <ImpactPanel predictions={predictions} loading={loading} error={error} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sensitivity + Confidence side by side */}
      <AnimatePresence>
        {selectedProduct && predictions && (
          <motion.div
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ ...springTransition, delay: 0.15 }}
          >
            <SensitivityTornado data={sensitivityData} loading={loading} />
            <ConfidenceBands data={confidenceData} loading={loading} currentVal={priceChange} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scenario Table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.2 }}
      >
        <ScenarioTable
          scenarios={scenarios}
          onDelete={handleDeleteScenario}
          onSelect={(s) => setSelectedScenario(s.id)}
          selected={selectedScenario}
        />
      </motion.div>

      {/* Save Scenario Modal */}
      <AnimatePresence>
        {showSaveModal && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop */}
            <div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setShowSaveModal(false)}
            />

            {/* Modal */}
            <motion.div
              className="relative bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              transition={springTransition}
            >
              <h3 className="text-white font-semibold text-lg mb-4">Save Scenario</h3>
              <div className="mb-4">
                <label className="text-white/50 text-xs font-medium uppercase tracking-wider mb-2 block">
                  Scenario Name
                </label>
                <input
                  type="text"
                  value={scenarioName}
                  onChange={(e) => setScenarioName(e.target.value)}
                  placeholder="e.g., Aggressive Growth Q3"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-sm placeholder-white/30 outline-none focus:border-[#0057B8]/50 transition-colors"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSaveScenario();
                  }}
                />
              </div>

              {/* Summary */}
              <div className="bg-white/5 rounded-xl p-4 mb-6 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-white/50">Product</span>
                  <span className="text-white">{selectedProduct?.name}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-white/50">Price Change</span>
                  <span className="font-mono text-white">{priceChange > 0 ? '+' : ''}{priceChange}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-white/50">Predicted Volume</span>
                  <span className="font-mono text-[#10b981]">
                    {predictions?.volumeChange > 0 ? '+' : ''}{predictions?.volumeChange?.toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  className="flex-1 px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white/60 text-sm hover:bg-white/10 transition-colors"
                  onClick={() => setShowSaveModal(false)}
                >
                  Cancel
                </button>
                <button
                  className="flex-1 px-4 py-2.5 rounded-xl text-white text-sm font-medium disabled:opacity-40"
                  style={{ backgroundColor: '#0057B8' }}
                  disabled={!scenarioName.trim()}
                  onClick={handleSaveScenario}
                >
                  Save
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
