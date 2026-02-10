import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from 'recharts';
import {
  ExclamationTriangleIcon,
  ArrowDownTrayIcon,
  PlayIcon,
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatCurrency, formatPercent, formatNumber, formatCompact } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchPrecomputedUplift, runUpliftSimulation } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <GlassCard animate={false}>
        <LoadingShimmer width="60%" height="14px" className="mb-4" />
        <div className="flex gap-4">
          <LoadingShimmer width="150px" height="40px" />
          <LoadingShimmer width="150px" height="40px" />
          <LoadingShimmer width="100px" height="40px" />
        </div>
      </GlassCard>
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {Array.from({ length: 7 }).map((_, i) => (
          <GlassCard key={i} animate={false} padding="p-4">
            <LoadingShimmer width="80%" height="12px" className="mb-2" />
            <LoadingShimmer width="60%" height="24px" />
          </GlassCard>
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlassCard animate={false}>
          <LoadingShimmer width="40%" height="14px" className="mb-4" />
          <LoadingShimmer height="300px" rounded="rounded-xl" />
        </GlassCard>
        <GlassCard animate={false}>
          <LoadingShimmer width="40%" height="14px" className="mb-4" />
          <LoadingShimmer height="300px" rounded="rounded-xl" />
        </GlassCard>
      </div>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <GlassCard className="text-center py-16">
      <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <ExclamationTriangleIcon className="w-8 h-8 text-rose-400" />
      </div>
      <h3 className="text-white font-semibold text-lg mb-2">Failed to Load Data</h3>
      <p className="text-white/50 text-sm mb-6 max-w-md mx-auto">{message}</p>
      <motion.button
        className="px-6 py-2.5 rounded-xl text-white text-sm font-medium bg-[#0057B8]"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.97 }}
        transition={springTransition}
        onClick={onRetry}
      >
        Retry
      </motion.button>
    </GlassCard>
  );
}

function exportCSV(data, filename) {
  if (!data || !data.length) return;
  const keys = Object.keys(data[0]);
  const csv = [
    keys.join(','),
    ...data.map((row) => keys.map((k) => JSON.stringify(row[k] ?? '')).join(',')),
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function UpliftSimulatorPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: 'rank', dir: 'asc' });

  // Simulation controls
  const [targetUplift, setTargetUplift] = useState(1.0);
  const [maxPerSKU, setMaxPerSKU] = useState(5.0);
  const [excludeCountries, setExcludeCountries] = useState('');
  const [excludeSegments, setExcludeSegments] = useState('');
  const [excludeSKUs, setExcludeSKUs] = useState('');

  const loadDefault = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPrecomputedUplift(1.0);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to load simulation data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadDefault(); }, [loadDefault]);

  const handleRun = async () => {
    setSimulating(true);
    setError(null);
    try {
      const params = {
        target_uplift_pct: targetUplift,
        max_per_sku_increase: maxPerSKU,
      };
      if (excludeCountries.trim()) params.exclude_countries = excludeCountries.split(',').map((s) => s.trim());
      if (excludeSegments.trim()) params.exclude_segments = excludeSegments.split(',').map((s) => s.trim());
      if (excludeSKUs.trim()) params.exclude_skus = excludeSKUs.split(',').map((s) => s.trim());
      const data = await runUpliftSimulation(params);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Simulation failed');
    } finally {
      setSimulating(false);
    }
  };

  const handleReset = () => {
    setTargetUplift(1.0);
    setMaxPerSKU(5.0);
    setExcludeCountries('');
    setExcludeSegments('');
    setExcludeSKUs('');
    loadDefault();
  };

  const recommendations = useMemo(() => {
    if (!result?.recommendations) return [];
    const rows = [...result.recommendations];
    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? 0;
      const bv = b[sortConfig.key] ?? 0;
      if (typeof av === 'string') return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortConfig.dir === 'asc' ? av - bv : bv - av;
    });
    return rows;
  }, [result, sortConfig]);

  const cumulativeData = useMemo(() => {
    if (!result?.cumulative_curve) return [];
    return result.cumulative_curve;
  }, [result]);

  const waterfallData = useMemo(() => {
    if (!result?.waterfall) return [];
    return result.waterfall.slice(0, 20);
  }, [result]);

  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      dir: prev.key === key && prev.dir === 'desc' ? 'asc' : 'desc',
    }));
  };

  const SortIcon = ({ colKey }) => {
    if (sortConfig.key !== colKey) return null;
    return sortConfig.dir === 'asc'
      ? <ChevronUpIcon className="w-3 h-3 inline ml-1" />
      : <ChevronDownIcon className="w-3 h-3 inline ml-1" />;
  };

  if (loading) return <div className="p-6"><LoadingSkeleton /></div>;
  if (error && !result) return <div className="p-6"><ErrorState message={error} onRetry={loadDefault} /></div>;

  const summary = result?.summary || {};
  const summaryCards = [
    { title: 'Target', value: formatPercent(summary.target_pct), color: COLORS.primary },
    { title: 'Achieved', value: formatPercent(summary.achieved_pct), color: summary.achieved_pct >= summary.target_pct ? COLORS.success : COLORS.warning },
    { title: 'Actions Needed', value: formatNumber(summary.actions_needed), color: COLORS.accent },
    { title: 'Net Rev Impact', value: formatCurrency(summary.net_revenue_impact, { compact: true }), color: COLORS.success },
    { title: 'SKUs Affected', value: formatNumber(summary.skus_affected), color: COLORS.info },
    { title: 'Customers Affected', value: formatNumber(summary.customers_affected), color: COLORS.primaryLight },
    { title: 'Avg Vol Impact', value: formatPercent(summary.avg_volume_impact), color: COLORS.danger },
  ];

  const columns = [
    { key: 'rank', label: '#' },
    { key: 'sku', label: 'SKU' },
    { key: 'product_family', label: 'Product Family' },
    { key: 'segment', label: 'Segment' },
    { key: 'country', label: 'Country' },
    { key: 'current_price', label: 'Current Price', fmt: (v) => formatCurrency(v) },
    { key: 'recommended_price', label: 'Rec. Price', fmt: (v) => formatCurrency(v) },
    { key: 'increase_pct', label: 'Increase %', fmt: (v) => formatPercent(v) },
    { key: 'revenue_impact', label: 'Rev Impact', fmt: (v) => formatCurrency(v, { compact: true }) },
    { key: 'volume_impact_pct', label: 'Vol Impact', fmt: (v) => formatPercent(v) },
    { key: 'within_target', label: 'In Target' },
  ];

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={springTransition}
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={springTransition}>
        <h2 className="text-white text-xl font-bold">Uplift Simulator</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Model price uplift scenarios and view cumulative revenue impact
        </p>
      </motion.div>

      {/* Simulation Controls */}
      <GlassCard animate={false}>
        <h3 className="text-white/70 text-sm font-semibold mb-4">Simulation Parameters</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Target Uplift %</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="20"
              value={targetUplift}
              onChange={(e) => setTargetUplift(parseFloat(e.target.value) || 0)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Max Per-SKU Increase %</label>
            <input
              type="number"
              step="0.5"
              min="0"
              max="50"
              value={maxPerSKU}
              onChange={(e) => setMaxPerSKU(parseFloat(e.target.value) || 0)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Exclude Countries</label>
            <input
              type="text"
              placeholder="US, UK, DE..."
              value={excludeCountries}
              onChange={(e) => setExcludeCountries(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Exclude Segments</label>
            <input
              type="text"
              placeholder="Spine, Trauma..."
              value={excludeSegments}
              onChange={(e) => setExcludeSegments(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Exclude SKUs</label>
            <input
              type="text"
              placeholder="SKU-001, SKU-002..."
              value={excludeSKUs}
              onChange={(e) => setExcludeSKUs(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <motion.button
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-medium bg-[#0057B8]"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            transition={springTransition}
            onClick={handleRun}
            disabled={simulating}
          >
            {simulating ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
              />
            ) : (
              <PlayIcon className="w-4 h-4" />
            )}
            {simulating ? 'Running...' : 'Run Simulation'}
          </motion.button>
          <motion.button
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-white/60 text-sm font-medium bg-white/5 border border-white/10 hover:text-white hover:bg-white/10"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleReset}
          >
            <ArrowPathIcon className="w-4 h-4" />
            Reset
          </motion.button>
        </div>
      </GlassCard>

      {/* Error Banner */}
      <AnimatePresence>
        {error && result && (
          <motion.div
            className="bg-white/5 border border-rose-500/30 rounded-2xl p-4 text-center"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <p className="text-rose-400 text-sm">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {summaryCards.map((card, i) => (
          <motion.div
            key={card.title}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springTransition, delay: i * 0.03 }}
          >
            <p className="text-white/40 text-[10px] font-medium uppercase tracking-wider">{card.title}</p>
            <p className="font-mono text-lg font-bold mt-1" style={{ color: card.color }}>{card.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Cumulative Uplift Line Chart */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Cumulative Uplift Curve</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={cumulativeData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} />
              <XAxis
                dataKey="rank"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                label={{ value: 'Action Rank', position: 'bottom', fill: COLORS.textMuted, fontSize: 11 }}
              />
              <YAxis
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                label={{ value: 'Cumulative Uplift %', angle: -90, position: 'insideLeft', fill: COLORS.textMuted, fontSize: 11 }}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val) => [formatPercent(val), 'Cumulative Uplift']}
              />
              {summary.target_pct && (
                <ReferenceLine
                  y={summary.target_pct}
                  stroke={COLORS.accent}
                  strokeDasharray="6 3"
                  label={{ value: `Target: ${formatPercent(summary.target_pct)}`, fill: COLORS.accent, fontSize: 11, position: 'right' }}
                />
              )}
              <Line
                type="monotone"
                dataKey="cumulative_uplift_pct"
                stroke={COLORS.primary}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, fill: COLORS.primary, stroke: COLORS.background, strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Waterfall Chart - Revenue Impact */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Revenue Impact (Top 20 Actions)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={waterfallData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} />
              <XAxis
                dataKey="sku"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                angle={-45}
                textAnchor="end"
                height={60}
                tick={{ fill: COLORS.textMuted, fontSize: 9 }}
              />
              <YAxis
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val) => [formatCurrency(val, { compact: true }), 'Revenue Impact']}
              />
              <Bar dataKey="revenue_impact" radius={[4, 4, 0, 0]}>
                {waterfallData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={(entry.revenue_impact || 0) >= 0 ? COLORS.success : COLORS.danger}
                    fillOpacity={0.7}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Recommendations Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="flex items-center justify-between p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Recommendations ({recommendations.length} actions)
          </h3>
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => exportCSV(recommendations, 'uplift-recommendations.csv')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ArrowDownTrayIcon className="w-4 h-4" />
            Export CSV
          </motion.button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-white/5">
                {columns.map((col) => (
                  <th
                    key={col.key}
                    className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider cursor-pointer hover:text-white/70 whitespace-nowrap"
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    <SortIcon colKey={col.key} />
                  </th>
                ))}
                <th className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider w-8" />
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {recommendations.map((row, i) => (
                <React.Fragment key={i}>
                  <tr
                    className="hover:bg-white/5 cursor-pointer transition-colors"
                    onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                  >
                    {columns.map((col) => (
                      <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                        {col.key === 'within_target' ? (
                          <span className={clsx(
                            'px-2 py-0.5 rounded-full text-xs font-medium',
                            row.within_target
                              ? 'bg-emerald-500/20 text-emerald-400'
                              : 'bg-white/5 text-white/30'
                          )}>
                            {row.within_target ? 'Yes' : 'No'}
                          </span>
                        ) : col.fmt ? col.fmt(row[col.key]) : (row[col.key] ?? '--')}
                      </td>
                    ))}
                    <td className="px-4 py-3">
                      <ChevronRightIcon className={clsx(
                        'w-4 h-4 text-white/30 transition-transform',
                        expandedRow === i && 'rotate-90'
                      )} />
                    </td>
                  </tr>
                  <AnimatePresence>
                    {expandedRow === i && row.rationale && (
                      <tr>
                        <td colSpan={columns.length + 1}>
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="px-6 py-4 bg-white/[0.02] border-l-2 border-[#0057B8]/50"
                          >
                            <p className="text-white/40 text-xs font-medium uppercase tracking-wider mb-1">Rationale</p>
                            <p className="text-white/70 text-sm">{row.rationale}</p>
                          </motion.div>
                        </td>
                      </tr>
                    )}
                  </AnimatePresence>
                </React.Fragment>
              ))}
              {recommendations.length === 0 && (
                <tr>
                  <td colSpan={columns.length + 1} className="px-4 py-12 text-center text-white/30 text-sm">
                    No recommendations available. Run a simulation to generate results.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </motion.div>
  );
}
