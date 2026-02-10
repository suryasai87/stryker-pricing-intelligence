import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Cell, Legend,
} from 'recharts';
import {
  ExclamationTriangleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatPercent, formatNumber } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchPriceElasticity, fetchPriceElasticityDistribution } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const CLASSIFICATION_COLORS = {
  highly_inelastic: COLORS.success,
  inelastic: COLORS.successLight,
  unit_elastic: COLORS.accent,
  elastic: COLORS.warning,
  highly_elastic: COLORS.danger,
};

const CLASSIFICATION_LABELS = {
  highly_inelastic: 'Highly Inelastic',
  inelastic: 'Inelastic',
  unit_elastic: 'Unit Elastic',
  elastic: 'Elastic',
  highly_elastic: 'Highly Elastic',
};

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <GlassCard key={i} animate={false}>
            <LoadingShimmer width="60%" height="14px" className="mb-4" />
            <LoadingShimmer width="50%" height="28px" className="mb-3" />
            <LoadingShimmer width="40%" height="12px" />
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
      <GlassCard animate={false}>
        <LoadingShimmer width="40%" height="14px" className="mb-4" />
        <LoadingShimmer height="300px" rounded="rounded-xl" />
      </GlassCard>
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

function KPICard({ title, value, subtitle, color, index }) {
  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...springTransition, delay: index * 0.05 }}
    >
      <p className="text-white/40 text-xs font-medium uppercase tracking-wider">{title}</p>
      <p className="font-mono text-2xl font-bold mt-1" style={{ color }}>{value}</p>
      {subtitle && <p className="text-white/30 text-xs mt-0.5">{subtitle}</p>}
    </motion.div>
  );
}

function ElasticityHeatmap({ data }) {
  if (!data || !data.length) return null;

  const segments = data.length > 0 && data[0].segments ? Object.keys(data[0].segments) : [];

  const getHeatColor = (val) => {
    if (val === null || val === undefined) return COLORS.surface;
    // Inelastic (< -1) = green, elastic (> -1) = red, unit (-1) = yellow
    const absVal = Math.abs(val);
    if (absVal < 0.5) return COLORS.danger;
    if (absVal < 0.8) return COLORS.warning;
    if (absVal < 1.2) return COLORS.accent;
    if (absVal < 2.0) return COLORS.successLight;
    return COLORS.success;
  };

  return (
    <GlassCard animate={false}>
      <h3 className="text-white/70 text-sm font-semibold mb-4">Elasticity Heatmap by Product Family & Segment</h3>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-white/40 uppercase tracking-wider">Product Family</th>
              {segments.map((seg) => (
                <th key={seg} className="px-3 py-2 text-center text-xs font-medium text-white/40 uppercase tracking-wider whitespace-nowrap">
                  {seg}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {data.map((row, i) => (
              <tr key={i}>
                <td className="px-3 py-2 text-sm text-white/80 whitespace-nowrap">{row.product_family}</td>
                {segments.map((seg) => {
                  const val = row.segments?.[seg];
                  return (
                    <td key={seg} className="px-3 py-2 text-center">
                      <div
                        className="mx-auto w-14 h-8 rounded-md flex items-center justify-center text-xs font-mono font-medium"
                        style={{
                          backgroundColor: `${getHeatColor(val)}20`,
                          color: getHeatColor(val),
                        }}
                      >
                        {val != null ? val.toFixed(2) : '--'}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-4 mt-4 justify-center">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.success }} />
          <span className="text-white/40 text-xs">Highly Inelastic</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.accent }} />
          <span className="text-white/40 text-xs">Unit Elastic</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS.danger }} />
          <span className="text-white/40 text-xs">Highly Elastic</span>
        </div>
      </div>
    </GlassCard>
  );
}

export default function PriceElasticityPage() {
  const [data, setData] = useState(null);
  const [distribution, setDistribution] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({});
  const [filterOptions, setFilterOptions] = useState({
    businessUnits: [], segments: [], productFamilies: [], classifications: [], confidences: [],
  });
  const [sortConfig, setSortConfig] = useState({ key: 'elasticity_coefficient', dir: 'asc' });

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {};
      Object.entries(filters).forEach(([k, v]) => { if (v) params[k] = v; });
      const [elasticity, dist] = await Promise.all([
        fetchPriceElasticity(params),
        fetchPriceElasticityDistribution(params),
      ]);
      const rows = elasticity?.data || elasticity || [];
      setData(rows);
      setDistribution(dist);
      if (rows.length) {
        setFilterOptions({
          businessUnits: [...new Set(rows.map((r) => r.business_unit).filter(Boolean))].sort(),
          segments: [...new Set(rows.map((r) => r.segment).filter(Boolean))].sort(),
          productFamilies: [...new Set(rows.map((r) => r.product_family).filter(Boolean))].sort(),
          classifications: [...new Set(rows.map((r) => r.classification).filter(Boolean))].sort(),
          confidences: [...new Set(rows.map((r) => r.confidence).filter(Boolean))].sort(),
        });
      }
    } catch (err) {
      setError(err.message || 'Failed to load price elasticity data');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleFilterReset = () => { setFilters({}); };

  const sortedData = useMemo(() => {
    if (!data) return [];
    const rows = [...data];
    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? 0;
      const bv = b[sortConfig.key] ?? 0;
      if (typeof av === 'string') return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortConfig.dir === 'asc' ? av - bv : bv - av;
    });
    return rows;
  }, [data, sortConfig]);

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

  // Build histogram data from distribution or raw data
  const histogramData = useMemo(() => {
    if (distribution?.histogram) return distribution.histogram;
    if (!data || !data.length) return [];
    const buckets = {};
    data.forEach((r) => {
      const cls = r.classification || 'unknown';
      buckets[cls] = (buckets[cls] || 0) + 1;
    });
    return Object.entries(buckets).map(([cls, count]) => ({
      classification: cls,
      label: CLASSIFICATION_LABELS[cls] || cls,
      count,
      color: CLASSIFICATION_COLORS[cls] || COLORS.primary,
    }));
  }, [data, distribution]);

  // Build safe range data per product family
  const safeRangeData = useMemo(() => {
    if (distribution?.safe_ranges) return distribution.safe_ranges;
    if (!data || !data.length) return [];
    const families = {};
    data.forEach((r) => {
      const fam = r.product_family || 'Other';
      if (!families[fam]) families[fam] = { product_family: fam, safe_min: 0, safe_max: 0, count: 0 };
      families[fam].safe_min += r.safe_increase_min || 0;
      families[fam].safe_max += r.safe_increase_max || 0;
      families[fam].count += 1;
    });
    return Object.values(families).map((f) => ({
      product_family: f.product_family,
      safe_min: f.count ? f.safe_min / f.count : 0,
      safe_max: f.count ? f.safe_max / f.count : 0,
    }));
  }, [data, distribution]);

  // Build heatmap data
  const heatmapData = useMemo(() => {
    if (distribution?.heatmap) return distribution.heatmap;
    if (!data || !data.length) return [];
    const families = {};
    data.forEach((r) => {
      const fam = r.product_family || 'Other';
      const seg = r.segment || 'Other';
      if (!families[fam]) families[fam] = { product_family: fam, segments: {} };
      if (!families[fam].segments[seg]) families[fam].segments[seg] = [];
      families[fam].segments[seg].push(r.elasticity_coefficient || 0);
    });
    return Object.values(families).map((f) => ({
      product_family: f.product_family,
      segments: Object.fromEntries(
        Object.entries(f.segments).map(([seg, vals]) => [seg, vals.reduce((a, b) => a + b, 0) / vals.length])
      ),
    }));
  }, [data, distribution]);

  if (loading) return <div className="p-6"><LoadingSkeleton /></div>;
  if (error) return <div className="p-6"><ErrorState message={error} onRetry={fetchData} /></div>;

  const avgElasticity = data?.length
    ? (data.reduce((s, r) => s + (r.elasticity_coefficient || 0), 0) / data.length).toFixed(2)
    : '--';
  const inelasticPct = data?.length
    ? ((data.filter((r) => r.classification === 'highly_inelastic').length / data.length) * 100).toFixed(1)
    : '--';
  const elasticPct = data?.length
    ? ((data.filter((r) => ['elastic', 'highly_elastic'].includes(r.classification)).length / data.length) * 100).toFixed(1)
    : '--';
  const avgSafeIncrease = data?.length
    ? (data.reduce((s, r) => s + (r.safe_increase_max || 0), 0) / data.length).toFixed(1)
    : '--';

  const kpis = [
    { title: 'Avg Elasticity', value: avgElasticity, color: COLORS.primary, subtitle: 'Avg elasticity coefficient' },
    { title: 'Highly Inelastic', value: `${inelasticPct}%`, color: COLORS.success, subtitle: 'SKUs with low price sensitivity' },
    { title: 'Elastic SKUs', value: `${elasticPct}%`, color: COLORS.danger, subtitle: 'SKUs with high price sensitivity' },
    { title: 'Avg Safe Increase', value: `${avgSafeIncrease}%`, color: COLORS.accent, subtitle: 'Average safe price increase range' },
  ];

  const columns = [
    { key: 'sku', label: 'SKU' },
    { key: 'product_family', label: 'Product Family' },
    { key: 'segment', label: 'Segment' },
    { key: 'business_unit', label: 'Business Unit' },
    { key: 'classification', label: 'Classification' },
    { key: 'elasticity_coefficient', label: 'Elasticity', fmt: (v) => v?.toFixed(3) ?? '--' },
    { key: 'confidence', label: 'Confidence' },
    { key: 'safe_increase_min', label: 'Safe Min %', fmt: (v) => formatPercent(v) },
    { key: 'safe_increase_max', label: 'Safe Max %', fmt: (v) => formatPercent(v) },
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
        <h2 className="text-white text-xl font-bold">Price Elasticity Analysis</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Understand price sensitivity across products and identify safe pricing ranges
        </p>
      </motion.div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, i) => <KPICard key={kpi.title} {...kpi} index={i} />)}
      </div>

      {/* Filter Bar */}
      <motion.div
        className="flex flex-wrap items-center gap-3"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.1 }}
      >
        {[
          { key: 'business_unit', label: 'Business Unit', choices: filterOptions.businessUnits },
          { key: 'segment', label: 'Segment', choices: filterOptions.segments },
          { key: 'product_family', label: 'Product Family', choices: filterOptions.productFamilies },
          { key: 'classification', label: 'Classification', choices: filterOptions.classifications },
          { key: 'confidence', label: 'Confidence', choices: filterOptions.confidences },
        ].map((f) => (
          <select
            key={f.key}
            value={filters[f.key] || ''}
            onChange={(e) => handleFilterChange(f.key, e.target.value)}
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer min-w-[140px]"
          >
            <option value="" className="bg-[#1E293B]">{f.label}</option>
            {(f.choices || []).map((c) => (
              <option key={c} value={c} className="bg-[#1E293B]">{c}</option>
            ))}
          </select>
        ))}
        {Object.values(filters).some(Boolean) && (
          <motion.button
            className="px-3 py-2 rounded-xl text-xs font-medium text-white/50 hover:text-white bg-white/5 border border-white/10 hover:bg-white/10"
            onClick={handleFilterReset}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Clear All
          </motion.button>
        )}
      </motion.div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Histogram of Elasticity Coefficients */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Elasticity Distribution by Classification</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={histogramData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} />
              <XAxis
                dataKey="label"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                angle={-15}
                textAnchor="end"
                height={50}
              />
              <YAxis {...RECHARTS_THEME.tick} axisLine={RECHARTS_THEME.axisLine} tickLine={RECHARTS_THEME.tickLine} />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val) => [formatNumber(val), 'SKUs']}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {histogramData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.color || CLASSIFICATION_COLORS[entry.classification] || CHART_PALETTE[idx]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Safe Range Stacked Bar per Product Family */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Safe Price Increase Range by Product Family</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={safeRangeData} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 80 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} horizontal={false} />
              <XAxis
                type="number"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                label={{ value: 'Safe Increase %', position: 'bottom', fill: COLORS.textMuted, fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="product_family"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                width={75}
                tick={{ fill: COLORS.textSecondary, fontSize: 10 }}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val) => [formatPercent(val), '']}
              />
              <Legend wrapperStyle={RECHARTS_THEME.legend.wrapperStyle} />
              <Bar dataKey="safe_min" stackId="safe" name="Min Safe" fill={COLORS.success} fillOpacity={0.6} radius={[0, 0, 0, 0]} />
              <Bar dataKey="safe_max" stackId="safe" name="Max Safe" fill={COLORS.successLight} fillOpacity={0.6} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Heatmap */}
      <ElasticityHeatmap data={heatmapData} />

      {/* Data Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Elasticity Details ({sortedData.length} SKUs)
          </h3>
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
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {sortedData.map((row, i) => (
                <tr key={i} className="hover:bg-white/5 transition-colors">
                  {columns.map((col) => (
                    <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                      {col.key === 'classification' ? (
                        <span
                          className="px-2 py-0.5 rounded-full text-xs font-medium"
                          style={{
                            backgroundColor: `${CLASSIFICATION_COLORS[row.classification] || COLORS.primary}20`,
                            color: CLASSIFICATION_COLORS[row.classification] || COLORS.primary,
                          }}
                        >
                          {CLASSIFICATION_LABELS[row.classification] || row.classification || '--'}
                        </span>
                      ) : col.key === 'safe_increase_min' || col.key === 'safe_increase_max' ? (
                        <span className="font-mono font-semibold text-emerald-400">
                          {col.fmt(row[col.key])}
                        </span>
                      ) : col.fmt ? col.fmt(row[col.key]) : (row[col.key] ?? '--')}
                    </td>
                  ))}
                </tr>
              ))}
              {sortedData.length === 0 && (
                <tr>
                  <td colSpan={columns.length} className="px-4 py-12 text-center text-white/30 text-sm">
                    No elasticity data found matching current filters.
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
