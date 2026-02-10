import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ZAxis, Legend,
} from 'recharts';
import {
  ExclamationTriangleIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatCurrency, formatPercent, formatNumber, formatCompact } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchDiscountOutliers, fetchDiscountOutliersSummary } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const SEVERITY_COLORS = {
  severe: COLORS.danger,
  high: COLORS.warning,
  moderate: COLORS.accent,
  low: COLORS.info,
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
        <LoadingShimmer width="30%" height="14px" className="mb-4" />
        <LoadingShimmer height="400px" rounded="rounded-xl" />
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

function FilterBar({ filters, options, onChange, onReset }) {
  return (
    <motion.div
      className="flex flex-wrap items-center gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...springTransition, delay: 0.1 }}
    >
      {[
        { key: 'business_unit', label: 'Business Unit', choices: options.businessUnits },
        { key: 'segment', label: 'Segment', choices: options.segments },
        { key: 'country', label: 'Country', choices: options.countries },
        { key: 'severity', label: 'Severity', choices: options.severities },
        { key: 'product_family', label: 'Product Family', choices: options.productFamilies },
      ].map((f) => (
        <select
          key={f.key}
          value={filters[f.key] || ''}
          onChange={(e) => onChange(f.key, e.target.value)}
          className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer min-w-[140px]"
        >
          <option value="" className="bg-[#1E293B]">{f.label}</option>
          {(f.choices || []).map((c) => (
            <option key={c} value={c} className="bg-[#1E293B]">{c}</option>
          ))}
        </select>
      ))}

      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
        <input
          type="text"
          placeholder="Search rep name..."
          value={filters.rep_search || ''}
          onChange={(e) => onChange('rep_search', e.target.value)}
          className="bg-white/5 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50 w-48"
        />
      </div>

      {Object.values(filters).some(Boolean) && (
        <motion.button
          className="px-3 py-2 rounded-xl text-xs font-medium text-white/50 hover:text-white bg-white/5 border border-white/10 hover:bg-white/10"
          onClick={onReset}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Clear All
        </motion.button>
      )}
    </motion.div>
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

function ScatterTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div style={RECHARTS_THEME.tooltip.contentStyle}>
      <p className="text-white font-semibold text-sm mb-1">{d.rep_name}</p>
      <p className="text-white/70 text-xs">Peer Avg: {formatPercent(d.peer_avg_discount)}</p>
      <p className="text-white/70 text-xs">Rep Avg: {formatPercent(d.rep_avg_discount)}</p>
      <p className="text-white/70 text-xs">Revenue: {formatCurrency(d.revenue, { compact: true })}</p>
      <p className="text-white/70 text-xs">Severity: {d.severity}</p>
    </div>
  );
}

function DrillDownPanel({ row, onClose }) {
  if (!row) return null;
  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 40 }}
      transition={springTransition}
      className="fixed right-0 top-0 h-full w-full max-w-md z-50 bg-[#1E293B]/95 backdrop-blur-xl border-l border-white/10 shadow-2xl overflow-y-auto"
    >
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white font-bold text-lg">Outlier Detail</h3>
          <motion.button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <XMarkIcon className="w-5 h-5 text-white/60" />
          </motion.button>
        </div>

        <div className="space-y-4">
          {[
            ['Rep Name', row.rep_name],
            ['Business Unit', row.business_unit],
            ['Product Family', row.product_family],
            ['Segment', row.segment],
            ['Country', row.country],
            ['Severity', row.severity],
            ['Rep Avg Discount', formatPercent(row.rep_avg_discount)],
            ['Peer Avg Discount', formatPercent(row.peer_avg_discount)],
            ['Deviation', formatPercent(row.deviation)],
            ['Revenue', formatCurrency(row.revenue)],
            ['Potential Recovery', formatCurrency(row.potential_recovery)],
            ['Deal Count', formatNumber(row.deal_count)],
          ].map(([label, val]) => (
            <div key={label} className="flex justify-between py-2 border-b border-white/5">
              <span className="text-white/40 text-sm">{label}</span>
              <span className="text-white text-sm font-medium">{val || '--'}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
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

export default function DiscountOutliersPage() {
  const [data, setData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({});
  const [filterOptions, setFilterOptions] = useState({
    businessUnits: [], segments: [], countries: [], severities: [], productFamilies: [],
  });
  const [sortConfig, setSortConfig] = useState({ key: 'potential_recovery', dir: 'desc' });
  const [selectedRow, setSelectedRow] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {};
      Object.entries(filters).forEach(([k, v]) => { if (v) params[k] = v; });
      const [outliers, summ] = await Promise.all([
        fetchDiscountOutliers(params),
        fetchDiscountOutliersSummary(params),
      ]);
      setData(outliers?.data || outliers || []);
      setSummary(summ);
      // Extract filter options from data
      const rows = outliers?.data || outliers || [];
      if (rows.length) {
        setFilterOptions({
          businessUnits: [...new Set(rows.map((r) => r.business_unit).filter(Boolean))].sort(),
          segments: [...new Set(rows.map((r) => r.segment).filter(Boolean))].sort(),
          countries: [...new Set(rows.map((r) => r.country).filter(Boolean))].sort(),
          severities: [...new Set(rows.map((r) => r.severity).filter(Boolean))].sort(),
          productFamilies: [...new Set(rows.map((r) => r.product_family).filter(Boolean))].sort(),
        });
      }
    } catch (err) {
      setError(err.message || 'Failed to load discount outlier data');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleFilterChange = useCallback((key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleFilterReset = useCallback(() => { setFilters({}); }, []);

  const filteredData = useMemo(() => {
    if (!data) return [];
    let rows = [...data];
    if (filters.rep_search) {
      const q = filters.rep_search.toLowerCase();
      rows = rows.filter((r) => (r.rep_name || '').toLowerCase().includes(q));
    }
    return rows;
  }, [data, filters.rep_search]);

  const sortedData = useMemo(() => {
    const rows = [...filteredData];
    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? 0;
      const bv = b[sortConfig.key] ?? 0;
      if (typeof av === 'string') return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortConfig.dir === 'asc' ? av - bv : bv - av;
    });
    return rows;
  }, [filteredData, sortConfig]);

  const topRecoveryReps = useMemo(() => {
    return [...filteredData]
      .sort((a, b) => (b.potential_recovery || 0) - (a.potential_recovery || 0))
      .slice(0, 10);
  }, [filteredData]);

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
  if (error) return <div className="p-6"><ErrorState message={error} onRetry={fetchData} /></div>;

  const kpis = [
    {
      title: 'Total Outliers',
      value: formatNumber(summary?.total_outliers ?? filteredData.length),
      color: COLORS.primary,
      subtitle: 'Discount outliers detected',
    },
    {
      title: 'Severe',
      value: formatNumber(summary?.severe_count ?? filteredData.filter((r) => r.severity === 'severe').length),
      color: COLORS.danger,
      subtitle: 'Require immediate attention',
    },
    {
      title: 'Potential Recovery',
      value: formatCurrency(summary?.total_recovery ?? filteredData.reduce((s, r) => s + (r.potential_recovery || 0), 0), { compact: true }),
      color: COLORS.success,
      subtitle: 'Revenue recovery opportunity',
    },
    {
      title: 'Top BU Affected',
      value: summary?.top_bu || (filteredData.length ? filteredData[0]?.business_unit : '--'),
      color: COLORS.accent,
      subtitle: 'Most outliers by business unit',
    },
  ];

  const columns = [
    { key: 'rep_name', label: 'Rep Name' },
    { key: 'business_unit', label: 'Business Unit' },
    { key: 'product_family', label: 'Product Family' },
    { key: 'segment', label: 'Segment' },
    { key: 'country', label: 'Country' },
    { key: 'severity', label: 'Severity' },
    { key: 'rep_avg_discount', label: 'Rep Avg Discount', fmt: (v) => formatPercent(v) },
    { key: 'peer_avg_discount', label: 'Peer Avg Discount', fmt: (v) => formatPercent(v) },
    { key: 'deviation', label: 'Deviation', fmt: (v) => formatPercent(v) },
    { key: 'potential_recovery', label: 'Recovery $', fmt: (v) => formatCurrency(v, { compact: true }) },
    { key: 'deal_count', label: 'Deals', fmt: (v) => formatNumber(v) },
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
        <h2 className="text-white text-xl font-bold">Discount Outliers</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Identify reps with discounting patterns deviating significantly from peers
        </p>
      </motion.div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, i) => <KPICard key={kpi.title} {...kpi} index={i} />)}
      </div>

      {/* Filter Bar */}
      <FilterBar filters={filters} options={filterOptions} onChange={handleFilterChange} onReset={handleFilterReset} />

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Scatter Chart */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Rep vs Peer Discount Distribution</h3>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} />
              <XAxis
                dataKey="peer_avg_discount"
                name="Peer Avg Discount"
                type="number"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                label={{ value: 'Peer Avg Discount %', position: 'bottom', fill: COLORS.textMuted, fontSize: 11 }}
              />
              <YAxis
                dataKey="rep_avg_discount"
                name="Rep Avg Discount"
                type="number"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                label={{ value: 'Rep Avg Discount %', angle: -90, position: 'insideLeft', fill: COLORS.textMuted, fontSize: 11 }}
              />
              <ZAxis dataKey="revenue" range={[40, 400]} name="Revenue" />
              <RechartsTooltip content={<ScatterTooltip />} />
              <Scatter data={filteredData} onClick={(entry) => setSelectedRow(entry)}>
                {filteredData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={SEVERITY_COLORS[entry.severity] || COLORS.primary}
                    fillOpacity={0.7}
                    stroke={SEVERITY_COLORS[entry.severity] || COLORS.primary}
                    strokeWidth={1}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Top 10 Recovery Bar */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Top 10 Reps by Recovery $</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={topRecoveryReps} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 80 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} horizontal={false} />
              <XAxis type="number" {...RECHARTS_THEME.tick} axisLine={RECHARTS_THEME.axisLine} tickLine={RECHARTS_THEME.tickLine} />
              <YAxis
                type="category"
                dataKey="rep_name"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                width={75}
                tick={{ fill: COLORS.textSecondary, fontSize: 10 }}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val) => [formatCurrency(val, { compact: true }), 'Recovery']}
              />
              <Bar dataKey="potential_recovery" radius={[0, 6, 6, 0]}>
                {topRecoveryReps.map((entry, idx) => (
                  <Cell key={idx} fill={CHART_PALETTE[idx % CHART_PALETTE.length]} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Data Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="flex items-center justify-between p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Outlier Details ({sortedData.length} records)
          </h3>
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => exportCSV(sortedData, 'discount-outliers.csv')}
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
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {sortedData.map((row, i) => (
                <tr
                  key={i}
                  className="hover:bg-white/5 cursor-pointer transition-colors"
                  onClick={() => setSelectedRow(row)}
                >
                  {columns.map((col) => (
                    <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                      {col.key === 'severity' ? (
                        <span
                          className="px-2 py-0.5 rounded-full text-xs font-medium"
                          style={{
                            backgroundColor: `${SEVERITY_COLORS[row.severity] || COLORS.primary}20`,
                            color: SEVERITY_COLORS[row.severity] || COLORS.primary,
                          }}
                        >
                          {row.severity}
                        </span>
                      ) : col.fmt ? col.fmt(row[col.key]) : (row[col.key] ?? '--')}
                    </td>
                  ))}
                </tr>
              ))}
              {sortedData.length === 0 && (
                <tr>
                  <td colSpan={columns.length} className="px-4 py-12 text-center text-white/30 text-sm">
                    No outliers found matching current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Drill-Down Panel */}
      <AnimatePresence>
        {selectedRow && <DrillDownPanel row={selectedRow} onClose={() => setSelectedRow(null)} />}
      </AnimatePresence>
    </motion.div>
  );
}
