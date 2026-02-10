import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSearchParams } from 'react-router-dom';
import {
  ExclamationTriangleIcon,
  ArrowDownTrayIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronRightIcon,
  XMarkIcon,
  AdjustmentsHorizontalIcon,
  ClipboardDocumentIcon,
  DocumentArrowDownIcon,
  TableCellsIcon,
  FunnelIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatCurrency, formatPercent, formatNumber, formatCompact, formatDelta } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchTop100PriceChanges, fetchTop100FilterOptions } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const RISK_COLORS = {
  low: COLORS.success,
  medium: COLORS.accent,
  high: COLORS.warning,
  critical: COLORS.danger,
};

const DEFAULT_VISIBLE_COLUMNS = [
  'rank', 'action_summary', 'sku', 'product_family', 'segment', 'country',
  'current_price', 'recommended_price', 'change_pct', 'expected_rev_gain',
  'expected_margin_gain', 'risk_level',
];

const ALL_COLUMNS = [
  { key: 'rank', label: '#', alwaysVisible: true },
  { key: 'action_summary', label: 'Action', alwaysVisible: true },
  { key: 'sku', label: 'SKU' },
  { key: 'product_name', label: 'Product Name' },
  { key: 'product_family', label: 'Product Family' },
  { key: 'business_unit', label: 'Business Unit' },
  { key: 'segment', label: 'Segment' },
  { key: 'country', label: 'Country' },
  { key: 'rep_name', label: 'Rep' },
  { key: 'current_price', label: 'Current Price', fmt: (v) => formatCurrency(v) },
  { key: 'recommended_price', label: 'Rec. Price', fmt: (v) => formatCurrency(v) },
  { key: 'change_pct', label: 'Change %', fmt: (v) => formatPercent(v) },
  { key: 'expected_rev_gain', label: 'Rev Gain', fmt: (v) => formatCurrency(v, { compact: true }) },
  { key: 'expected_margin_gain', label: 'Margin $', fmt: (v) => formatCurrency(v, { compact: true }) },
  { key: 'risk_level', label: 'Risk' },
  { key: 'confidence_score', label: 'Confidence', fmt: (v) => formatPercent(v) },
  { key: 'elasticity', label: 'Elasticity', fmt: (v) => v?.toFixed(3) ?? '--' },
  { key: 'volume_impact', label: 'Vol Impact', fmt: (v) => formatPercent(v) },
  { key: 'customer_count', label: 'Customers', fmt: (v) => formatNumber(v) },
];

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
      <GlassCard animate={false}>
        <LoadingShimmer width="100%" height="50px" className="mb-4" />
      </GlassCard>
      <GlassCard animate={false}>
        <LoadingShimmer width="30%" height="14px" className="mb-4" />
        <LoadingShimmer height="500px" rounded="rounded-xl" />
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

function exportData(data, format) {
  if (!data || !data.length) return;
  const keys = Object.keys(data[0]);

  if (format === 'copy') {
    const text = [
      keys.join('\t'),
      ...data.map((row) => keys.map((k) => row[k] ?? '').join('\t')),
    ].join('\n');
    navigator.clipboard.writeText(text).catch(() => {});
    return;
  }

  const csv = [
    keys.join(','),
    ...data.map((row) => keys.map((k) => JSON.stringify(row[k] ?? '')).join(',')),
  ].join('\n');

  if (format === 'excel') {
    // For .xlsx, we export as CSV with .xls extension (basic Excel compat)
    const blob = new Blob([csv], { type: 'application/vnd.ms-excel' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'top100-price-changes.xls';
    a.click();
    URL.revokeObjectURL(url);
    return;
  }

  // CSV
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'top100-price-changes.csv';
  a.click();
  URL.revokeObjectURL(url);
}

export default function Top100ChangesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [data, setData] = useState(null);
  const [filterOpts, setFilterOpts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: 'rank', dir: 'asc' });
  const [visibleColumns, setVisibleColumns] = useState(new Set(DEFAULT_VISIBLE_COLUMNS));
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [repSearch, setRepSearch] = useState(searchParams.get('rep') || '');

  // Read initial filters from URL
  const filters = useMemo(() => ({
    country: searchParams.getAll('country'),
    product_family: searchParams.get('product_family') || '',
    segment: searchParams.get('segment') || '',
    risk_level: searchParams.get('risk') || '',
    business_unit: searchParams.get('bu') || '',
  }), [searchParams]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [changes, opts] = await Promise.all([
        fetchTop100PriceChanges(),
        fetchTop100FilterOptions(),
      ]);
      setData(changes?.data || changes || []);
      setFilterOpts(opts);
    } catch (err) {
      setError(err.message || 'Failed to load top 100 price changes');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const updateFilter = useCallback((key, value) => {
    const newParams = new URLSearchParams(searchParams);
    if (key === 'country') {
      // Toggle chip
      const current = newParams.getAll('country');
      if (current.includes(value)) {
        newParams.delete('country');
        current.filter((c) => c !== value).forEach((c) => newParams.append('country', c));
      } else {
        newParams.append('country', value);
      }
    } else if (value) {
      newParams.set(key, value);
    } else {
      newParams.delete(key);
    }
    setSearchParams(newParams);
  }, [searchParams, setSearchParams]);

  const clearAllFilters = useCallback(() => {
    setSearchParams(new URLSearchParams());
    setRepSearch('');
  }, [setSearchParams]);

  const hasActiveFilters = useMemo(() => {
    return filters.country.length > 0 || filters.product_family || filters.segment ||
           filters.risk_level || filters.business_unit || repSearch;
  }, [filters, repSearch]);

  // Client-side filtering (only 100 rows)
  const filteredData = useMemo(() => {
    if (!data) return [];
    let rows = [...data];

    if (filters.country.length) {
      rows = rows.filter((r) => filters.country.includes(r.country));
    }
    if (filters.product_family) {
      rows = rows.filter((r) => r.product_family === filters.product_family);
    }
    if (filters.segment) {
      rows = rows.filter((r) => r.segment === filters.segment);
    }
    if (filters.risk_level) {
      rows = rows.filter((r) => r.risk_level === filters.risk_level);
    }
    if (filters.business_unit) {
      rows = rows.filter((r) => r.business_unit === filters.business_unit);
    }
    if (repSearch.trim()) {
      const q = repSearch.toLowerCase();
      rows = rows.filter((r) => (r.rep_name || '').toLowerCase().includes(q));
    }

    return rows;
  }, [data, filters, repSearch]);

  // Client-side sorting
  const sortedData = useMemo(() => {
    const rows = [...filteredData];
    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? '';
      const bv = b[sortConfig.key] ?? '';
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortConfig.dir === 'asc' ? (av || 0) - (bv || 0) : (bv || 0) - (av || 0);
    });
    return rows;
  }, [filteredData, sortConfig]);

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

  const toggleColumn = (key) => {
    setVisibleColumns((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  // Derive unique filter values from data
  const derivedOptions = useMemo(() => {
    if (!data) return {};
    return {
      countries: [...new Set(data.map((r) => r.country).filter(Boolean))].sort(),
      productFamilies: [...new Set(data.map((r) => r.product_family).filter(Boolean))].sort(),
      segments: [...new Set(data.map((r) => r.segment).filter(Boolean))].sort(),
      riskLevels: [...new Set(data.map((r) => r.risk_level).filter(Boolean))].sort(),
      businessUnits: [...new Set(data.map((r) => r.business_unit).filter(Boolean))].sort(),
    };
  }, [data]);

  const options = filterOpts || derivedOptions;

  if (loading) return <div className="p-6"><LoadingSkeleton /></div>;
  if (error) return <div className="p-6"><ErrorState message={error} onRetry={fetchData} /></div>;

  // Aggregate KPIs
  const totalActions = filteredData.length;
  const totalRevGain = filteredData.reduce((s, r) => s + (r.expected_rev_gain || 0), 0);
  const totalMarginGain = filteredData.reduce((s, r) => s + (r.expected_margin_gain || 0), 0);
  const avgRisk = (() => {
    const riskMap = { low: 1, medium: 2, high: 3, critical: 4 };
    const sum = filteredData.reduce((s, r) => s + (riskMap[r.risk_level] || 0), 0);
    const avg = filteredData.length ? sum / filteredData.length : 0;
    if (avg <= 1.5) return 'Low';
    if (avg <= 2.5) return 'Medium';
    if (avg <= 3.5) return 'High';
    return 'Critical';
  })();

  const kpis = [
    { title: 'Total Actions', value: formatNumber(totalActions), color: COLORS.primary, subtitle: `of ${data?.length || 100} recommended` },
    { title: 'Expected Rev Gain', value: formatCurrency(totalRevGain, { compact: true }), color: COLORS.success, subtitle: 'Projected revenue uplift' },
    { title: 'Expected Margin $', value: formatCurrency(totalMarginGain, { compact: true }), color: COLORS.accent, subtitle: 'Projected margin improvement' },
    { title: 'Avg Risk Level', value: avgRisk, color: RISK_COLORS[avgRisk.toLowerCase()] || COLORS.info, subtitle: 'Across filtered actions' },
  ];

  const activeColumns = ALL_COLUMNS.filter((col) => col.alwaysVisible || visibleColumns.has(col.key));

  const lastUpdated = data?.length && data[0].last_updated
    ? new Date(data[0].last_updated).toLocaleDateString()
    : new Date().toLocaleDateString();

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={springTransition}
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={springTransition}>
        <h2 className="text-white text-xl font-bold">Top 100 Recommended Price Changes</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Highest-impact pricing actions ranked by expected value -- Last updated: {lastUpdated}
        </p>
      </motion.div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, i) => (
          <motion.div
            key={kpi.title}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springTransition, delay: i * 0.05 }}
          >
            <p className="text-white/40 text-xs font-medium uppercase tracking-wider">{kpi.title}</p>
            <p className="font-mono text-2xl font-bold mt-1" style={{ color: kpi.color }}>{kpi.value}</p>
            {kpi.subtitle && <p className="text-white/30 text-xs mt-0.5">{kpi.subtitle}</p>}
          </motion.div>
        ))}
      </div>

      {/* PROMINENT Filter Bar */}
      <GlassCard animate={false} padding="p-4">
        <div className="flex items-center gap-2 mb-3">
          <FunnelIcon className="w-4 h-4 text-white/40" />
          <span className="text-white/50 text-xs font-semibold uppercase tracking-wider">Filters</span>
          {hasActiveFilters && (
            <motion.button
              className="ml-auto text-xs text-white/40 hover:text-white underline"
              onClick={clearAllFilters}
              whileHover={{ scale: 1.02 }}
            >
              Clear All
            </motion.button>
          )}
        </div>

        {/* Country Chips */}
        {(options.countries || []).length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {(options.countries || []).map((country) => {
              const active = filters.country.includes(country);
              return (
                <motion.button
                  key={country}
                  className={clsx(
                    'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                    active
                      ? 'bg-[#0057B8]/20 text-[#0057B8] border border-[#0057B8]/40'
                      : 'bg-white/5 text-white/50 border border-white/10 hover:bg-white/10 hover:text-white/80'
                  )}
                  onClick={() => updateFilter('country', country)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {country}
                </motion.button>
              );
            })}
          </div>
        )}

        {/* Dropdowns Row */}
        <div className="flex flex-wrap items-center gap-3">
          {[
            { key: 'product_family', label: 'Product Family', choices: options.productFamilies, urlKey: 'product_family' },
            { key: 'segment', label: 'Segment', choices: options.segments, urlKey: 'segment' },
            { key: 'business_unit', label: 'Business Unit', choices: options.businessUnits, urlKey: 'bu' },
            { key: 'risk_level', label: 'Risk Level', choices: options.riskLevels, urlKey: 'risk' },
          ].map((f) => (
            <select
              key={f.key}
              value={filters[f.key] || ''}
              onChange={(e) => updateFilter(f.urlKey, e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer min-w-[140px]"
            >
              <option value="" className="bg-[#1E293B]">{f.label}</option>
              {(f.choices || []).map((c) => (
                <option key={c} value={c} className="bg-[#1E293B]">{c}</option>
              ))}
            </select>
          ))}

          <div className="relative">
            <input
              type="text"
              placeholder="Search rep..."
              value={repSearch}
              onChange={(e) => setRepSearch(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl pl-3 pr-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50 w-40"
            />
          </div>
        </div>
      </GlassCard>

      {/* Toolbar: Column Picker & Exports */}
      <div className="flex items-center justify-between">
        <div className="relative">
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => setShowColumnPicker(!showColumnPicker)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <TableCellsIcon className="w-4 h-4" />
            Columns
          </motion.button>

          <AnimatePresence>
            {showColumnPicker && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="absolute z-40 top-full left-0 mt-2 w-56 bg-[#1E293B] border border-white/10 rounded-xl shadow-2xl p-3 max-h-80 overflow-y-auto"
              >
                {ALL_COLUMNS.filter((c) => !c.alwaysVisible).map((col) => (
                  <label key={col.key} className="flex items-center gap-2 py-1.5 px-2 rounded-lg hover:bg-white/5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={visibleColumns.has(col.key)}
                      onChange={() => toggleColumn(col.key)}
                      className="w-3.5 h-3.5 rounded accent-[#0057B8]"
                    />
                    <span className="text-white/70 text-xs">{col.label}</span>
                  </label>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="flex items-center gap-2">
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => exportData(sortedData, 'csv')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ArrowDownTrayIcon className="w-4 h-4" />
            CSV
          </motion.button>
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => exportData(sortedData, 'excel')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <DocumentArrowDownIcon className="w-4 h-4" />
            Excel
          </motion.button>
          <motion.button
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10"
            onClick={() => exportData(sortedData, 'copy')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ClipboardDocumentIcon className="w-4 h-4" />
            Copy
          </motion.button>
        </div>
      </div>

      {/* Main Data Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Price Change Actions ({sortedData.length} of {data?.length || 0})
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-white/5">
                {activeColumns.map((col) => (
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
              {sortedData.map((row, i) => (
                <React.Fragment key={i}>
                  <tr
                    className="hover:bg-white/5 cursor-pointer transition-colors"
                    onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                  >
                    {activeColumns.map((col) => (
                      <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                        {col.key === 'action_summary' ? (
                          <span className="font-semibold text-white">{row.action_summary || row.action || '--'}</span>
                        ) : col.key === 'risk_level' ? (
                          <span
                            className="px-2 py-0.5 rounded-full text-xs font-medium"
                            style={{
                              backgroundColor: `${RISK_COLORS[row.risk_level] || COLORS.primary}20`,
                              color: RISK_COLORS[row.risk_level] || COLORS.primary,
                            }}
                          >
                            {row.risk_level || '--'}
                          </span>
                        ) : col.key === 'change_pct' ? (
                          <span className={clsx('font-mono font-semibold', (row.change_pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                            {col.fmt(row[col.key])}
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
                    {expandedRow === i && (
                      <tr>
                        <td colSpan={activeColumns.length + 1}>
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="px-6 py-4 bg-white/[0.02] border-l-2 border-[#0057B8]/50"
                          >
                            {row.rationale && (
                              <div className="mb-3">
                                <p className="text-white/40 text-xs font-medium uppercase tracking-wider mb-1">Rationale</p>
                                <p className="text-white/70 text-sm">{row.rationale}</p>
                              </div>
                            )}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              {[
                                ['Product Name', row.product_name],
                                ['Business Unit', row.business_unit],
                                ['Confidence', formatPercent(row.confidence_score)],
                                ['Volume Impact', formatPercent(row.volume_impact)],
                                ['Elasticity', row.elasticity?.toFixed(3)],
                                ['Customers', formatNumber(row.customer_count)],
                                ['Rep', row.rep_name],
                                ['SKU', row.sku],
                              ].map(([label, val]) => (
                                <div key={label}>
                                  <p className="text-white/30 text-xs">{label}</p>
                                  <p className="text-white/80 text-sm font-medium">{val || '--'}</p>
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        </td>
                      </tr>
                    )}
                  </AnimatePresence>
                </React.Fragment>
              ))}
              {sortedData.length === 0 && (
                <tr>
                  <td colSpan={activeColumns.length + 1} className="px-4 py-12 text-center text-white/30 text-sm">
                    No price changes match the current filters.
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
