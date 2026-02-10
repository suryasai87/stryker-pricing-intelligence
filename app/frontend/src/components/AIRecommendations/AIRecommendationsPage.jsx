import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';
import {
  ExclamationTriangleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronRightIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatCurrency, formatPercent, formatNumber, formatCompact } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchPricingRecommendations, fetchPricingRecommendationsSummary } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const RISK_COLORS = {
  low: COLORS.success,
  medium: COLORS.accent,
  high: COLORS.warning,
  critical: COLORS.danger,
};

const ACTION_TYPE_COLORS = {
  increase: COLORS.success,
  decrease: COLORS.danger,
  hold: COLORS.info,
  bundle: COLORS.accent,
  restructure: COLORS.primaryLight,
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

function MiniDonut({ data }) {
  if (!data || !data.length) return null;
  return (
    <div className="w-16 h-16">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="type"
            cx="50%"
            cy="50%"
            innerRadius={14}
            outerRadius={28}
            strokeWidth={0}
          >
            {data.map((entry, idx) => (
              <Cell key={idx} fill={ACTION_TYPE_COLORS[entry.type] || CHART_PALETTE[idx % CHART_PALETTE.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

function DonutTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0];
  return (
    <div style={RECHARTS_THEME.tooltip.contentStyle}>
      <p className="text-white font-semibold text-sm">{d.name}</p>
      <p className="text-white/70 text-xs">{d.value} recommendations</p>
    </div>
  );
}

export default function AIRecommendationsPage() {
  const [data, setData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({});
  const [filterOptions, setFilterOptions] = useState({
    actionTypes: [], riskLevels: [], businessUnits: [], productFamilies: [], countries: [],
  });
  const [sortConfig, setSortConfig] = useState({ key: 'priority_score', dir: 'desc' });
  const [expandedRow, setExpandedRow] = useState(null);
  const [priorityMin, setPriorityMin] = useState('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {};
      Object.entries(filters).forEach(([k, v]) => { if (v) params[k] = v; });
      if (priorityMin) params.priority_min = priorityMin;
      const [recs, summ] = await Promise.all([
        fetchPricingRecommendations(params),
        fetchPricingRecommendationsSummary(params),
      ]);
      const rows = recs?.data || recs || [];
      setData(rows);
      setSummary(summ);
      if (rows.length) {
        setFilterOptions({
          actionTypes: [...new Set(rows.map((r) => r.action_type).filter(Boolean))].sort(),
          riskLevels: [...new Set(rows.map((r) => r.risk_level).filter(Boolean))].sort(),
          businessUnits: [...new Set(rows.map((r) => r.business_unit).filter(Boolean))].sort(),
          productFamilies: [...new Set(rows.map((r) => r.product_family).filter(Boolean))].sort(),
          countries: [...new Set(rows.map((r) => r.country).filter(Boolean))].sort(),
        });
      }
    } catch (err) {
      setError(err.message || 'Failed to load AI recommendations');
    } finally {
      setLoading(false);
    }
  }, [filters, priorityMin]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleFilterReset = () => {
    setFilters({});
    setPriorityMin('');
  };

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

  // Build donut chart data
  const donutData = useMemo(() => {
    if (summary?.by_type) return summary.by_type;
    if (!data) return [];
    const counts = {};
    data.forEach((r) => {
      const t = r.action_type || 'other';
      counts[t] = (counts[t] || 0) + 1;
    });
    return Object.entries(counts).map(([type, count]) => ({ type, count }));
  }, [data, summary]);

  // Top 15 by priority score
  const topByPriority = useMemo(() => {
    if (!data) return [];
    return [...data]
      .sort((a, b) => (b.priority_score || 0) - (a.priority_score || 0))
      .slice(0, 15);
  }, [data]);

  if (loading) return <div className="p-6"><LoadingSkeleton /></div>;
  if (error) return <div className="p-6"><ErrorState message={error} onRetry={fetchData} /></div>;

  const totalRecs = data?.length || 0;
  const totalRevGain = data?.reduce((s, r) => s + (r.expected_rev_gain || 0), 0) || 0;
  const avgRiskScore = data?.length
    ? (data.reduce((s, r) => {
        const rm = { low: 1, medium: 2, high: 3, critical: 4 };
        return s + (rm[r.risk_level] || 0);
      }, 0) / data.length).toFixed(1)
    : '--';

  const kpis = [
    { title: 'Total Recommendations', value: formatNumber(totalRecs), color: COLORS.primary, subtitle: 'AI-generated pricing actions' },
    { title: 'Expected Rev Gain', value: formatCurrency(totalRevGain, { compact: true }), color: COLORS.success, subtitle: 'Projected revenue impact' },
    { title: 'Avg Risk Score', value: avgRiskScore, color: COLORS.warning, subtitle: '1=Low, 4=Critical' },
    {
      title: 'By Type',
      value: null,
      color: COLORS.accent,
      subtitle: 'Action distribution',
      miniDonut: donutData,
    },
  ];

  const columns = [
    { key: 'priority_score', label: 'Priority', fmt: (v) => v?.toFixed(1) ?? '--' },
    { key: 'action_type', label: 'Action Type' },
    { key: 'sku', label: 'SKU' },
    { key: 'product_family', label: 'Product Family' },
    { key: 'business_unit', label: 'BU' },
    { key: 'country', label: 'Country' },
    { key: 'expected_rev_gain', label: 'Rev Gain', fmt: (v) => formatCurrency(v, { compact: true }) },
    { key: 'risk_level', label: 'Risk' },
    { key: 'confidence', label: 'Confidence', fmt: (v) => formatPercent(v) },
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
        <div className="flex items-center gap-2">
          <SparklesIcon className="w-5 h-5 text-[#FFB81C]" />
          <h2 className="text-white text-xl font-bold">AI Pricing Recommendations</h2>
        </div>
        <p className="text-white/40 text-sm mt-0.5">
          ML-driven pricing actions ranked by priority and expected impact
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
            {kpi.miniDonut ? (
              <div className="flex items-center gap-3 mt-1">
                <MiniDonut data={kpi.miniDonut} />
                <div className="space-y-0.5">
                  {kpi.miniDonut.slice(0, 3).map((d) => (
                    <p key={d.type} className="text-white/50 text-[10px]">
                      <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ backgroundColor: ACTION_TYPE_COLORS[d.type] || COLORS.primary }} />
                      {d.type}: {d.count}
                    </p>
                  ))}
                </div>
              </div>
            ) : (
              <p className="font-mono text-2xl font-bold mt-1" style={{ color: kpi.color }}>{kpi.value}</p>
            )}
            {kpi.subtitle && <p className="text-white/30 text-xs mt-0.5">{kpi.subtitle}</p>}
          </motion.div>
        ))}
      </div>

      {/* Filter Bar */}
      <motion.div
        className="flex flex-wrap items-center gap-3"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...springTransition, delay: 0.1 }}
      >
        {[
          { key: 'action_type', label: 'Action Type', choices: filterOptions.actionTypes },
          { key: 'risk_level', label: 'Risk Level', choices: filterOptions.riskLevels },
          { key: 'business_unit', label: 'Business Unit', choices: filterOptions.businessUnits },
          { key: 'product_family', label: 'Product Family', choices: filterOptions.productFamilies },
          { key: 'country', label: 'Country', choices: filterOptions.countries },
        ].map((f) => (
          <select
            key={f.key}
            value={filters[f.key] || ''}
            onChange={(e) => handleFilterChange(f.key, e.target.value)}
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer min-w-[130px]"
          >
            <option value="" className="bg-[#1E293B]">{f.label}</option>
            {(f.choices || []).map((c) => (
              <option key={c} value={c} className="bg-[#1E293B]">{c}</option>
            ))}
          </select>
        ))}

        <div>
          <input
            type="number"
            placeholder="Min Priority"
            value={priorityMin}
            onChange={(e) => setPriorityMin(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50 w-32"
          />
        </div>

        {(Object.values(filters).some(Boolean) || priorityMin) && (
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
        {/* Donut Chart: Recs by Action Type */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Recommendations by Action Type</h3>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={donutData}
                dataKey="count"
                nameKey="type"
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={110}
                paddingAngle={2}
                strokeWidth={0}
                label={({ type, count }) => `${type} (${count})`}
              >
                {donutData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={ACTION_TYPE_COLORS[entry.type] || CHART_PALETTE[idx % CHART_PALETTE.length]}
                    fillOpacity={0.8}
                  />
                ))}
              </Pie>
              <RechartsTooltip content={<DonutTooltip />} />
              <Legend wrapperStyle={RECHARTS_THEME.legend.wrapperStyle} />
            </PieChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Bar Chart: Top 15 by Priority */}
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">Top 15 by Priority Score</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={topByPriority} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 60 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} horizontal={false} />
              <XAxis type="number" {...RECHARTS_THEME.tick} axisLine={RECHARTS_THEME.axisLine} tickLine={RECHARTS_THEME.tickLine} />
              <YAxis
                type="category"
                dataKey="sku"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
                width={55}
                tick={{ fill: COLORS.textSecondary, fontSize: 9 }}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
                formatter={(val, name) => {
                  if (name === 'priority_score') return [val?.toFixed(1), 'Priority'];
                  return [val, name];
                }}
              />
              <Bar dataKey="priority_score" radius={[0, 6, 6, 0]}>
                {topByPriority.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={RISK_COLORS[entry.risk_level] || CHART_PALETTE[idx % CHART_PALETTE.length]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Recommendations Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            All Recommendations ({sortedData.length} actions)
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
                    {columns.map((col) => (
                      <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                        {col.key === 'action_type' ? (
                          <span
                            className="px-2 py-0.5 rounded-full text-xs font-medium"
                            style={{
                              backgroundColor: `${ACTION_TYPE_COLORS[row.action_type] || COLORS.primary}20`,
                              color: ACTION_TYPE_COLORS[row.action_type] || COLORS.primary,
                            }}
                          >
                            {row.action_type || '--'}
                          </span>
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
                        <td colSpan={columns.length + 1}>
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="px-6 py-4 bg-white/[0.02] border-l-2 border-[#FFB81C]/50"
                          >
                            {row.rationale && (
                              <div className="mb-3">
                                <p className="text-white/40 text-xs font-medium uppercase tracking-wider mb-1">Rationale</p>
                                <p className="text-white/70 text-sm">{row.rationale}</p>
                              </div>
                            )}
                            {row.competitive_context && (
                              <div className="mb-3">
                                <p className="text-white/40 text-xs font-medium uppercase tracking-wider mb-1">Competitive Context</p>
                                <p className="text-white/70 text-sm">{row.competitive_context}</p>
                              </div>
                            )}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                              {[
                                ['Segment', row.segment],
                                ['Current Price', formatCurrency(row.current_price)],
                                ['Rec. Price', formatCurrency(row.recommended_price)],
                                ['Change %', formatPercent(row.change_pct)],
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
                  <td colSpan={columns.length + 1} className="px-4 py-12 text-center text-white/30 text-sm">
                    No recommendations found matching current filters.
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
