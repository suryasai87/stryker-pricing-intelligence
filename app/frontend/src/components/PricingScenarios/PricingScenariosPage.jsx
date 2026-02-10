import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  PlusIcon,
  UserCircleIcon,
  MagnifyingGlassIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatPercent, formatNumber, formatCurrency } from '../../utils/formatters';
import { COLORS, CHART_PALETTE } from '../../utils/colors';
import {
  fetchPricingScenarios,
  fetchScenarioUserInfo,
  createPricingScenario,
  updateScenarioStatus,
} from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const STATUS_COLORS = {
  draft: COLORS.textTertiary,
  submitted: COLORS.primary,
  in_review: COLORS.accent,
  approved: COLORS.success,
  rejected: COLORS.danger,
  implemented: COLORS.info,
};

const STATUS_LABELS = {
  draft: 'Draft',
  submitted: 'Submitted',
  in_review: 'In Review',
  approved: 'Approved',
  rejected: 'Rejected',
  implemented: 'Implemented',
};

const STATUS_OPTIONS = ['draft', 'submitted', 'in_review', 'approved', 'rejected', 'implemented'];

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <GlassCard animate={false}>
        <LoadingShimmer width="40%" height="14px" className="mb-4" />
        <LoadingShimmer height="60px" rounded="rounded-xl" />
      </GlassCard>
      <GlassCard animate={false}>
        <LoadingShimmer width="40%" height="14px" className="mb-4" />
        <LoadingShimmer height="300px" rounded="rounded-xl" />
      </GlassCard>
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

function CreateScenarioForm({ onSubmit, submitting }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [assumptions, setAssumptions] = useState('');
  const [targetUplift, setTargetUplift] = useState('');
  const [skus, setSkus] = useState('');
  const [segments, setSegments] = useState('');
  const [showForm, setShowForm] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    onSubmit({
      name: name.trim(),
      description: description.trim(),
      assumptions: assumptions.trim(),
      target_uplift_pct: targetUplift ? parseFloat(targetUplift) : undefined,
      skus: skus.trim() ? skus.split(',').map((s) => s.trim()) : [],
      segments: segments.trim() ? segments.split(',').map((s) => s.trim()) : [],
    });
    // Reset form
    setName('');
    setDescription('');
    setAssumptions('');
    setTargetUplift('');
    setSkus('');
    setSegments('');
    setShowForm(false);
  };

  return (
    <GlassCard animate={false}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white/70 text-sm font-semibold">Create New Scenario</h3>
        <motion.button
          className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors',
            showForm
              ? 'bg-white/10 text-white/60'
              : 'bg-[#0057B8] text-white'
          )}
          onClick={() => setShowForm(!showForm)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <PlusIcon className="w-4 h-4" />
          {showForm ? 'Cancel' : 'New Scenario'}
        </motion.button>
      </div>

      <AnimatePresence>
        {showForm && (
          <motion.form
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={springTransition}
            onSubmit={handleSubmit}
            className="space-y-4"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="block text-white/40 text-xs font-medium mb-1">Scenario Name *</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Q3 2025 Joint Replacement Uplift"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
                />
              </div>
              <div>
                <label className="block text-white/40 text-xs font-medium mb-1">Target Uplift %</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="50"
                  value={targetUplift}
                  onChange={(e) => setTargetUplift(e.target.value)}
                  placeholder="e.g., 2.5"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white font-mono placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
                />
              </div>
            </div>

            <div>
              <label className="block text-white/40 text-xs font-medium mb-1">Description</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of this scenario"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
              />
            </div>

            <div>
              <label className="block text-white/40 text-xs font-medium mb-1">Assumptions</label>
              <textarea
                rows={3}
                value={assumptions}
                onChange={(e) => setAssumptions(e.target.value)}
                placeholder="Key assumptions, constraints, and business context..."
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50 resize-none"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="block text-white/40 text-xs font-medium mb-1">SKUs (comma-separated)</label>
                <input
                  type="text"
                  value={skus}
                  onChange={(e) => setSkus(e.target.value)}
                  placeholder="SKU-001, SKU-002, SKU-003..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
                />
              </div>
              <div>
                <label className="block text-white/40 text-xs font-medium mb-1">Segments (comma-separated)</label>
                <input
                  type="text"
                  value={segments}
                  onChange={(e) => setSegments(e.target.value)}
                  placeholder="Orthopaedics, Spine, MedSurg..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
                />
              </div>
            </div>

            <motion.button
              type="submit"
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-white text-sm font-medium bg-[#0057B8]"
              disabled={!name.trim() || submitting}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {submitting ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                  />
                  Submitting...
                </>
              ) : (
                'Run & Submit Scenario'
              )}
            </motion.button>
          </motion.form>
        )}
      </AnimatePresence>
    </GlassCard>
  );
}

export default function PricingScenariosPage() {
  const [scenarios, setScenarios] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'created_at', dir: 'desc' });
  const [globalSearch, setGlobalSearch] = useState('');

  const isAdmin = userInfo?.role === 'admin' || userInfo?.is_admin;

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [scenarioData, user] = await Promise.all([
        fetchPricingScenarios(),
        fetchScenarioUserInfo(),
      ]);
      setScenarios(scenarioData?.data || scenarioData || []);
      setUserInfo(user);
    } catch (err) {
      setError(err.message || 'Failed to load pricing scenarios');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleCreateScenario = useCallback(async (scenario) => {
    setSubmitting(true);
    try {
      await createPricingScenario(scenario);
      fetchData();
    } catch (err) {
      console.error('Create scenario error:', err);
    } finally {
      setSubmitting(false);
    }
  }, [fetchData]);

  const handleStatusUpdate = useCallback(async (scenarioId, newStatus) => {
    try {
      await updateScenarioStatus(scenarioId, newStatus);
      fetchData();
    } catch (err) {
      console.error('Status update error:', err);
    }
  }, [fetchData]);

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

  const sortedScenarios = useMemo(() => {
    if (!scenarios) return [];
    let rows = [...scenarios];

    // Global search (admin)
    if (globalSearch.trim()) {
      const q = globalSearch.toLowerCase();
      rows = rows.filter((r) =>
        Object.values(r).some((v) => String(v).toLowerCase().includes(q))
      );
    }

    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? '';
      const bv = b[sortConfig.key] ?? '';
      if (typeof av === 'string') return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortConfig.dir === 'asc' ? (av || 0) - (bv || 0) : (bv || 0) - (av || 0);
    });
    return rows;
  }, [scenarios, sortConfig, globalSearch]);

  if (loading) return <div className="p-6"><LoadingSkeleton /></div>;
  if (error) return <div className="p-6"><ErrorState message={error} onRetry={fetchData} /></div>;

  const baseColumns = [
    { key: 'name', label: 'Scenario Name' },
    { key: 'status', label: 'Status' },
    { key: 'target_uplift_pct', label: 'Target %', fmt: (v) => formatPercent(v) },
    { key: 'created_at', label: 'Date', fmt: (v) => v ? new Date(v).toLocaleDateString() : '--' },
    { key: 'description', label: 'Description' },
    { key: 'result_revenue_impact', label: 'Rev Impact', fmt: (v) => v ? formatCurrency(v, { compact: true }) : '--' },
    { key: 'result_actions', label: 'Actions', fmt: (v) => v ? formatNumber(v) : '--' },
  ];

  const columns = isAdmin
    ? [{ key: 'user_name', label: 'User' }, ...baseColumns]
    : baseColumns;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={springTransition}
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={springTransition}>
        <h2 className="text-white text-xl font-bold">Pricing Scenarios</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Create, submit, and track pricing scenario analyses
        </p>
      </motion.div>

      {/* User Info */}
      {userInfo && (
        <motion.div
          className="flex items-center gap-3"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...springTransition, delay: 0.05 }}
        >
          <div className="w-10 h-10 rounded-full bg-[#0057B8]/20 flex items-center justify-center">
            <UserCircleIcon className="w-6 h-6 text-[#0057B8]" />
          </div>
          <div>
            <p className="text-white text-sm font-medium">{userInfo.name || userInfo.email || 'User'}</p>
            <p className="text-white/40 text-xs">
              {isAdmin ? 'Administrator' : 'Analyst'}
              {userInfo.department && ` -- ${userInfo.department}`}
            </p>
          </div>
        </motion.div>
      )}

      {/* Create Scenario Form */}
      <CreateScenarioForm onSubmit={handleCreateScenario} submitting={submitting} />

      {/* Admin: Global Search */}
      {isAdmin && (
        <motion.div
          className="flex items-center gap-3"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...springTransition, delay: 0.1 }}
        >
          <div className="relative flex-1 max-w-md">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
            <input
              type="text"
              placeholder="Search all scenarios (admin)..."
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50"
            />
          </div>
        </motion.div>
      )}

      {/* Scenario History Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Scenario History ({sortedScenarios.length})
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
                {isAdmin && (
                  <th className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider">Update Status</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {sortedScenarios.map((row, i) => (
                <tr key={row.id || i} className="hover:bg-white/5 transition-colors">
                  {columns.map((col) => (
                    <td key={col.key} className="px-4 py-3 text-sm text-white/80 whitespace-nowrap">
                      {col.key === 'status' ? (
                        <span
                          className="px-2.5 py-1 rounded-full text-xs font-medium"
                          style={{
                            backgroundColor: `${STATUS_COLORS[row.status] || COLORS.textTertiary}20`,
                            color: STATUS_COLORS[row.status] || COLORS.textTertiary,
                          }}
                        >
                          {STATUS_LABELS[row.status] || row.status || '--'}
                        </span>
                      ) : col.key === 'name' ? (
                        <span className="font-semibold text-white">{row.name || '--'}</span>
                      ) : col.key === 'description' ? (
                        <span className="text-white/50 max-w-xs truncate block">{row.description || '--'}</span>
                      ) : col.fmt ? col.fmt(row[col.key]) : (row[col.key] ?? '--')}
                    </td>
                  ))}
                  {isAdmin && (
                    <td className="px-4 py-3">
                      <select
                        value={row.status || ''}
                        onChange={(e) => handleStatusUpdate(row.id, e.target.value)}
                        className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs text-white/70 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer"
                      >
                        {STATUS_OPTIONS.map((s) => (
                          <option key={s} value={s} className="bg-[#1E293B]">
                            {STATUS_LABELS[s]}
                          </option>
                        ))}
                      </select>
                    </td>
                  )}
                </tr>
              ))}
              {sortedScenarios.length === 0 && (
                <tr>
                  <td colSpan={columns.length + (isAdmin ? 1 : 0)} className="px-4 py-12 text-center text-white/30 text-sm">
                    No scenarios found. Create a new scenario to get started.
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
