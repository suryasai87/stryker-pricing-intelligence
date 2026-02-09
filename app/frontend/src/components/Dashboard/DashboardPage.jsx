import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  CurrencyDollarIcon,
  ChartBarSquareIcon,
  ArrowTrendingUpIcon,
  CubeIcon,
} from '@heroicons/react/24/outline';
import KPICard from './KPICard';
import RevenueTreemap from './RevenueTreemap';
import PriceHeatmap from './PriceHeatmap';
import TrendChart from './TrendChart';
import LoadingShimmer from '../shared/LoadingShimmer';
import GlassCard from '../shared/GlassCard';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * usePortfolioData - Custom hook that fetches portfolio-level pricing intelligence data.
 *
 * Returns { data, loading, error, refetch }.
 * In a real application, this would call the backend API.
 * Here it simulates an API call with realistic Stryker-like data.
 */
function usePortfolioData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Simulate API latency
      await new Promise((resolve) => setTimeout(resolve, 800));

      // Simulated portfolio data mirroring Stryker's business segments
      const portfolioData = {
        kpis: {
          totalRevenue: 20400,
          avgMargin: 66.8,
          yoyGrowth: 8.3,
          activeProducts: 4287,
        },
        sparklines: {
          revenue: [
            { value: 18200 }, { value: 18500 }, { value: 18900 },
            { value: 19100 }, { value: 19400 }, { value: 19600 },
            { value: 19800 }, { value: 20000 }, { value: 20100 },
            { value: 20200 }, { value: 20300 }, { value: 20400 },
          ],
          margin: [
            { value: 63 }, { value: 64 }, { value: 64.5 },
            { value: 65 }, { value: 65.2 }, { value: 65.8 },
            { value: 66 }, { value: 66.2 }, { value: 66.5 },
            { value: 66.6 }, { value: 66.7 }, { value: 66.8 },
          ],
          growth: [
            { value: 6.2 }, { value: 6.5 }, { value: 6.8 },
            { value: 7.0 }, { value: 7.2 }, { value: 7.5 },
            { value: 7.7 }, { value: 7.9 }, { value: 8.0 },
            { value: 8.1 }, { value: 8.2 }, { value: 8.3 },
          ],
          products: [
            { value: 4100 }, { value: 4120 }, { value: 4150 },
            { value: 4170 }, { value: 4190 }, { value: 4210 },
            { value: 4230 }, { value: 4245 }, { value: 4260 },
            { value: 4270 }, { value: 4280 }, { value: 4287 },
          ],
        },
        segments: [
          { name: 'Orthopaedics', value: 6200, growth: 5.2, color: '#0057B8' },
          { name: 'MedSurg & Neurotechnology', value: 5100, growth: 7.8, color: '#10b981' },
          { name: 'Neurovascular', value: 3800, growth: 9.1, color: '#FFB81C' },
          { name: 'Capital Equipment', value: 3100, growth: 3.2, color: '#8b5cf6' },
          { name: 'Consumables', value: 2200, growth: 6.5, color: '#f43f5e' },
        ],
        heatmap: [
          { category: 'Hip Implants', values: [2.1, 1.8, 3.0, 2.5, 1.2, 0.8, -0.5, 1.0, 2.3, 3.1, 2.8, 1.5] },
          { category: 'Knee Implants', values: [1.5, 2.0, 1.8, 2.2, 3.0, 2.5, 1.0, 0.5, 1.8, 2.0, 2.5, 3.0] },
          { category: 'Spine Devices', values: [-0.5, -1.0, 0.2, 1.0, 1.5, 2.0, 2.5, 3.0, 2.8, 2.2, 1.8, 1.0] },
          { category: 'Trauma Fixation', values: [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, 0.5, 1.5, 2.0] },
          { category: 'Surgical Instruments', values: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5] },
          { category: 'Endoscopy', values: [0.5, -0.2, -0.8, 0.3, 1.2, 2.0, 2.5, 3.0, 2.8, 2.0, 1.5, 0.8] },
          { category: 'Power Tools', values: [2.0, 2.5, 3.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5] },
          { category: 'Navigation Systems', values: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 4.5, 4.0] },
        ],
        trends: [
          { month: 'Jan 24', revenue: 1450, margin: 62 },
          { month: 'Feb 24', revenue: 1520, margin: 63 },
          { month: 'Mar 24', revenue: 1480, margin: 61 },
          { month: 'Apr 24', revenue: 1600, margin: 64 },
          { month: 'May 24', revenue: 1650, margin: 63 },
          { month: 'Jun 24', revenue: 1580, margin: 62 },
          { month: 'Jul 24', revenue: 1700, margin: 65 },
          { month: 'Aug 24', revenue: 1750, margin: 64 },
          { month: 'Sep 24', revenue: 1680, margin: 63 },
          { month: 'Oct 24', revenue: 1820, margin: 66 },
          { month: 'Nov 24', revenue: 1900, margin: 67 },
          { month: 'Dec 24', revenue: 1850, margin: 65 },
          { month: 'Jan 25', revenue: 1920, margin: 66 },
          { month: 'Feb 25', revenue: 1980, margin: 67 },
          { month: 'Mar 25', revenue: 1950, margin: 66 },
          { month: 'Apr 25', revenue: 2050, margin: 68 },
          { month: 'May 25', revenue: 2100, margin: 67 },
          { month: 'Jun 25', revenue: 2020, margin: 66 },
          { month: 'Jul 25', revenue: 2150, margin: 69 },
          { month: 'Aug 25', revenue: 2200, margin: 68 },
          { month: 'Sep 25', revenue: 2180, margin: 67 },
          { month: 'Oct 25', revenue: 2300, margin: 70 },
          { month: 'Nov 25', revenue: 2380, margin: 71 },
          { month: 'Dec 25', revenue: 2350, margin: 69 },
        ],
      };

      setData(portfolioData);
    } catch (err) {
      setError(err.message || 'Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}

/**
 * LoadingSkeleton - Full dashboard skeleton while data loads.
 */
function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <GlassCard key={i} animate={false}>
            <LoadingShimmer width="60%" height="14px" className="mb-4" />
            <LoadingShimmer width="50%" height="28px" className="mb-3" />
            <LoadingShimmer width="40%" height="12px" />
          </GlassCard>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <GlassCard animate={false}>
          <LoadingShimmer width="40%" height="14px" className="mb-4" />
          <LoadingShimmer height="280px" rounded="rounded-xl" />
        </GlassCard>
        <GlassCard animate={false}>
          <LoadingShimmer width="40%" height="14px" className="mb-4" />
          <LoadingShimmer height="280px" rounded="rounded-xl" />
        </GlassCard>
      </div>

      {/* Heatmap */}
      <GlassCard animate={false}>
        <LoadingShimmer width="40%" height="14px" className="mb-4" />
        <LoadingShimmer height="320px" rounded="rounded-xl" />
      </GlassCard>
    </div>
  );
}

/**
 * ErrorState - Error display with retry button.
 */
function ErrorState({ message, onRetry }) {
  return (
    <GlassCard className="text-center py-16">
      <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <svg
          className="w-8 h-8 text-rose-400"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
          />
        </svg>
      </div>
      <h3 className="text-white font-semibold text-lg mb-2">Failed to Load Data</h3>
      <p className="text-white/50 text-sm mb-6 max-w-md mx-auto">{message}</p>
      <motion.button
        className="px-6 py-2.5 rounded-xl text-white text-sm font-medium"
        style={{ backgroundColor: '#0057B8' }}
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

/**
 * EmptyState - Shown when data loads but contains no meaningful content.
 */
function EmptyState() {
  return (
    <GlassCard className="text-center py-16">
      <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white/5 flex items-center justify-center">
        <CubeIcon className="w-8 h-8 text-white/30" />
      </div>
      <h3 className="text-white font-semibold text-lg mb-2">No Data Available</h3>
      <p className="text-white/50 text-sm max-w-md mx-auto">
        Portfolio data is not yet available. Please ensure data pipelines are running and check back shortly.
      </p>
    </GlassCard>
  );
}

/**
 * DashboardPage - Main dashboard page combining all dashboard components.
 *
 * Uses the usePortfolioData hook for data fetching and handles
 * loading, error, and empty states gracefully.
 */
export default function DashboardPage() {
  const { data, loading, error, refetch } = usePortfolioData();

  if (loading) {
    return (
      <div className="p-6">
        <LoadingSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <ErrorState message={error} onRetry={refetch} />
      </div>
    );
  }

  if (!data || !data.kpis) {
    return (
      <div className="p-6">
        <EmptyState />
      </div>
    );
  }

  const { kpis, sparklines, segments, heatmap, trends } = data;

  const kpiCards = [
    {
      icon: CurrencyDollarIcon,
      title: 'Total Revenue',
      value: kpis.totalRevenue,
      format: 'compact',
      delta: 8.3,
      deltaLabel: 'vs last year',
      sparkline: sparklines.revenue,
    },
    {
      icon: ChartBarSquareIcon,
      title: 'Avg Margin',
      value: kpis.avgMargin,
      format: 'percent',
      delta: 2.1,
      deltaLabel: 'vs last year',
      sparkline: sparklines.margin,
    },
    {
      icon: ArrowTrendingUpIcon,
      title: 'YoY Growth',
      value: kpis.yoyGrowth,
      format: 'percent',
      delta: 1.5,
      deltaLabel: 'vs prior period',
      sparkline: sparklines.growth,
    },
    {
      icon: CubeIcon,
      title: 'Active Products',
      value: kpis.activeProducts,
      format: 'number',
      delta: 3.2,
      deltaLabel: 'new this year',
      sparkline: sparklines.products,
    },
  ];

  return (
    <motion.div
      className="p-6 space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={springTransition}
    >
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpiCards.map((kpi, index) => (
          <KPICard
            key={kpi.title}
            icon={kpi.icon}
            title={kpi.title}
            value={kpi.value}
            format={kpi.format}
            delta={kpi.delta}
            deltaLabel={kpi.deltaLabel}
            sparkline={kpi.sparkline}
            index={index}
          />
        ))}
      </div>

      {/* Charts Row: Treemap + Trend */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RevenueTreemap data={segments} />
        <TrendChart data={trends} />
      </div>

      {/* Full-width Heatmap */}
      <PriceHeatmap data={heatmap} />
    </motion.div>
  );
}
