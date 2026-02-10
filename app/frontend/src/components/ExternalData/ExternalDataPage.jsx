import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import {
  ExclamationTriangleIcon,
  ArrowUpTrayIcon,
  DocumentIcon,
  CheckCircleIcon,
  XCircleIcon,
  CloudArrowUpIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import GlassCard from '../shared/GlassCard';
import LoadingShimmer from '../shared/LoadingShimmer';
import { formatNumber } from '../../utils/formatters';
import { COLORS, CHART_PALETTE, RECHARTS_THEME } from '../../utils/colors';
import { fetchExternalData, fetchExternalDataSources, uploadExternalData } from '../../utils/apiV2';
import clsx from 'clsx';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const CATEGORY_OPTIONS = [
  'Competitor Pricing',
  'Market Index',
  'Inflation Data',
  'FX Rates',
  'Supply Chain Costs',
  'Clinical Outcomes',
  'Other',
];

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <GlassCard animate={false}>
        <LoadingShimmer width="40%" height="14px" className="mb-4" />
        <LoadingShimmer height="200px" rounded="rounded-xl" />
      </GlassCard>
      <GlassCard animate={false}>
        <LoadingShimmer width="30%" height="14px" className="mb-4" />
        <LoadingShimmer height="300px" rounded="rounded-xl" />
      </GlassCard>
      <GlassCard animate={false}>
        <LoadingShimmer width="30%" height="14px" className="mb-4" />
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

function UploadZone({ onUpload, uploading, uploadStatus }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = useCallback((e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  }, []);

  const handleSubmit = useCallback(() => {
    if (selectedFile && selectedCategory) {
      onUpload(selectedFile, selectedCategory);
    }
  }, [selectedFile, selectedCategory, onUpload]);

  return (
    <GlassCard animate={false}>
      <h3 className="text-white/70 text-sm font-semibold mb-4">Upload External Data</h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drop Zone */}
        <div
          className={clsx(
            'border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer',
            dragActive ? 'border-[#0057B8] bg-[#0057B8]/10' : 'border-white/10 hover:border-white/20'
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.xls,.csv"
            onChange={handleFileSelect}
            className="hidden"
          />
          <CloudArrowUpIcon className="w-12 h-12 text-white/20 mx-auto mb-3" />
          <p className="text-white/60 text-sm mb-1">
            Drag & drop your file here, or click to browse
          </p>
          <p className="text-white/30 text-xs">
            Supported formats: .xlsx, .xls, .csv
          </p>

          {selectedFile && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 flex items-center justify-center gap-2"
            >
              <DocumentIcon className="w-5 h-5 text-[#0057B8]" />
              <span className="text-white/80 text-sm font-medium">{selectedFile.name}</span>
              <span className="text-white/30 text-xs">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
            </motion.div>
          )}
        </div>

        {/* Upload Controls */}
        <div className="space-y-4">
          <div>
            <label className="block text-white/40 text-xs font-medium mb-1">Data Category</label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white/80 focus:outline-none focus:border-[#0057B8]/50 appearance-none cursor-pointer"
            >
              <option value="" className="bg-[#1E293B]">Select category...</option>
              {CATEGORY_OPTIONS.map((cat) => (
                <option key={cat} value={cat} className="bg-[#1E293B]">{cat}</option>
              ))}
            </select>
          </div>

          <motion.button
            className={clsx(
              'w-full flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-sm font-medium transition-colors',
              selectedFile && selectedCategory
                ? 'bg-[#0057B8] text-white hover:bg-[#003D82]'
                : 'bg-white/5 text-white/30 cursor-not-allowed'
            )}
            disabled={!selectedFile || !selectedCategory || uploading}
            onClick={handleSubmit}
            whileHover={selectedFile && selectedCategory ? { scale: 1.02 } : {}}
            whileTap={selectedFile && selectedCategory ? { scale: 0.98 } : {}}
          >
            {uploading ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                />
                Uploading...
              </>
            ) : (
              <>
                <ArrowUpTrayIcon className="w-4 h-4" />
                Upload File
              </>
            )}
          </motion.button>

          {/* Upload Status */}
          <AnimatePresence>
            {uploadStatus && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className={clsx(
                  'flex items-center gap-2 p-3 rounded-xl text-sm',
                  uploadStatus.success
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                )}
              >
                {uploadStatus.success
                  ? <CheckCircleIcon className="w-5 h-5 flex-shrink-0" />
                  : <XCircleIcon className="w-5 h-5 flex-shrink-0" />
                }
                <span>{uploadStatus.message}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </GlassCard>
  );
}

export default function ExternalDataPage() {
  const [sources, setSources] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [selectedSource, setSelectedSource] = useState(null);
  const [previewFilter, setPreviewFilter] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'upload_date', dir: 'desc' });

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [srcData, extData] = await Promise.all([
        fetchExternalDataSources(),
        fetchExternalData(),
      ]);
      setSources(srcData?.data || srcData || []);
      setPreviewData(extData?.data || extData || []);
    } catch (err) {
      setError(err.message || 'Failed to load external data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleUpload = useCallback(async (file, category) => {
    setUploading(true);
    setUploadStatus(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', category);
      await uploadExternalData(formData);
      setUploadStatus({ success: true, message: `Successfully uploaded ${file.name}` });
      // Refresh sources
      fetchData();
    } catch (err) {
      setUploadStatus({ success: false, message: err.message || 'Upload failed' });
    } finally {
      setUploading(false);
    }
  }, [fetchData]);

  const sortedSources = useMemo(() => {
    if (!sources) return [];
    const rows = [...sources];
    rows.sort((a, b) => {
      const av = a[sortConfig.key] ?? '';
      const bv = b[sortConfig.key] ?? '';
      if (typeof av === 'string') return sortConfig.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortConfig.dir === 'asc' ? (av || 0) - (bv || 0) : (bv || 0) - (av || 0);
    });
    return rows;
  }, [sources, sortConfig]);

  const filteredPreview = useMemo(() => {
    if (!previewData) return [];
    let rows = previewData;
    if (selectedSource) {
      rows = rows.filter((r) => r.source_name === selectedSource || r.source_id === selectedSource);
    }
    if (previewFilter.trim()) {
      const q = previewFilter.toLowerCase();
      rows = rows.filter((r) =>
        Object.values(r).some((v) => String(v).toLowerCase().includes(q))
      );
    }
    return rows;
  }, [previewData, selectedSource, previewFilter]);

  // Time series chart data
  const timeSeriesData = useMemo(() => {
    if (!previewData || !previewData.length) return [];
    const byDate = {};
    previewData.forEach((r) => {
      const date = r.date || r.period || r.timestamp;
      if (!date) return;
      if (!byDate[date]) byDate[date] = { date };
      const cat = r.category || r.source_name || 'Value';
      byDate[date][cat] = r.value ?? r.amount ?? r.price ?? 0;
    });
    return Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date));
  }, [previewData]);

  const timeSeriesKeys = useMemo(() => {
    if (!timeSeriesData.length) return [];
    return [...new Set(timeSeriesData.flatMap((d) => Object.keys(d).filter((k) => k !== 'date')))];
  }, [timeSeriesData]);

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

  const previewColumns = filteredPreview.length > 0
    ? Object.keys(filteredPreview[0]).slice(0, 10)
    : [];

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={springTransition}
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={springTransition}>
        <h2 className="text-white text-xl font-bold">External Data</h2>
        <p className="text-white/40 text-sm mt-0.5">
          Upload and manage external data sources for pricing analysis
        </p>
      </motion.div>

      {/* Upload Section */}
      <UploadZone onUpload={handleUpload} uploading={uploading} uploadStatus={uploadStatus} />

      {/* Sources Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Data Sources ({sortedSources.length})
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-white/5">
                {[
                  { key: 'source_name', label: 'Source Name' },
                  { key: 'category', label: 'Category' },
                  { key: 'row_count', label: 'Row Count' },
                  { key: 'upload_date', label: 'Upload Date' },
                ].map((col) => (
                  <th
                    key={col.key}
                    className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider cursor-pointer hover:text-white/70 whitespace-nowrap"
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    <SortIcon colKey={col.key} />
                  </th>
                ))}
                <th className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {sortedSources.map((src, i) => (
                <tr key={i} className="hover:bg-white/5 transition-colors">
                  <td className="px-4 py-3 text-sm text-white/80">{src.source_name || src.name || '--'}</td>
                  <td className="px-4 py-3 text-sm">
                    <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-[#0057B8]/20 text-[#0057B8]">
                      {src.category || '--'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-white/80 font-mono">{formatNumber(src.row_count)}</td>
                  <td className="px-4 py-3 text-sm text-white/60">
                    {src.upload_date ? new Date(src.upload_date).toLocaleDateString() : '--'}
                  </td>
                  <td className="px-4 py-3">
                    <motion.button
                      className="text-xs text-[#0057B8] hover:text-[#3B82F6] font-medium"
                      onClick={() => setSelectedSource(src.source_name || src.name || src.source_id)}
                      whileHover={{ scale: 1.05 }}
                    >
                      Preview
                    </motion.button>
                  </td>
                </tr>
              ))}
              {sortedSources.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-12 text-center text-white/30 text-sm">
                    No data sources uploaded yet. Use the upload section above to add external data.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Time Series Chart */}
      {timeSeriesData.length > 0 && (
        <GlassCard animate={false}>
          <h3 className="text-white/70 text-sm font-semibold mb-4">External Data Time Series</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid {...RECHARTS_THEME.grid} />
              <XAxis
                dataKey="date"
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
              />
              <YAxis
                {...RECHARTS_THEME.tick}
                axisLine={RECHARTS_THEME.axisLine}
                tickLine={RECHARTS_THEME.tickLine}
              />
              <RechartsTooltip
                contentStyle={RECHARTS_THEME.tooltip.contentStyle}
                labelStyle={RECHARTS_THEME.tooltip.labelStyle}
              />
              <Legend wrapperStyle={RECHARTS_THEME.legend.wrapperStyle} />
              {timeSeriesKeys.map((key, idx) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={CHART_PALETTE[idx % CHART_PALETTE.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </GlassCard>
      )}

      {/* Data Preview Table */}
      <GlassCard animate={false} padding="p-0">
        <div className="flex items-center justify-between p-4 border-b border-white/5">
          <h3 className="text-white/70 text-sm font-semibold">
            Data Preview {selectedSource ? `- ${selectedSource}` : ''} ({filteredPreview.length} rows)
          </h3>
          <div className="flex items-center gap-3">
            {selectedSource && (
              <motion.button
                className="text-xs text-white/40 hover:text-white underline"
                onClick={() => setSelectedSource(null)}
                whileHover={{ scale: 1.02 }}
              >
                Show All
              </motion.button>
            )}
            <input
              type="text"
              placeholder="Filter..."
              value={previewFilter}
              onChange={(e) => setPreviewFilter(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 placeholder:text-white/30 focus:outline-none focus:border-[#0057B8]/50 w-40"
            />
          </div>
        </div>

        {previewColumns.length > 0 ? (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left">
              <thead className="sticky top-0 bg-[#1E293B]">
                <tr className="border-b border-white/5">
                  {previewColumns.map((col) => (
                    <th key={col} className="px-4 py-3 text-xs font-medium text-white/40 uppercase tracking-wider whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {filteredPreview.slice(0, 100).map((row, i) => (
                  <tr key={i} className="hover:bg-white/5 transition-colors">
                    {previewColumns.map((col) => (
                      <td key={col} className="px-4 py-2.5 text-sm text-white/70 whitespace-nowrap max-w-xs truncate">
                        {row[col] ?? '--'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="px-4 py-12 text-center text-white/30 text-sm">
            Select a source to preview its data.
          </div>
        )}
      </GlassCard>
    </motion.div>
  );
}
