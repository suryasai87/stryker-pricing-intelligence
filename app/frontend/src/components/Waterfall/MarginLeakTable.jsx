import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * RAG badge component for margin health.
 */
function MarginBadge({ marginPct }) {
  let color, bg, border, label;
  if (marginPct >= 60) {
    color = '#10b981';
    bg = 'rgba(16, 185, 129, 0.1)';
    border = 'rgba(16, 185, 129, 0.3)';
    label = 'Healthy';
  } else if (marginPct >= 40) {
    color = '#FFB81C';
    bg = 'rgba(255, 184, 28, 0.1)';
    border = 'rgba(255, 184, 28, 0.3)';
    label = 'At Risk';
  } else {
    color = '#f43f5e';
    bg = 'rgba(244, 63, 94, 0.1)';
    border = 'rgba(244, 63, 94, 0.3)';
    label = 'Critical';
  }

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{ color, backgroundColor: bg, border: `1px solid ${border}` }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

/**
 * MarginLeakTable - Table showing margin leakage by product/category.
 *
 * Props:
 *   data    - array of product margin data
 *   loading - boolean
 *   title   - string (default: 'Margin Leakage by Product')
 */
export default function MarginLeakTable({
  data = null,
  loading = false,
  title = 'Margin Leakage by Product',
}) {
  const [sortKey, setSortKey] = useState('marginPct');
  const [sortAsc, setSortAsc] = useState(true);

  // Default demo data
  const tableData = data || [
    {
      id: '1', product: 'Mako SmartRobotics Hip System', category: 'Joint Replacement',
      listPrice: 45200, contractDiscount: 5420, gpoRebate: 3164, volumeBonus: 1808,
      freight: 904, cashDiscount: 452, pocketPrice: 33452, marginPct: 74.0,
    },
    {
      id: '2', product: 'Triathlon Knee System', category: 'Joint Replacement',
      listPrice: 12800, contractDiscount: 1920, gpoRebate: 1152, volumeBonus: 640,
      freight: 384, cashDiscount: 128, pocketPrice: 8576, marginPct: 67.0,
    },
    {
      id: '3', product: 'T2 Tibial Nail', category: 'Trauma & Extremities',
      listPrice: 3400, contractDiscount: 680, gpoRebate: 510, volumeBonus: 272,
      freight: 170, cashDiscount: 68, pocketPrice: 1700, marginPct: 50.0,
    },
    {
      id: '4', product: 'System 8 Power Tools', category: 'Instruments',
      listPrice: 22100, contractDiscount: 4420, gpoRebate: 3315, volumeBonus: 2210,
      freight: 1105, cashDiscount: 442, pocketPrice: 10608, marginPct: 48.0,
    },
    {
      id: '5', product: 'Neptune Waste Management', category: 'Instruments',
      listPrice: 18500, contractDiscount: 3700, gpoRebate: 2775, volumeBonus: 1850,
      freight: 925, cashDiscount: 555, pocketPrice: 8695, marginPct: 47.0,
    },
    {
      id: '6', product: 'Sage Turn & Position', category: 'Medical/Surgical',
      listPrice: 4200, contractDiscount: 1260, gpoRebate: 840, volumeBonus: 630,
      freight: 294, cashDiscount: 168, pocketPrice: 1008, marginPct: 24.0,
    },
    {
      id: '7', product: 'ProCuity Bed System', category: 'Medical/Surgical',
      listPrice: 35600, contractDiscount: 10680, gpoRebate: 5340, volumeBonus: 3560,
      freight: 1780, cashDiscount: 712, pocketPrice: 13528, marginPct: 38.0,
    },
  ];

  const columns = [
    { key: 'product', label: 'Product', align: 'left', width: 'w-48' },
    { key: 'listPrice', label: 'List Price', align: 'right' },
    { key: 'contractDiscount', label: 'Contract Disc.', align: 'right' },
    { key: 'gpoRebate', label: 'GPO Rebate', align: 'right' },
    { key: 'volumeBonus', label: 'Volume Bonus', align: 'right' },
    { key: 'freight', label: 'Freight', align: 'right' },
    { key: 'cashDiscount', label: 'Cash Disc.', align: 'right' },
    { key: 'pocketPrice', label: 'Pocket Price', align: 'right' },
    { key: 'marginPct', label: 'Margin %', align: 'center' },
  ];

  const sorted = useMemo(() => {
    return [...tableData].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === 'string') {
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortAsc ? aVal - bVal : bVal - aVal;
    });
  }, [tableData, sortKey, sortAsc]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const formatCurrency = (val) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <div className="px-6 py-4 border-b border-white/10">
        <h3 className="text-white font-semibold text-sm">{title}</h3>
        <p className="text-white/40 text-xs mt-0.5">
          RAG coding: <span className="text-[#10b981]">Green &gt;60%</span> | <span className="text-[#FFB81C]">Amber 40-60%</span> | <span className="text-[#f43f5e]">Red &lt;40%</span>
        </p>
      </div>

      {loading ? (
        <div className="p-6 space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="animate-pulse h-12 bg-white/5 rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                {columns.map((col) => (
                  <th
                    key={col.key}
                    className={`px-4 py-3 text-[10px] font-medium uppercase tracking-wider text-white/40 cursor-pointer hover:text-white/70 transition-colors ${
                      col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                    } ${col.width || ''}`}
                    onClick={() => handleSort(col.key)}
                  >
                    <div className={`flex items-center gap-1 ${col.align === 'right' ? 'justify-end' : col.align === 'center' ? 'justify-center' : ''}`}>
                      <span>{col.label}</span>
                      {sortKey === col.key && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.5 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={springTransition}
                        >
                          {sortAsc ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />}
                        </motion.div>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {sorted.map((row, index) => {
                  const marginColor = row.marginPct >= 60
                    ? 'rgba(16, 185, 129, 0.05)'
                    : row.marginPct >= 40
                      ? 'rgba(255, 184, 28, 0.03)'
                      : 'rgba(244, 63, 94, 0.05)';

                  return (
                    <motion.tr
                      key={row.id}
                      className="border-b border-white/5 hover:bg-white/5 transition-colors"
                      style={{ backgroundColor: marginColor }}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ ...springTransition, delay: index * 0.04 }}
                      layout
                    >
                      <td className="px-4 py-3">
                        <div>
                          <p className="text-white text-sm font-medium">{row.product}</p>
                          <p className="text-white/40 text-xs">{row.category}</p>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-white/80">{formatCurrency(row.listPrice)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#f43f5e]/80">-{formatCurrency(row.contractDiscount)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#f43f5e]/80">-{formatCurrency(row.gpoRebate)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#f43f5e]/80">-{formatCurrency(row.volumeBonus)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#f43f5e]/80">-{formatCurrency(row.freight)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#f43f5e]/80">-{formatCurrency(row.cashDiscount)}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono text-sm text-[#10b981] font-medium">{formatCurrency(row.pocketPrice)}</span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex flex-col items-center gap-1">
                          <span className="font-mono text-sm font-bold" style={{
                            color: row.marginPct >= 60 ? '#10b981' : row.marginPct >= 40 ? '#FFB81C' : '#f43f5e'
                          }}>
                            {row.marginPct.toFixed(1)}%
                          </span>
                          <MarginBadge marginPct={row.marginPct} />
                        </div>
                      </td>
                    </motion.tr>
                  );
                })}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  );
}
