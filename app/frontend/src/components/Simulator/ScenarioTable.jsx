import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronUpIcon, ChevronDownIcon, TrashIcon } from '@heroicons/react/24/outline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * ScenarioTable - Table comparing saved pricing scenarios.
 *
 * Props:
 *   scenarios  - array of scenario objects
 *   onDelete   - function(scenarioId), called when delete button is clicked
 *   onSelect   - function(scenario), called when a row is clicked
 *   selected   - string|null, id of the currently selected scenario
 */
export default function ScenarioTable({ scenarios = [], onDelete, onSelect, selected = null }) {
  const [sortKey, setSortKey] = useState('name');
  const [sortAsc, setSortAsc] = useState(true);

  const data = scenarios;

  const columns = [
    { key: 'name', label: 'Scenario', align: 'left' },
    { key: 'product', label: 'Product', align: 'left' },
    { key: 'priceChange', label: 'Price \u0394 %', align: 'right' },
    { key: 'volumeDelta', label: 'Volume \u0394 %', align: 'right' },
    { key: 'revenueDelta', label: 'Revenue \u0394', align: 'right' },
    { key: 'marginDelta', label: 'Margin \u0394', align: 'right' },
  ];

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === 'string') {
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortAsc ? aVal - bVal : bVal - aVal;
    });
  }, [data, sortKey, sortAsc]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const formatDelta = (value, isCurrency = false) => {
    if (isCurrency) {
      const abs = Math.abs(value);
      const formatted = abs >= 1000000
        ? `$${(abs / 1000000).toFixed(1)}M`
        : abs >= 1000
          ? `$${(abs / 1000).toFixed(0)}K`
          : `$${abs.toFixed(0)}`;
      return value >= 0 ? `+${formatted}` : `-${formatted}`;
    }
    return value >= 0 ? `+${value.toFixed(1)}%` : `${value.toFixed(1)}%`;
  };

  const getDeltaColor = (value) => {
    if (value > 0) return 'text-[#10b981]';
    if (value < 0) return 'text-[#f43f5e]';
    return 'text-white/50';
  };

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
        <div>
          <h3 className="text-white font-semibold text-sm">Saved Scenarios</h3>
          <p className="text-white/40 text-xs mt-0.5">{data.length} scenario{data.length !== 1 ? 's' : ''} compared</p>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-6 py-3 text-xs font-medium uppercase tracking-wider text-white/40 cursor-pointer hover:text-white/70 transition-colors ${
                    col.align === 'right' ? 'text-right' : 'text-left'
                  }`}
                  onClick={() => handleSort(col.key)}
                >
                  <div className={`flex items-center gap-1 ${col.align === 'right' ? 'justify-end' : ''}`}>
                    <span>{col.label}</span>
                    {sortKey === col.key && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={springTransition}
                      >
                        {sortAsc ? (
                          <ChevronUpIcon className="w-3 h-3" />
                        ) : (
                          <ChevronDownIcon className="w-3 h-3" />
                        )}
                      </motion.div>
                    )}
                  </div>
                </th>
              ))}
              <th className="px-4 py-3 w-10" />
            </tr>
          </thead>
          <tbody>
            <AnimatePresence>
              {sorted.map((scenario, index) => (
                <motion.tr
                  key={scenario.id}
                  className={`
                    border-b border-white/5 cursor-pointer transition-colors
                    ${selected === scenario.id ? 'bg-[#0057B8]/10' : 'hover:bg-white/5'}
                  `}
                  onClick={() => onSelect?.(scenario)}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ ...springTransition, delay: index * 0.04 }}
                  layout
                >
                  {/* Scenario Name */}
                  <td className="px-6 py-4">
                    <span className="text-white text-sm font-medium">{scenario.name}</span>
                  </td>

                  {/* Product */}
                  <td className="px-6 py-4">
                    <span className="text-white/70 text-sm">{scenario.product}</span>
                  </td>

                  {/* Price Change */}
                  <td className="px-6 py-4 text-right">
                    <span className={`font-mono text-sm ${getDeltaColor(scenario.priceChange)}`}>
                      {scenario.priceChange > 0 ? '+' : ''}{scenario.priceChange.toFixed(1)}%
                    </span>
                  </td>

                  {/* Volume Delta */}
                  <td className="px-6 py-4 text-right">
                    <span className={`font-mono text-sm ${getDeltaColor(scenario.volumeDelta)}`}>
                      {formatDelta(scenario.volumeDelta)}
                    </span>
                  </td>

                  {/* Revenue Delta */}
                  <td className="px-6 py-4 text-right">
                    <span className={`font-mono text-sm ${getDeltaColor(scenario.revenueDelta)}`}>
                      {formatDelta(scenario.revenueDelta, true)}
                    </span>
                  </td>

                  {/* Margin Delta */}
                  <td className="px-6 py-4 text-right">
                    <span className={`font-mono text-sm ${getDeltaColor(scenario.marginDelta)}`}>
                      {formatDelta(scenario.marginDelta, true)}
                    </span>
                  </td>

                  {/* Delete */}
                  <td className="px-4 py-4">
                    <motion.button
                      className="p-1.5 rounded-lg hover:bg-[#f43f5e]/20 text-white/30 hover:text-[#f43f5e] transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete?.(scenario.id);
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      transition={springTransition}
                    >
                      <TrashIcon className="w-4 h-4" />
                    </motion.button>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {data.length === 0 && (
        <div className="p-12 text-center">
          <p className="text-white/40 text-sm">No saved scenarios yet.</p>
          <p className="text-white/25 text-xs mt-1">Adjust the price slider and save a scenario to compare.</p>
        </div>
      )}
    </motion.div>
  );
}
