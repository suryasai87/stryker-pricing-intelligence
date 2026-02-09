/**
 * Formatting utilities for the Stryker Pricing Intelligence Platform
 *
 * All formatters handle null/undefined gracefully, returning a dash or
 * fallback string to prevent UI crashes from missing data.
 */

/**
 * Format a value as USD currency
 * @param {number|null} val - The numeric value
 * @param {Object} [options] - Formatting options
 * @param {number} [options.decimals=0] - Number of decimal places
 * @param {boolean} [options.compact=false] - Use compact notation (e.g., $1.2M)
 * @returns {string} Formatted currency string
 */
export function formatCurrency(val, { decimals = 0, compact = false } = {}) {
  if (val === null || val === undefined || isNaN(val)) return '--';

  if (compact) {
    return formatCompact(val, { prefix: '$', decimals: 1 });
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

/**
 * Format a value as a percentage
 * @param {number|null} val - The numeric value (e.g., 0.15 for 15% or 15 for 15%)
 * @param {Object} [options] - Formatting options
 * @param {number} [options.decimals=1] - Number of decimal places
 * @param {boolean} [options.isDecimal=false] - If true, val is treated as a decimal (0.15 -> 15%)
 * @returns {string} Formatted percentage string
 */
export function formatPercent(val, { decimals = 1, isDecimal = false } = {}) {
  if (val === null || val === undefined || isNaN(val)) return '--';

  const displayVal = isDecimal ? val * 100 : val;
  return `${displayVal.toFixed(decimals)}%`;
}

/**
 * Format a plain number with thousands separators
 * @param {number|null} val - The numeric value
 * @param {Object} [options] - Formatting options
 * @param {number} [options.decimals=0] - Number of decimal places
 * @returns {string} Formatted number string
 */
export function formatNumber(val, { decimals = 0 } = {}) {
  if (val === null || val === undefined || isNaN(val)) return '--';

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

/**
 * Format a large number in compact notation (K, M, B)
 * @param {number|null} val - The numeric value
 * @param {Object} [options] - Formatting options
 * @param {string} [options.prefix=''] - Prefix (e.g., '$')
 * @param {number} [options.decimals=1] - Number of decimal places
 * @returns {string} Compact formatted string (e.g., "$1.2M", "340K")
 */
export function formatCompact(val, { prefix = '', decimals = 1 } = {}) {
  if (val === null || val === undefined || isNaN(val)) return '--';

  const absVal = Math.abs(val);
  const sign = val < 0 ? '-' : '';

  if (absVal >= 1_000_000_000) {
    return `${sign}${prefix}${(absVal / 1_000_000_000).toFixed(decimals)}B`;
  }
  if (absVal >= 1_000_000) {
    return `${sign}${prefix}${(absVal / 1_000_000).toFixed(decimals)}M`;
  }
  if (absVal >= 1_000) {
    return `${sign}${prefix}${(absVal / 1_000).toFixed(decimals)}K`;
  }

  return `${sign}${prefix}${absVal.toFixed(decimals)}`;
}

/**
 * Format a delta value with +/- prefix and semantic color class name
 * @param {number|null} val - The delta value
 * @param {Object} [options] - Formatting options
 * @param {string} [options.format='percent'] - 'percent', 'currency', or 'number'
 * @param {number} [options.decimals=1] - Number of decimal places
 * @param {boolean} [options.invertColor=false] - If true, negative is green (for costs)
 * @returns {{ text: string, color: string, isPositive: boolean }}
 */
export function formatDelta(val, { format = 'percent', decimals = 1, invertColor = false } = {}) {
  if (val === null || val === undefined || isNaN(val)) {
    return { text: '--', color: 'text-white/50', isPositive: false };
  }

  const isPositive = val > 0;
  const isZero = val === 0;
  const prefix = isPositive ? '+' : '';

  let text;
  switch (format) {
    case 'currency':
      text = `${prefix}${formatCurrency(val, { decimals })}`;
      break;
    case 'number':
      text = `${prefix}${formatNumber(val, { decimals })}`;
      break;
    case 'compact':
      text = `${prefix}${formatCompact(val, { prefix: '$', decimals })}`;
      break;
    case 'percent':
    default:
      text = `${prefix}${val.toFixed(decimals)}%`;
      break;
  }

  let color;
  if (isZero) {
    color = 'text-white/50';
  } else if (invertColor) {
    color = isPositive ? 'text-stryker-danger' : 'text-stryker-success';
  } else {
    color = isPositive ? 'text-stryker-success' : 'text-stryker-danger';
  }

  return { text, color, isPositive };
}

export default {
  formatCurrency,
  formatPercent,
  formatNumber,
  formatCompact,
  formatDelta,
};
