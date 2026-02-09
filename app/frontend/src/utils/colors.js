/**
 * Theme color constants for the Stryker Pricing Intelligence Platform
 *
 * These values mirror the Tailwind config and are intended for use in
 * JavaScript contexts where Tailwind classes are not available, such as
 * Recharts fill/stroke props, D3 scales, and inline style calculations.
 */

export const COLORS = {
  // Brand / Primary palette
  primary: '#0057B8',
  primaryLight: '#3B82F6',
  primaryDark: '#003D82',

  // Accent (Gold)
  accent: '#FFB81C',
  accentLight: '#FBBF24',
  accentDark: '#D97706',

  // Semantic colors
  success: '#10B981',
  successLight: '#34D399',
  successDark: '#059669',

  danger: '#F43F5E',
  dangerLight: '#FB7185',
  dangerDark: '#E11D48',

  warning: '#F59E0B',
  warningLight: '#FBBF24',
  warningDark: '#D97706',

  info: '#06B6D4',
  infoLight: '#22D3EE',
  infoDark: '#0891B2',

  // Background / Surface
  background: '#0F172A',
  surface: '#1E293B',
  surfaceLight: '#334155',
  surfaceHover: '#475569',

  // Text
  text: '#FFFFFF',
  textSecondary: 'rgba(255, 255, 255, 0.7)',
  textTertiary: 'rgba(255, 255, 255, 0.5)',
  textMuted: 'rgba(255, 255, 255, 0.3)',

  // Borders
  border: 'rgba(255, 255, 255, 0.1)',
  borderHover: 'rgba(255, 255, 255, 0.2)',

  // Glass morphism
  glass: 'rgba(255, 255, 255, 0.05)',
  glassHover: 'rgba(255, 255, 255, 0.08)',
  glassBorder: 'rgba(255, 255, 255, 0.1)',
};

/**
 * Chart color palette for multi-series visualizations
 * Ordered for maximum visual distinction in charts and graphs
 */
export const CHART_PALETTE = [
  COLORS.primary,
  COLORS.accent,
  COLORS.success,
  COLORS.danger,
  COLORS.info,
  COLORS.warning,
  COLORS.primaryLight,
  COLORS.accentLight,
  COLORS.successLight,
  COLORS.dangerLight,
];

/**
 * Gradient definitions for chart areas and backgrounds
 */
export const GRADIENTS = {
  primary: {
    start: COLORS.primary,
    end: 'rgba(0, 87, 184, 0.05)',
  },
  accent: {
    start: COLORS.accent,
    end: 'rgba(255, 184, 28, 0.05)',
  },
  success: {
    start: COLORS.success,
    end: 'rgba(16, 185, 129, 0.05)',
  },
  danger: {
    start: COLORS.danger,
    end: 'rgba(244, 63, 94, 0.05)',
  },
};

/**
 * Common Recharts theme overrides for consistent chart styling
 */
export const RECHARTS_THEME = {
  // Axis styling
  axisLine: { stroke: COLORS.border },
  tickLine: { stroke: COLORS.border },
  tick: { fill: COLORS.textMuted, fontSize: 11, fontFamily: 'JetBrains Mono' },

  // Grid styling
  grid: { stroke: COLORS.border, strokeDasharray: '3 3' },

  // Tooltip styling
  tooltip: {
    contentStyle: {
      backgroundColor: COLORS.surface,
      border: `1px solid ${COLORS.glassBorder}`,
      borderRadius: '8px',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
      padding: '12px',
      fontFamily: 'Inter, sans-serif',
      fontSize: '12px',
    },
    labelStyle: {
      color: COLORS.textSecondary,
      fontWeight: 600,
      marginBottom: '4px',
    },
    itemStyle: {
      color: COLORS.text,
      padding: '2px 0',
    },
  },

  // Legend styling
  legend: {
    wrapperStyle: {
      fontFamily: 'Inter, sans-serif',
      fontSize: '12px',
      color: COLORS.textTertiary,
    },
  },
};

export default COLORS;
