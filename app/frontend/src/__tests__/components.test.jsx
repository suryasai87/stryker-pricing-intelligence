/**
 * Smoke tests for all major React components.
 * Ensures each component renders without crashing.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, vi } from 'vitest';

// Wrapper that provides Router context for components that need it
function RouterWrapper({ children }) {
  return <BrowserRouter>{children}</BrowserRouter>;
}

// =====================================================================
// Shared Components
// =====================================================================
describe('Shared Components', () => {
  it('GlassCard renders without crashing', async () => {
    const { default: GlassCard } = await import('../components/shared/GlassCard');
    const { container } = render(<GlassCard>Test Content</GlassCard>);
    expect(container).toBeTruthy();
    expect(container.textContent).toContain('Test Content');
  });

  it('AnimatedNumber renders without crashing', async () => {
    const { default: AnimatedNumber } = await import('../components/shared/AnimatedNumber');
    const { container } = render(<AnimatedNumber value={42} format="number" />);
    expect(container).toBeTruthy();
  });

  it('LoadingShimmer renders without crashing', async () => {
    const { default: LoadingShimmer } = await import('../components/shared/LoadingShimmer');
    const { container } = render(<LoadingShimmer />);
    expect(container).toBeTruthy();
  });

  it('LoadingShimmer renders multiple lines', async () => {
    const { default: LoadingShimmer } = await import('../components/shared/LoadingShimmer');
    const { container } = render(<LoadingShimmer count={3} />);
    expect(container).toBeTruthy();
  });

  it('Tooltip renders without crashing', async () => {
    const { default: Tooltip } = await import('../components/shared/Tooltip');
    const { container } = render(
      <Tooltip content="Tip text">
        <span>Hover me</span>
      </Tooltip>
    );
    expect(container).toBeTruthy();
    expect(container.textContent).toContain('Hover me');
  });
});

// =====================================================================
// Layout Components
// =====================================================================
describe('Layout Components', () => {
  it('Sidebar renders without crashing', async () => {
    const { default: Sidebar } = await import('../components/Layout/Sidebar');
    const { container } = render(
      <RouterWrapper>
        <Sidebar isOpen={true} onClose={() => {}} />
      </RouterWrapper>
    );
    expect(container).toBeTruthy();
  });

  it('TopBar renders without crashing', async () => {
    const { default: TopBar } = await import('../components/Layout/TopBar');
    const { container } = render(<TopBar onMenuToggle={() => {}} />);
    expect(container).toBeTruthy();
  });

  it('PageTransition renders without crashing', async () => {
    const { default: PageTransition } = await import('../components/Layout/PageTransition');
    const { container } = render(
      <PageTransition pageKey="test">
        <div>Page content</div>
      </PageTransition>
    );
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// Dashboard Components
// =====================================================================
describe('Dashboard Components', () => {
  it('KPICard renders without crashing', async () => {
    const { default: KPICard } = await import('../components/Dashboard/KPICard');
    const MockIcon = () => <svg />;
    const { container } = render(
      <KPICard
        icon={MockIcon}
        title="Test KPI"
        value={1234}
        format="number"
        delta={5.2}
        deltaLabel="vs last year"
        sparkline={[{ value: 1 }, { value: 2 }, { value: 3 }]}
        index={0}
      />
    );
    expect(container).toBeTruthy();
  });

  it('RevenueTreemap renders without crashing', async () => {
    const { default: RevenueTreemap } = await import('../components/Dashboard/RevenueTreemap');
    const data = [
      { name: 'Ortho', value: 6200, growth: 5.2, color: '#0057B8' },
      { name: 'MedSurg', value: 5100, growth: 7.8, color: '#10b981' },
    ];
    const { container } = render(<RevenueTreemap data={data} />);
    expect(container).toBeTruthy();
  });

  it('PriceHeatmap renders without crashing', async () => {
    const { default: PriceHeatmap } = await import('../components/Dashboard/PriceHeatmap');
    const data = [
      { category: 'Hip Implants', values: [2.1, 1.8, 3.0, 2.5, 1.2, 0.8, -0.5, 1.0, 2.3, 3.1, 2.8, 1.5] },
    ];
    const { container } = render(<PriceHeatmap data={data} />);
    expect(container).toBeTruthy();
  });

  it('TrendChart renders without crashing', async () => {
    const { default: TrendChart } = await import('../components/Dashboard/TrendChart');
    const data = [
      { month: 'Jan 24', revenue: 1450, margin: 62 },
      { month: 'Feb 24', revenue: 1520, margin: 63 },
    ];
    const { container } = render(<TrendChart data={data} />);
    expect(container).toBeTruthy();
  });

  it('DashboardPage renders without crashing', async () => {
    const { default: DashboardPage } = await import('../components/Dashboard/DashboardPage');
    const { container } = render(<DashboardPage />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// Simulator Components
// =====================================================================
describe('Simulator Components', () => {
  it('ProductSelector renders without crashing', async () => {
    const { default: ProductSelector } = await import('../components/Simulator/ProductSelector');
    const { container } = render(<ProductSelector onSelect={() => {}} />);
    expect(container).toBeTruthy();
  });

  it('PriceSlider renders without crashing', async () => {
    const { default: PriceSlider } = await import('../components/Simulator/PriceSlider');
    const { container } = render(<PriceSlider value={0} onChange={() => {}} />);
    expect(container).toBeTruthy();
  });

  it('ImpactPanel renders without crashing', async () => {
    const { default: ImpactPanel } = await import('../components/Simulator/ImpactPanel');
    const { container } = render(<ImpactPanel loading={false} prediction={null} />);
    expect(container).toBeTruthy();
  });

  it('SensitivityTornado renders without crashing', async () => {
    const { default: SensitivityTornado } = await import('../components/Simulator/SensitivityTornado');
    const factors = [
      { feature: 'price_delta', impact: -0.5 },
      { feature: 'market_share', impact: 0.3 },
    ];
    const { container } = render(<SensitivityTornado factors={factors} />);
    expect(container).toBeTruthy();
  });

  it('ScenarioTable renders without crashing', async () => {
    const { default: ScenarioTable } = await import('../components/Simulator/ScenarioTable');
    const { container } = render(<ScenarioTable scenarios={[]} onDelete={() => {}} />);
    expect(container).toBeTruthy();
  });

  it('ConfidenceBands renders without crashing', async () => {
    const { default: ConfidenceBands } = await import('../components/Simulator/ConfidenceBands');
    const { container } = render(<ConfidenceBands />);
    expect(container).toBeTruthy();
  });

  it('SimulatorPage renders without crashing', async () => {
    const { default: SimulatorPage } = await import('../components/Simulator/SimulatorPage');
    const { container } = render(<SimulatorPage />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// Waterfall Components
// =====================================================================
describe('Waterfall Components', () => {
  it('WaterfallChart renders without crashing', async () => {
    const { default: WaterfallChart } = await import('../components/Waterfall/WaterfallChart');
    const { container } = render(<WaterfallChart />);
    expect(container).toBeTruthy();
  });

  it('MarginLeakTable renders without crashing', async () => {
    const { default: MarginLeakTable } = await import('../components/Waterfall/MarginLeakTable');
    const { container } = render(<MarginLeakTable />);
    expect(container).toBeTruthy();
  });

  it('WaterfallPage renders without crashing', async () => {
    const { default: WaterfallPage } = await import('../components/Waterfall/WaterfallPage');
    const { container } = render(<WaterfallPage />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// Competitive Components
// =====================================================================
describe('Competitive Components', () => {
  it('ASPGapChart renders without crashing', async () => {
    const { default: ASPGapChart } = await import('../components/Competitive/ASPGapChart');
    const { container } = render(<ASPGapChart />);
    expect(container).toBeTruthy();
  });

  it('MarketShareTrend renders without crashing', async () => {
    const { default: MarketShareTrend } = await import('../components/Competitive/MarketShareTrend');
    const { container } = render(<MarketShareTrend />);
    expect(container).toBeTruthy();
  });

  it('PatentTimeline renders without crashing', async () => {
    const { default: PatentTimeline } = await import('../components/Competitive/PatentTimeline');
    const { container } = render(<PatentTimeline />);
    expect(container).toBeTruthy();
  });

  it('CompetitivePage renders without crashing', async () => {
    const { default: CompetitivePage } = await import('../components/Competitive/CompetitivePage');
    const { container } = render(<CompetitivePage />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// External Factors Components
// =====================================================================
describe('External Factors Components', () => {
  it('TariffDashboard renders without crashing', async () => {
    const { default: TariffDashboard } = await import('../components/ExternalFactors/TariffDashboard');
    const { container } = render(<TariffDashboard />);
    expect(container).toBeTruthy();
  });

  it('MacroGauge renders without crashing', async () => {
    const { default: MacroGauge } = await import('../components/ExternalFactors/MacroGauge');
    const { container } = render(<MacroGauge value={65} label="CPI" />);
    expect(container).toBeTruthy();
  });

  it('CurrencyCalculator renders without crashing', async () => {
    const { default: CurrencyCalculator } = await import('../components/ExternalFactors/CurrencyCalculator');
    const { container } = render(<CurrencyCalculator />);
    expect(container).toBeTruthy();
  });

  it('ExternalFactorsPage renders without crashing', async () => {
    const { default: ExternalFactorsPage } = await import('../components/ExternalFactors/ExternalFactorsPage');
    const { container } = render(<ExternalFactorsPage />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// Page Components
// =====================================================================
describe('Page Components', () => {
  it('Dashboard page renders without crashing', async () => {
    const { default: Dashboard } = await import('../pages/Dashboard');
    const { container } = render(<Dashboard />);
    expect(container).toBeTruthy();
  });

  it('PriceSimulator page renders without crashing', async () => {
    const { default: PriceSimulator } = await import('../pages/PriceSimulator');
    const { container } = render(<PriceSimulator />);
    expect(container).toBeTruthy();
  });

  it('PriceWaterfall page renders without crashing', async () => {
    const { default: PriceWaterfall } = await import('../pages/PriceWaterfall');
    const { container } = render(<PriceWaterfall />);
    expect(container).toBeTruthy();
  });

  it('CompetitiveLandscape page renders without crashing', async () => {
    const { default: CompetitiveLandscape } = await import('../pages/CompetitiveLandscape');
    const { container } = render(<CompetitiveLandscape />);
    expect(container).toBeTruthy();
  });

  it('ExternalFactors page renders without crashing', async () => {
    const { default: ExternalFactors } = await import('../pages/ExternalFactors');
    const { container } = render(<ExternalFactors />);
    expect(container).toBeTruthy();
  });
});

// =====================================================================
// App Component (with Router)
// =====================================================================
describe('App Component', () => {
  it('App renders without crashing', async () => {
    const { default: App } = await import('../App');
    const { container } = render(
      <RouterWrapper>
        <App />
      </RouterWrapper>
    );
    expect(container).toBeTruthy();
    expect(container.textContent).toContain('Stryker Pricing');
  });
});

// =====================================================================
// Utility Functions
// =====================================================================
describe('Utility Functions', () => {
  it('formatters work correctly', async () => {
    const { formatCurrency, formatPercent, formatNumber, formatCompact, formatDelta } = await import('../utils/formatters');

    expect(formatCurrency(1234.56)).toContain('1,23');
    expect(formatPercent(0.85)).toContain('0.8');
    expect(formatNumber(50000)).toContain('50,000');
    expect(formatCompact(1500000)).toBeTruthy();

    const delta = formatDelta(5.5);
    expect(delta.text).toContain('+');
    expect(delta.isPositive).toBe(true);

    const negativeDelta = formatDelta(-3.2);
    expect(negativeDelta.isPositive).toBe(false);

    // Null handling
    expect(formatCurrency(null)).toBe('--');
    expect(formatPercent(undefined)).toBe('--');
    expect(formatNumber(NaN)).toBe('--');
  });

  it('colors exports are valid', async () => {
    const { COLORS, CHART_PALETTE, RECHARTS_THEME } = await import('../utils/colors');
    expect(COLORS.primary).toBe('#0057B8');
    expect(COLORS.accent).toBe('#FFB81C');
    expect(CHART_PALETTE.length).toBeGreaterThanOrEqual(5);
    expect(RECHARTS_THEME).toBeTruthy();
  });

  it('api module exports functions', async () => {
    const api = await import('../utils/api');
    expect(typeof api.fetchProducts).toBe('function');
    expect(typeof api.fetchPortfolioKPIs).toBe('function');
    expect(typeof api.simulatePriceChange).toBe('function');
    expect(typeof api.healthCheck).toBe('function');
  });
});

// =====================================================================
// Hooks
// =====================================================================
describe('Hooks', () => {
  it('useProducts hook module exports correctly', async () => {
    const mod = await import('../hooks/useProducts');
    expect(typeof mod.default).toBe('function');
  });

  it('useModelPrediction hook module exports correctly', async () => {
    const mod = await import('../hooks/useModelPrediction');
    expect(typeof mod.default).toBe('function');
  });

  it('usePortfolioData hook module exports correctly', async () => {
    const mod = await import('../hooks/usePortfolioData');
    expect(typeof mod.default).toBe('function');
  });
});
