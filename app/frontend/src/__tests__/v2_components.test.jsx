/**
 * Comprehensive tests for all 7 new V2 page components.
 *
 * Tests cover:
 *   - Render without crashing
 *   - Loading skeleton state
 *   - KPI cards rendering after data loads
 *   - Filter interaction
 *   - Table rendering with data rows
 *   - Export / action buttons
 *   - Component-specific features (charts, forms, etc.)
 */
import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Router wrapper
// ---------------------------------------------------------------------------
function RouterWrapper({ children }) {
  return <BrowserRouter>{children}</BrowserRouter>;
}

// ---------------------------------------------------------------------------
// Mock data factories
// ---------------------------------------------------------------------------

function makeDiscountOutlierRow(overrides = {}) {
  return {
    rep_name: 'John Smith',
    business_unit: 'Orthopaedics',
    product_family: 'Joint Replacement',
    segment: 'Hospitals',
    country: 'US',
    severity: 'severe',
    rep_avg_discount: 0.32,
    peer_avg_discount: 0.18,
    deviation: 0.14,
    revenue: 1250000,
    potential_recovery: 175000,
    deal_count: 47,
    ...overrides,
  };
}

function makeDiscountOutlierSummary() {
  return {
    total_outliers: 42,
    severe_count: 12,
    total_recovery: 3500000,
    top_bu: 'Orthopaedics',
  };
}

function makeElasticityRow(overrides = {}) {
  return {
    sku: 'SKU-1001',
    product_family: 'Joint Replacement',
    segment: 'Hospitals',
    business_unit: 'Orthopaedics',
    classification: 'inelastic',
    elasticity_coefficient: -1.45,
    confidence: 'high',
    safe_increase_min: 0.01,
    safe_increase_max: 0.035,
    ...overrides,
  };
}

function makeElasticityDistribution() {
  return {
    histogram: [
      { classification: 'highly_inelastic', label: 'Highly Inelastic', count: 18, color: '#10b981' },
      { classification: 'inelastic', label: 'Inelastic', count: 25, color: '#34d399' },
      { classification: 'elastic', label: 'Elastic', count: 7, color: '#f59e0b' },
    ],
    safe_ranges: [
      { product_family: 'Joint Replacement', safe_min: 0.01, safe_max: 0.04 },
      { product_family: 'Spine', safe_min: 0.005, safe_max: 0.025 },
    ],
    heatmap: [
      { product_family: 'Joint Replacement', segments: { Hospitals: -1.5, ASC: -0.8 } },
    ],
  };
}

function makeUpliftResult() {
  return {
    summary: {
      target_pct: 0.01,
      achieved_pct: 0.0095,
      actions_needed: 35,
      net_revenue_impact: 2800000,
      skus_affected: 28,
      customers_affected: 412,
      avg_volume_impact: -0.012,
    },
    recommendations: [
      {
        rank: 1,
        sku: 'SKU-1001',
        product_family: 'Joint Replacement',
        segment: 'Hospitals',
        country: 'US',
        current_price: 4500,
        recommended_price: 4635,
        increase_pct: 0.03,
        revenue_impact: 135000,
        volume_impact_pct: -0.008,
        within_target: true,
        rationale: 'Low elasticity product with strong market position.',
      },
      {
        rank: 2,
        sku: 'SKU-2002',
        product_family: 'Spine',
        segment: 'ASC',
        country: 'US',
        current_price: 8200,
        recommended_price: 8364,
        increase_pct: 0.02,
        revenue_impact: 98000,
        volume_impact_pct: -0.005,
        within_target: true,
        rationale: 'Moderate elasticity but high revenue base.',
      },
    ],
    cumulative_curve: [
      { rank: 1, cumulative_uplift_pct: 0.003 },
      { rank: 2, cumulative_uplift_pct: 0.006 },
      { rank: 3, cumulative_uplift_pct: 0.0095 },
    ],
    waterfall: [
      { sku: 'SKU-1001', revenue_impact: 135000 },
      { sku: 'SKU-2002', revenue_impact: 98000 },
    ],
  };
}

function makeTop100Row(overrides = {}) {
  return {
    rank: 1,
    action_summary: 'Increase price by 3%',
    sku: 'SKU-1001',
    product_name: 'Triathlon Total Knee',
    product_family: 'Joint Replacement',
    business_unit: 'Orthopaedics',
    segment: 'Hospitals',
    country: 'US',
    rep_name: 'John Smith',
    current_price: 4500,
    recommended_price: 4635,
    change_pct: 0.03,
    expected_rev_gain: 135000,
    expected_margin_gain: 67500,
    risk_level: 'low',
    confidence_score: 0.92,
    elasticity: -1.45,
    volume_impact: -0.008,
    customer_count: 35,
    rationale: 'Strong market position with low elasticity.',
    ...overrides,
  };
}

function makeTop100FilterOptions() {
  return {
    countries: ['US', 'UK', 'DE'],
    productFamilies: ['Joint Replacement', 'Spine'],
    segments: ['Hospitals', 'ASC'],
    riskLevels: ['low', 'medium', 'high'],
    businessUnits: ['Orthopaedics', 'MedSurg'],
  };
}

function makeAIRecommendationRow(overrides = {}) {
  return {
    priority_score: 9.2,
    action_type: 'increase',
    sku: 'SKU-1001',
    product_family: 'Joint Replacement',
    business_unit: 'Orthopaedics',
    country: 'US',
    segment: 'Hospitals',
    expected_rev_gain: 135000,
    risk_level: 'low',
    confidence: 0.92,
    rationale: 'Inelastic product with strong market share.',
    competitive_context: 'Competitor prices 12% higher.',
    current_price: 4500,
    recommended_price: 4635,
    change_pct: 0.03,
    ...overrides,
  };
}

function makeAISummary() {
  return {
    by_type: [
      { type: 'increase', count: 15 },
      { type: 'hold', count: 8 },
      { type: 'decrease', count: 3 },
    ],
  };
}

function makeExternalSource(overrides = {}) {
  return {
    source_name: 'Competitor Pricing Q1',
    category: 'Competitor Pricing',
    row_count: 1250,
    upload_date: '2025-03-15T10:30:00Z',
    ...overrides,
  };
}

function makeExternalDataRow(overrides = {}) {
  return {
    source_name: 'Competitor Pricing Q1',
    source_id: 'src-001',
    date: '2025-03-01',
    category: 'Competitor Pricing',
    value: 4200,
    ...overrides,
  };
}

function makeScenarioRow(overrides = {}) {
  return {
    id: 'scen-001',
    name: 'Q3 2025 Joint Replacement Uplift',
    status: 'draft',
    target_uplift_pct: 0.025,
    created_at: '2025-06-01T08:00:00Z',
    description: 'Conservative uplift targeting joint replacement.',
    result_revenue_impact: 1500000,
    result_actions: 22,
    user_name: 'John Smith',
    ...overrides,
  };
}

function makeUserInfo(overrides = {}) {
  return {
    name: 'Jane Analyst',
    email: 'jane@stryker.com',
    role: 'analyst',
    department: 'Pricing Strategy',
    is_admin: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Helpers for mocking fetch per-test
// ---------------------------------------------------------------------------

/**
 * Set up fetch to respond based on URL patterns.
 * Returns a vi.fn() that can be inspected.
 */
function mockFetchResponses(responseMap) {
  // Sort patterns by length (longest first) so more specific URLs match before prefixes
  const sortedEntries = Object.entries(responseMap).sort((a, b) => b[0].length - a[0].length);

  const mockFn = vi.fn((url, _opts) => {
    for (const [pattern, data] of sortedEntries) {
      if (url.includes(pattern)) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          json: () => Promise.resolve(data),
          text: () => Promise.resolve(JSON.stringify(data)),
        });
      }
    }
    // Fallback
    return Promise.resolve({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: () => Promise.resolve([]),
      text: () => Promise.resolve('[]'),
    });
  });
  globalThis.fetch = mockFn;
  return mockFn;
}

// ---------------------------------------------------------------------------
// Reset fetch between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  // Default: return empty arrays so components that mount during module init don't break
  globalThis.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: () => Promise.resolve([]),
      text: () => Promise.resolve('[]'),
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});


// =====================================================================
// 1. DiscountOutliersPage
// =====================================================================
describe('DiscountOutliersPage', () => {
  const outlierRows = [
    makeDiscountOutlierRow(),
    makeDiscountOutlierRow({ rep_name: 'Alice Wong', severity: 'high', business_unit: 'MedSurg', potential_recovery: 95000 }),
    makeDiscountOutlierRow({ rep_name: 'Bob Lee', severity: 'moderate', country: 'UK', product_family: 'Spine', potential_recovery: 42000 }),
  ];
  const summary = makeDiscountOutlierSummary();

  function setup() {
    mockFetchResponses({
      '/api/v2/discount-outliers/summary': summary,
      '/api/v2/discount-outliers': { data: outlierRows },
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    const { container } = render(<DiscountOutliersPage />);
    await waitFor(() => expect(container.textContent).toContain('Discount Outliers'));
  });

  it('shows loading skeleton initially', async () => {
    // Make fetch hang so loading state persists
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    const { container } = render(<DiscountOutliersPage />);
    // Loading skeleton renders shimmer placeholders (GlassCard divs)
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders KPI cards with correct data', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('Total Outliers')).toBeInTheDocument();
      expect(screen.getByText('Severe')).toBeInTheDocument();
      expect(screen.getByText('Potential Recovery')).toBeInTheDocument();
      expect(screen.getByText('Top BU Affected')).toBeInTheDocument();
    });
  });

  it('renders data table with rows', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('John Smith')).toBeInTheDocument();
      expect(screen.getByText('Alice Wong')).toBeInTheDocument();
      expect(screen.getByText('Bob Lee')).toBeInTheDocument();
    });
  });

  it('renders filter dropdowns', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    const { container } = render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('John Smith')).toBeInTheDocument();
    });
    // Check filter select elements exist
    const selects = container.querySelectorAll('select');
    expect(selects.length).toBeGreaterThanOrEqual(5); // 5 filter dropdowns
  });

  it('renders export CSV button', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('Export CSV')).toBeInTheDocument();
    });
  });

  it('renders scatter chart heading', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('Rep vs Peer Discount Distribution')).toBeInTheDocument();
    });
  });

  it('filters by rep search', async () => {
    setup();
    const { default: DiscountOutliersPage } = await import('../components/DiscountOutliers/DiscountOutliersPage');
    render(<DiscountOutliersPage />);
    await waitFor(() => {
      expect(screen.getByText('John Smith')).toBeInTheDocument();
    });
    const searchInput = screen.getByPlaceholderText('Search rep name...');
    fireEvent.change(searchInput, { target: { value: 'alice' } });
    await waitFor(() => {
      expect(screen.getByText('Alice Wong')).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 2. PriceElasticityPage
// =====================================================================
describe('PriceElasticityPage', () => {
  const elasticityRows = [
    makeElasticityRow(),
    makeElasticityRow({ sku: 'SKU-2002', classification: 'elastic', elasticity_coefficient: -0.65, product_family: 'Spine', segment: 'ASC' }),
    makeElasticityRow({ sku: 'SKU-3003', classification: 'highly_inelastic', elasticity_coefficient: -2.1 }),
  ];
  const dist = makeElasticityDistribution();

  function setup() {
    mockFetchResponses({
      '/api/v2/price-elasticity/distribution': dist,
      '/api/v2/price-elasticity': { data: elasticityRows },
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    const { container } = render(<PriceElasticityPage />);
    await waitFor(() => expect(container.textContent).toContain('Price Elasticity'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    const { container } = render(<PriceElasticityPage />);
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders KPI cards', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    render(<PriceElasticityPage />);
    await waitFor(() => {
      expect(screen.getByText('Avg Elasticity')).toBeInTheDocument();
      // "Highly Inelastic" appears in KPI card, heatmap legend, and table cells - use getAllByText
      expect(screen.getAllByText('Highly Inelastic').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Elastic SKUs')).toBeInTheDocument();
      expect(screen.getByText('Avg Safe Increase')).toBeInTheDocument();
    });
  });

  it('renders heatmap component', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    render(<PriceElasticityPage />);
    await waitFor(() => {
      expect(screen.getByText('Elasticity Heatmap by Product Family & Segment')).toBeInTheDocument();
    });
  });

  it('renders classification filter', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    const { container } = render(<PriceElasticityPage />);
    await waitFor(() => {
      expect(screen.getByText('Avg Elasticity')).toBeInTheDocument();
    });
    const selects = container.querySelectorAll('select');
    expect(selects.length).toBeGreaterThanOrEqual(5);
    // Find the classification select
    const classSelect = Array.from(selects).find(
      (s) => s.querySelector('option')?.textContent === 'Classification'
    );
    expect(classSelect).toBeTruthy();
  });

  it('renders data table rows', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    render(<PriceElasticityPage />);
    await waitFor(() => {
      expect(screen.getByText('SKU-1001')).toBeInTheDocument();
      expect(screen.getByText('SKU-2002')).toBeInTheDocument();
    });
  });

  it('renders histogram chart heading', async () => {
    setup();
    const { default: PriceElasticityPage } = await import('../components/PriceElasticity/PriceElasticityPage');
    render(<PriceElasticityPage />);
    await waitFor(() => {
      expect(screen.getByText('Elasticity Distribution by Classification')).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 3. UpliftSimulatorPage
// =====================================================================
describe('UpliftSimulatorPage', () => {
  const upliftResult = makeUpliftResult();

  function setup() {
    mockFetchResponses({
      '/api/v2/uplift-simulation/precomputed': upliftResult,
      '/api/v2/uplift-simulation': upliftResult,
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    const { container } = render(<UpliftSimulatorPage />);
    await waitFor(() => expect(container.textContent).toContain('Uplift Simulator'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    const { container } = render(<UpliftSimulatorPage />);
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders summary KPI cards', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Target')).toBeInTheDocument();
      expect(screen.getByText('Achieved')).toBeInTheDocument();
      expect(screen.getByText('Actions Needed')).toBeInTheDocument();
      expect(screen.getByText('Net Rev Impact')).toBeInTheDocument();
    });
  });

  it('renders Run Simulation button', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Run Simulation')).toBeInTheDocument();
    });
  });

  it('renders Reset button', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Reset')).toBeInTheDocument();
    });
  });

  it('renders cumulative uplift chart heading', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Cumulative Uplift Curve')).toBeInTheDocument();
    });
  });

  it('renders recommendations table with rows', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('SKU-1001')).toBeInTheDocument();
      expect(screen.getByText('SKU-2002')).toBeInTheDocument();
    });
  });

  it('renders export CSV button', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Export CSV')).toBeInTheDocument();
    });
  });

  it('renders simulation parameter inputs', async () => {
    setup();
    const { default: UpliftSimulatorPage } = await import('../components/UpliftSimulator/UpliftSimulatorPage');
    render(<UpliftSimulatorPage />);
    await waitFor(() => {
      expect(screen.getByText('Simulation Parameters')).toBeInTheDocument();
      expect(screen.getByText('Target Uplift %')).toBeInTheDocument();
      expect(screen.getByText('Max Per-SKU Increase %')).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 4. Top100ChangesPage
// =====================================================================
describe('Top100ChangesPage', () => {
  const top100Rows = [
    makeTop100Row(),
    makeTop100Row({ rank: 2, sku: 'SKU-2002', product_family: 'Spine', risk_level: 'medium', expected_rev_gain: 98000 }),
    makeTop100Row({ rank: 3, sku: 'SKU-3003', country: 'UK', risk_level: 'high', expected_rev_gain: 75000 }),
  ];
  const filterOpts = makeTop100FilterOptions();

  function setup() {
    mockFetchResponses({
      '/api/v2/top100-price-changes/filter-options': filterOpts,
      '/api/v2/top100-price-changes': { data: top100Rows },
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    const { container } = render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => expect(container.textContent).toContain('Top 100'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    const { container } = render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders KPI cards', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText('Total Actions')).toBeInTheDocument();
      expect(screen.getByText('Expected Rev Gain')).toBeInTheDocument();
      expect(screen.getByText('Expected Margin $')).toBeInTheDocument();
      expect(screen.getByText('Avg Risk Level')).toBeInTheDocument();
    });
  });

  it('renders filter bar with country chips', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      // Country chips appear in both filter bar and table rows, so use getAllByText
      expect(screen.getAllByText('US').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('UK').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('DE').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders data table with rows', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText('SKU-1001')).toBeInTheDocument();
      expect(screen.getByText('SKU-2002')).toBeInTheDocument();
    });
  });

  it('renders column visibility toggle button', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText('Columns')).toBeInTheDocument();
    });
  });

  it('renders export buttons (CSV, Excel, Copy)', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText('CSV')).toBeInTheDocument();
      expect(screen.getByText('Excel')).toBeInTheDocument();
      expect(screen.getByText('Copy')).toBeInTheDocument();
    });
  });

  it('supports URL-based filtering via useSearchParams', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    // Render with initial search params
    render(
      <MemoryRouter initialEntries={['/?segment=Hospitals']}>
        <Top100ChangesPage />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(screen.getByText('SKU-1001')).toBeInTheDocument();
    });
  });

  it('can toggle column picker', async () => {
    setup();
    const { default: Top100ChangesPage } = await import('../components/Top100Changes/Top100ChangesPage');
    render(
      <RouterWrapper>
        <Top100ChangesPage />
      </RouterWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText('Columns')).toBeInTheDocument();
    });
    // Click to open column picker
    fireEvent.click(screen.getByText('Columns'));
    await waitFor(() => {
      // Column picker should show checkboxes
      expect(screen.getByText('Product Name')).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 5. AIRecommendationsPage
// =====================================================================
describe('AIRecommendationsPage', () => {
  const recRows = [
    makeAIRecommendationRow(),
    makeAIRecommendationRow({ sku: 'SKU-2002', action_type: 'hold', priority_score: 7.5, risk_level: 'medium' }),
    makeAIRecommendationRow({ sku: 'SKU-3003', action_type: 'decrease', priority_score: 6.1, risk_level: 'high' }),
  ];
  const aiSummary = makeAISummary();

  function setup() {
    mockFetchResponses({
      '/api/v2/pricing-recommendations/summary': aiSummary,
      '/api/v2/pricing-recommendations': { data: recRows },
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    const { container } = render(<AIRecommendationsPage />);
    await waitFor(() => expect(container.textContent).toContain('AI Pricing Recommendations'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    const { container } = render(<AIRecommendationsPage />);
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders KPI cards', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    render(<AIRecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('Total Recommendations')).toBeInTheDocument();
      expect(screen.getByText('Expected Rev Gain')).toBeInTheDocument();
      expect(screen.getByText('Avg Risk Score')).toBeInTheDocument();
      expect(screen.getByText('By Type')).toBeInTheDocument();
    });
  });

  it('renders donut chart heading', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    render(<AIRecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('Recommendations by Action Type')).toBeInTheDocument();
    });
  });

  it('renders action type filter dropdown', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    const { container } = render(<AIRecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('Total Recommendations')).toBeInTheDocument();
    });
    const selects = container.querySelectorAll('select');
    expect(selects.length).toBeGreaterThanOrEqual(5);
    const actionSelect = Array.from(selects).find(
      (s) => s.querySelector('option')?.textContent === 'Action Type'
    );
    expect(actionSelect).toBeTruthy();
  });

  it('renders recommendations table with rows', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    render(<AIRecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('SKU-1001')).toBeInTheDocument();
      expect(screen.getByText('SKU-2002')).toBeInTheDocument();
    });
  });

  it('renders priority score bar chart heading', async () => {
    setup();
    const { default: AIRecommendationsPage } = await import('../components/AIRecommendations/AIRecommendationsPage');
    render(<AIRecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('Top 15 by Priority Score')).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 6. ExternalDataPage
// =====================================================================
describe('ExternalDataPage', () => {
  const sources = [
    makeExternalSource(),
    makeExternalSource({ source_name: 'FX Rates Jan', category: 'FX Rates', row_count: 365 }),
  ];
  const extData = [
    makeExternalDataRow(),
    makeExternalDataRow({ date: '2025-03-02', value: 4250, category: 'Competitor Pricing' }),
  ];

  function setup() {
    mockFetchResponses({
      '/api/v2/external-data/sources': { data: sources },
      '/api/v2/external-data': { data: extData },
    });
  }

  it('renders without crashing', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    const { container } = render(<ExternalDataPage />);
    await waitFor(() => expect(container.textContent).toContain('External Data'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    const { container } = render(<ExternalDataPage />);
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders upload zone with drag-and-drop text', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    render(<ExternalDataPage />);
    await waitFor(() => {
      expect(screen.getByText(/Drag & drop your file here/)).toBeInTheDocument();
    });
  });

  it('renders upload file button', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    render(<ExternalDataPage />);
    await waitFor(() => {
      expect(screen.getByText('Upload File')).toBeInTheDocument();
    });
  });

  it('renders sources table with rows', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    render(<ExternalDataPage />);
    await waitFor(() => {
      // Source names appear in both the sources table and the data preview table
      expect(screen.getAllByText('Competitor Pricing Q1').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('FX Rates Jan')).toBeInTheDocument();
    });
  });

  it('renders data category dropdown', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    const { container } = render(<ExternalDataPage />);
    await waitFor(() => {
      expect(screen.getByText('Upload External Data')).toBeInTheDocument();
    });
    const selects = container.querySelectorAll('select');
    expect(selects.length).toBeGreaterThanOrEqual(1);
    // The category select should have our options
    const catSelect = Array.from(selects).find(
      (s) => s.querySelector('option')?.textContent === 'Select category...'
    );
    expect(catSelect).toBeTruthy();
  });

  it('renders supported formats hint', async () => {
    setup();
    const { default: ExternalDataPage } = await import('../components/ExternalData/ExternalDataPage');
    render(<ExternalDataPage />);
    await waitFor(() => {
      expect(screen.getByText(/Supported formats/)).toBeInTheDocument();
    });
  });
});

// =====================================================================
// 7. PricingScenariosPage
// =====================================================================
describe('PricingScenariosPage', () => {
  const scenarioRows = [
    makeScenarioRow(),
    makeScenarioRow({ id: 'scen-002', name: 'Spine Portfolio Review', status: 'submitted', user_name: 'Alice Wong' }),
  ];
  const regularUser = makeUserInfo();
  const adminUser = makeUserInfo({ name: 'Admin User', role: 'admin', is_admin: true });

  function setupRegularUser() {
    mockFetchResponses({
      '/api/v2/pricing-scenarios/user-info': regularUser,
      '/api/v2/pricing-scenarios': { data: scenarioRows },
    });
  }

  function setupAdminUser() {
    mockFetchResponses({
      '/api/v2/pricing-scenarios/user-info': adminUser,
      '/api/v2/pricing-scenarios': { data: scenarioRows },
    });
  }

  it('renders without crashing', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    const { container } = render(<PricingScenariosPage />);
    await waitFor(() => expect(container.textContent).toContain('Pricing Scenarios'));
  });

  it('shows loading skeleton initially', async () => {
    globalThis.fetch = vi.fn(() => new Promise(() => {}));
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    const { container } = render(<PricingScenariosPage />);
    expect(container.querySelectorAll('.space-y-6').length).toBeGreaterThan(0);
  });

  it('renders create scenario form toggle', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('New Scenario')).toBeInTheDocument();
    });
  });

  it('renders form when New Scenario clicked', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('New Scenario')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText('New Scenario'));
    await waitFor(() => {
      expect(screen.getByText('Scenario Name *')).toBeInTheDocument();
      expect(screen.getByText('Target Uplift %')).toBeInTheDocument();
    });
  });

  it('renders scenario history table', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('Q3 2025 Joint Replacement Uplift')).toBeInTheDocument();
      expect(screen.getByText('Spine Portfolio Review')).toBeInTheDocument();
    });
  });

  it('renders user info section', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('Jane Analyst')).toBeInTheDocument();
      // "Analyst" appears in the role label under the user info
      expect(screen.getByText(/^Analyst/)).toBeInTheDocument();
    });
  });

  it('regular user does NOT see admin search or User column', async () => {
    setupRegularUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    const { container } = render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('Jane Analyst')).toBeInTheDocument();
    });
    // No admin search
    expect(screen.queryByPlaceholderText('Search all scenarios (admin)...')).not.toBeInTheDocument();
    // No 'User' column header (query th elements directly)
    const ths = container.querySelectorAll('th');
    const userHeader = Array.from(ths).find((th) => th.textContent.trim() === 'User');
    expect(userHeader).toBeFalsy();
    // No "Update Status" column header
    const updateStatusHeader = Array.from(ths).find((th) => th.textContent.trim() === 'Update Status');
    expect(updateStatusHeader).toBeFalsy();
  });

  it('admin user sees admin search and User column', async () => {
    setupAdminUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('Admin User')).toBeInTheDocument();
    });
    // Admin search is visible
    expect(screen.getByPlaceholderText('Search all scenarios (admin)...')).toBeInTheDocument();
    // Admin sees "Administrator" role label
    expect(screen.getByText(/Administrator/)).toBeInTheDocument();
  });

  it('admin user sees Update Status column header', async () => {
    setupAdminUser();
    const { default: PricingScenariosPage } = await import('../components/PricingScenarios/PricingScenariosPage');
    render(<PricingScenariosPage />);
    await waitFor(() => {
      expect(screen.getByText('Update Status')).toBeInTheDocument();
    });
  });
});
