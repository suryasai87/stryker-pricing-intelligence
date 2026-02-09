/**
 * API Client for Stryker Pricing Intelligence Platform
 *
 * Handles all communication with the FastAPI backend.
 * Auto-detects base URL for both local development and Databricks deployment.
 */

// Auto-detect base URL: in production the API is on the same origin;
// during local dev, Vite proxies /api to localhost:8000
const BASE_URL = (() => {
  if (typeof window !== 'undefined') {
    // If running inside Databricks Apps, the API is on the same origin
    const { protocol, hostname, port } = window.location;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      // Local dev -- Vite proxy handles /api -> localhost:8000
      return '';
    }
    return `${protocol}//${hostname}${port ? ':' + port : ''}`;
  }
  return '';
})();

/**
 * Generic fetch wrapper with error handling and JSON parsing
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;

  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  const config = {
    headers: { ...defaultHeaders, ...options.headers },
    ...options,
  };

  // Remove Content-Type for FormData
  if (options.body instanceof FormData) {
    delete config.headers['Content-Type'];
  }

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      const errorBody = await response.text();
      let errorMessage;
      try {
        const errorJson = JSON.parse(errorBody);
        errorMessage = errorJson.detail || errorJson.message || errorBody;
      } catch {
        errorMessage = errorBody || `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    // Handle empty responses (204 No Content, etc.)
    if (response.status === 204 || response.headers.get('content-length') === '0') {
      return null;
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to reach the server. Please check your connection.');
    }
    throw error;
  }
}

// ============================================
// Products & Portfolio
// ============================================

/**
 * Fetch all products in the portfolio
 * @returns {Promise<Array>} List of product objects
 */
export async function fetchProducts() {
  return apiFetch('/api/products');
}

/**
 * Fetch portfolio-level KPIs (revenue, margin, market share, etc.)
 * @returns {Promise<Object>} KPI data object
 */
export async function fetchPortfolioKPIs() {
  return apiFetch('/api/portfolio/kpis');
}

// ============================================
// Price Simulator
// ============================================

/**
 * Simulate a price change and get projected impact
 * @param {Object} request - Simulation parameters
 * @param {string} request.product_id - Product identifier
 * @param {number} request.price_change_pct - Percentage change (e.g., 5.0 for +5%)
 * @param {string} [request.scenario_name] - Optional scenario label
 * @returns {Promise<Object>} Simulation results with revenue/margin/volume projections
 */
export async function simulatePriceChange(request) {
  return apiFetch('/api/simulate/price', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Submit a batch of pricing scenarios for comparison
 * @param {Array<Object>} scenarios - Array of simulation requests
 * @returns {Promise<Object>} Batch simulation results
 */
export async function submitBatchScenario(scenarios) {
  return apiFetch('/api/simulate/batch', {
    method: 'POST',
    body: JSON.stringify({ scenarios }),
  });
}

// ============================================
// Price Waterfall
// ============================================

/**
 * Fetch price waterfall breakdown for a product
 * @param {string} productId - Product identifier
 * @returns {Promise<Object>} Waterfall data with list price -> net price steps
 */
export async function fetchPriceWaterfall(productId) {
  const params = productId ? `?product_id=${encodeURIComponent(productId)}` : '';
  return apiFetch(`/api/waterfall${params}`);
}

// ============================================
// Competitive Landscape
// ============================================

/**
 * Fetch competitive landscape data for a product category
 * @param {string} [category] - Product category filter
 * @returns {Promise<Object>} Competitor pricing and market share data
 */
export async function fetchCompetitiveLandscape(category) {
  const params = category ? `?category=${encodeURIComponent(category)}` : '';
  return apiFetch(`/api/competitive${params}`);
}

// ============================================
// External Factors
// ============================================

/**
 * Fetch external market factors affecting pricing
 * @returns {Promise<Object>} External factors data (inflation, FX rates, supply chain, etc.)
 */
export async function fetchExternalFactors() {
  return apiFetch('/api/external-factors');
}

// ============================================
// Health Check
// ============================================

/**
 * Check API health/connectivity
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  return apiFetch('/api/health');
}

export default {
  fetchProducts,
  fetchPortfolioKPIs,
  simulatePriceChange,
  submitBatchScenario,
  fetchPriceWaterfall,
  fetchCompetitiveLandscape,
  fetchExternalFactors,
  healthCheck,
};
