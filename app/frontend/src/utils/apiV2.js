/**
 * API Client for Stryker Pricing Intelligence Platform - V2 Endpoints
 *
 * Handles all communication with the FastAPI backend v2 routes.
 * Uses the same auto-detection and error handling pattern as api.js.
 */

const BASE_URL = (() => {
  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
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
// Discount Outliers
// ============================================

export async function fetchDiscountOutliers(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/discount-outliers${qs}`);
}

export async function fetchDiscountOutliersSummary(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/discount-outliers/summary${qs}`);
}

// ============================================
// Price Elasticity
// ============================================

export async function fetchPriceElasticity(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/price-elasticity${qs}`);
}

export async function fetchPriceElasticityDistribution(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/price-elasticity/distribution${qs}`);
}

// ============================================
// Uplift Simulator
// ============================================

export async function fetchPrecomputedUplift(target = 1.0) {
  return apiFetch(`/api/v2/uplift-simulation/precomputed?target=${target}`);
}

export async function runUpliftSimulation(params) {
  return apiFetch('/api/v2/uplift-simulation', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// ============================================
// Top 100 Price Changes
// ============================================

export async function fetchTop100PriceChanges(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/top100-price-changes${qs}`);
}

export async function fetchTop100FilterOptions() {
  return apiFetch('/api/v2/top100-price-changes/filter-options');
}

// ============================================
// AI Recommendations
// ============================================

export async function fetchPricingRecommendations(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/pricing-recommendations${qs}`);
}

export async function fetchPricingRecommendationsSummary(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/pricing-recommendations/summary${qs}`);
}

// ============================================
// External Data
// ============================================

export async function fetchExternalData(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/external-data${qs}`);
}

export async function fetchExternalDataSources() {
  return apiFetch('/api/v2/external-data/sources');
}

export async function uploadExternalData(formData) {
  return apiFetch('/api/v2/external-data/upload', {
    method: 'POST',
    body: formData,
  });
}

// ============================================
// Pricing Scenarios
// ============================================

export async function fetchPricingScenarios(params = {}) {
  const query = new URLSearchParams(params).toString();
  const qs = query ? `?${query}` : '';
  return apiFetch(`/api/v2/pricing-scenarios${qs}`);
}

export async function fetchScenarioUserInfo() {
  return apiFetch('/api/v2/pricing-scenarios/user-info');
}

export async function createPricingScenario(scenario) {
  return apiFetch('/api/v2/pricing-scenarios', {
    method: 'POST',
    body: JSON.stringify(scenario),
  });
}

export async function updateScenarioStatus(scenarioId, status) {
  return apiFetch(`/api/v2/pricing-scenarios/${scenarioId}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}

export default {
  fetchDiscountOutliers,
  fetchDiscountOutliersSummary,
  fetchPriceElasticity,
  fetchPriceElasticityDistribution,
  fetchPrecomputedUplift,
  runUpliftSimulation,
  fetchTop100PriceChanges,
  fetchTop100FilterOptions,
  fetchPricingRecommendations,
  fetchPricingRecommendationsSummary,
  fetchExternalData,
  fetchExternalDataSources,
  uploadExternalData,
  fetchPricingScenarios,
  fetchScenarioUserInfo,
  createPricingScenario,
  updateScenarioStatus,
};
