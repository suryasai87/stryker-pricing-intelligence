/**
 * useModelPrediction - Custom hook for fetching ML pricing predictions.
 *
 * Calls the /api/v1/simulate-price-change endpoint with a debounced request
 * so that rapid slider movements or input changes do not flood the API.
 *
 * @param {string|null} productId  - Unity Catalog product identifier
 * @param {number|null} priceChangePct - Proposed price change as a percentage (-100..+100)
 * @returns {{ prediction: object|null, loading: boolean, error: string|null }}
 */
import { useState, useEffect, useRef, useCallback } from "react";

const DEBOUNCE_MS = 300;

/**
 * Prediction shape returned by the API:
 * {
 *   predicted_volume_change_pct: number,
 *   predicted_revenue_impact: number,
 *   predicted_margin_impact: number,
 *   confidence_interval: { lower: number, upper: number },
 *   top_sensitivity_factors: Array<{ factor: string, weight: number }>,
 *   competitive_risk_score: number
 * }
 */

export default function useModelPrediction(productId, priceChangePct) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);
  const abortRef = useRef(null);

  const fetchPrediction = useCallback(async (pid, pct) => {
    // Cancel any in-flight request
    if (abortRef.current) {
      abortRef.current.abort();
    }

    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/simulate-price-change", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_id: pid,
          price_change_pct: pct,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          body.detail || `API error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      if (err.name === "AbortError") {
        // Request was intentionally cancelled; ignore
        return;
      }
      setError(err.message || "Failed to fetch prediction");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Clear previous timer on every dependency change
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // Only fire when both inputs are present
    if (productId == null || priceChangePct == null) {
      setPrediction(null);
      setLoading(false);
      setError(null);
      return;
    }

    timerRef.current = setTimeout(() => {
      fetchPrediction(productId, priceChangePct);
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timerRef.current);
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, [productId, priceChangePct, fetchPrediction]);

  return { prediction, loading, error };
}
