/**
 * usePortfolioData - Custom hook for dashboard-level portfolio KPIs.
 *
 * Fetches aggregate KPI data from /api/v1/portfolio-kpis on mount and
 * automatically refreshes every 5 minutes.  Exposes a manual refresh
 * function so the UI can trigger an on-demand reload.
 *
 * @returns {{
 *   kpis: object|null,
 *   loading: boolean,
 *   error: string|null,
 *   refresh: () => void
 * }}
 */
import { useState, useEffect, useCallback, useRef } from "react";

const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

export default function usePortfolioData() {
  const [kpis, setKpis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);
  const mountedRef = useRef(true);

  const fetchKpis = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/portfolio-kpis");

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          body.detail ||
            `Failed to load portfolio KPIs: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();

      if (mountedRef.current) {
        setKpis(data);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err.message || "Failed to fetch portfolio KPIs");
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  // Initial fetch and auto-refresh interval
  useEffect(() => {
    mountedRef.current = true;

    fetchKpis();

    intervalRef.current = setInterval(() => {
      fetchKpis();
    }, REFRESH_INTERVAL_MS);

    return () => {
      mountedRef.current = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchKpis]);

  /**
   * Manually trigger a KPI refresh.  Resets the auto-refresh timer so the
   * next automatic refresh happens a full interval after the manual one.
   */
  const refresh = useCallback(() => {
    // Reset the interval so we get a full 5 min from this manual refresh
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    fetchKpis();

    intervalRef.current = setInterval(() => {
      fetchKpis();
    }, REFRESH_INTERVAL_MS);
  }, [fetchKpis]);

  return { kpis, loading, error, refresh };
}
