/**
 * useProducts - Custom hook for the product catalog.
 *
 * Fetches the full product list on mount from /api/v1/products and exposes a
 * local search helper that filters by product name or category without
 * additional network requests.
 *
 * @returns {{
 *   products: Array<object>,
 *   loading: boolean,
 *   error: string|null,
 *   searchProducts: (query: string) => Array<object>
 * }}
 */
import { useState, useEffect, useCallback, useRef } from "react";

export default function useProducts() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const allProductsRef = useRef([]);

  useEffect(() => {
    let cancelled = false;

    const fetchProducts = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch("/api/v1/products");

        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(
            body.detail ||
              `Failed to load products: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json();
        const productList = Array.isArray(data) ? data : data.products ?? [];

        if (!cancelled) {
          allProductsRef.current = productList;
          setProducts(productList);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to fetch products");
          setProducts([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchProducts();

    return () => {
      cancelled = true;
    };
  }, []);

  /**
   * Filter the cached product list locally by name or category.
   *
   * The search is case-insensitive and matches partial strings against both
   * the `name` and `category` fields of each product object.
   *
   * @param {string} query - Search term
   * @returns {Array<object>} Filtered product list
   */
  const searchProducts = useCallback((query) => {
    if (!query || typeof query !== "string" || query.trim() === "") {
      const all = allProductsRef.current;
      setProducts(all);
      return all;
    }

    const lowerQuery = query.toLowerCase().trim();

    const filtered = allProductsRef.current.filter((product) => {
      const name = (product.name || "").toLowerCase();
      const category = (product.category || "").toLowerCase();
      return name.includes(lowerQuery) || category.includes(lowerQuery);
    });

    setProducts(filtered);
    return filtered;
  }, []);

  return { products, loading, error, searchProducts };
}
