import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MagnifyingGlassIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * ProductSelector - Autocomplete search component for selecting products.
 *
 * Props:
 *   selectedProduct - object|null, the currently selected product
 *   onSelect       - function(product), called when a product is selected
 *   apiEndpoint    - string, the API endpoint to fetch products (default: '/api/products')
 */
export default function ProductSelector({ selectedProduct = null, onSelect, apiEndpoint = '/api/v1/products' }) {
  const [query, setQuery] = useState('');
  const [products, setProducts] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  // Fetch products from API
  useEffect(() => {
    let cancelled = false;
    const fetchProducts = async () => {
      setLoading(true);
      try {
        const res = await fetch(apiEndpoint);
        if (!res.ok) throw new Error('Failed to fetch products');
        const data = await res.json();
        if (!cancelled) {
          setProducts(data.products || data || []);
        }
      } catch (err) {
        console.error('ProductSelector fetch error:', err);
        if (!cancelled) {
          // Fallback demo data for development
          setProducts([
            { id: 'SKU-001', name: 'Mako SmartRobotics Hip System', category: 'Joint Replacement', asp: 45200 },
            { id: 'SKU-002', name: 'Triathlon Knee System', category: 'Joint Replacement', asp: 12800 },
            { id: 'SKU-003', name: 'T2 Tibial Nail', category: 'Trauma & Extremities', asp: 3400 },
            { id: 'SKU-004', name: 'Surgimap Spine Planning', category: 'Spine', asp: 8900 },
            { id: 'SKU-005', name: 'Neptune Waste Management', category: 'Instruments', asp: 18500 },
            { id: 'SKU-006', name: 'System 8 Power Tools', category: 'Instruments', asp: 22100 },
            { id: 'SKU-007', name: 'ProCuity Bed System', category: 'Medical/Surgical', asp: 35600 },
            { id: 'SKU-008', name: 'Sage Turn & Position', category: 'Medical/Surgical', asp: 4200 },
          ]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchProducts();
    return () => { cancelled = true; };
  }, [apiEndpoint]);

  // Filter products based on query
  useEffect(() => {
    if (!query.trim()) {
      setFiltered(products);
    } else {
      const lowerQuery = query.toLowerCase();
      setFiltered(
        products.filter(
          (p) =>
            p.name.toLowerCase().includes(lowerQuery) ||
            p.category.toLowerCase().includes(lowerQuery) ||
            (p.id && p.id.toLowerCase().includes(lowerQuery))
        )
      );
    }
    setHighlightIndex(-1);
  }, [query, products]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = useCallback(
    (product) => {
      onSelect?.(product);
      setQuery('');
      setIsOpen(false);
    },
    [onSelect]
  );

  const handleKeyDown = (e) => {
    if (!isOpen) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightIndex((prev) => Math.min(prev + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && highlightIndex >= 0) {
      e.preventDefault();
      handleSelect(filtered[highlightIndex]);
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  const formatCurrency = (value) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      {/* Input Field */}
      <motion.div
        className={`
          flex items-center gap-2 px-4 py-3 rounded-2xl transition-colors
          bg-white/5 backdrop-blur-xl border
          ${isOpen ? 'border-[#0057B8]/50 bg-white/10' : 'border-white/10'}
        `}
        whileHover={{ borderColor: 'rgba(0, 87, 184, 0.3)' }}
        transition={springTransition}
      >
        <MagnifyingGlassIcon className="w-5 h-5 text-white/40 shrink-0" />
        <input
          ref={inputRef}
          type="text"
          placeholder={selectedProduct ? selectedProduct.name : 'Search products, SKUs...'}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-transparent text-white text-sm placeholder-white/30 outline-none"
        />
        {selectedProduct && !query && (
          <span className="text-xs font-mono text-[#FFB81C]">{formatCurrency(selectedProduct.asp)}</span>
        )}
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={springTransition}
        >
          <ChevronDownIcon className="w-4 h-4 text-white/40" />
        </motion.div>
      </motion.div>

      {/* Selected Product Display */}
      {selectedProduct && !isOpen && (
        <motion.div
          className="mt-2 px-4 py-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl flex items-center justify-between"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={springTransition}
        >
          <div>
            <p className="text-white text-sm font-medium">{selectedProduct.name}</p>
            <p className="text-white/40 text-xs">{selectedProduct.category}</p>
          </div>
          <span className="font-mono text-sm text-[#FFB81C]">{formatCurrency(selectedProduct.asp)}</span>
        </motion.div>
      )}

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute top-full left-0 right-0 mt-2 z-50 bg-slate-800/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden max-h-80 overflow-y-auto"
            initial={{ opacity: 0, y: -10, scaleY: 0.95 }}
            animate={{ opacity: 1, y: 0, scaleY: 1 }}
            exit={{ opacity: 0, y: -10, scaleY: 0.95 }}
            transition={springTransition}
            style={{ transformOrigin: 'top' }}
          >
            {loading ? (
              <div className="p-4 space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="animate-pulse flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-white/10" />
                    <div className="flex-1 h-4 bg-white/10 rounded" />
                    <div className="w-16 h-4 bg-white/10 rounded" />
                  </div>
                ))}
              </div>
            ) : filtered.length === 0 ? (
              <div className="p-4 text-center text-white/40 text-sm">
                No products found for &quot;{query}&quot;
              </div>
            ) : (
              <div className="py-2">
                {filtered.map((product, index) => (
                  <motion.button
                    key={product.id}
                    className={`
                      w-full px-4 py-3 flex items-center justify-between text-left transition-colors
                      ${highlightIndex === index ? 'bg-[#0057B8]/20' : 'hover:bg-white/5'}
                      ${selectedProduct?.id === product.id ? 'bg-[#0057B8]/10' : ''}
                    `}
                    onClick={() => handleSelect(product)}
                    onMouseEnter={() => setHighlightIndex(index)}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ ...springTransition, delay: index * 0.03 }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: selectedProduct?.id === product.id ? '#0057B8' : '#FFB81C' }}
                      />
                      <div>
                        <p className="text-white text-sm font-medium">{product.name}</p>
                        <p className="text-white/40 text-xs">{product.category}</p>
                      </div>
                    </div>
                    <span className="font-mono text-xs text-white/60">{formatCurrency(product.asp)}</span>
                  </motion.button>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
