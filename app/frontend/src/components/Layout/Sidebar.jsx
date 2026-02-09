import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChartBarIcon,
  CalculatorIcon,
  FunnelIcon,
  ShieldCheckIcon,
  GlobeAltIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: ChartBarIcon, path: '/' },
  { id: 'simulator', label: 'Price Simulator', icon: CalculatorIcon, path: '/simulator' },
  { id: 'waterfall', label: 'Price Waterfall', icon: FunnelIcon, path: '/waterfall' },
  { id: 'competitive', label: 'Competitive', icon: ShieldCheckIcon, path: '/competitive' },
  { id: 'external', label: 'External Factors', icon: GlobeAltIcon, path: '/external' },
];

/**
 * Sidebar - Animated sidebar navigation for the Stryker Pricing Intelligence Platform.
 *
 * Props:
 *   activeItem  - string, the id of the currently active nav item (default: 'dashboard')
 *   onNavigate  - function(id, path), called when a nav item is clicked
 */
export default function Sidebar({ activeItem = 'dashboard', onNavigate }) {
  const [collapsed, setCollapsed] = useState(false);

  const sidebarVariants = {
    expanded: { width: 260 },
    collapsed: { width: 72 },
  };

  return (
    <motion.aside
      className="h-screen bg-slate-900 border-r border-white/10 flex flex-col overflow-hidden relative z-30"
      variants={sidebarVariants}
      animate={collapsed ? 'collapsed' : 'expanded'}
      transition={springTransition}
    >
      {/* Brand / Logo */}
      <div className="flex items-center gap-3 px-4 py-6 border-b border-white/10">
        <motion.div
          className="w-10 h-10 rounded-xl flex items-center justify-center font-bold text-white text-lg shrink-0"
          style={{ backgroundColor: '#0057B8' }}
          whileHover={{ scale: 1.08 }}
          transition={springTransition}
        >
          S
        </motion.div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={springTransition}
              className="overflow-hidden whitespace-nowrap"
            >
              <p className="text-white font-semibold text-sm leading-tight">Stryker</p>
              <p className="text-white/50 text-xs leading-tight">Pricing Intelligence</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4 px-2 space-y-1">
        {navItems.map((item) => {
          const isActive = activeItem === item.id;
          const Icon = item.icon;

          return (
            <motion.button
              key={item.id}
              onClick={() => onNavigate?.(item.id, item.path)}
              className={`
                relative w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
                transition-colors cursor-pointer
                ${isActive ? 'text-white' : 'text-white/50 hover:text-white/80'}
              `}
              whileHover={{ x: 4 }}
              transition={springTransition}
            >
              {/* Active indicator background */}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-bg"
                  className="absolute inset-0 rounded-xl"
                  style={{ backgroundColor: 'rgba(0, 87, 184, 0.2)', border: '1px solid rgba(0, 87, 184, 0.3)' }}
                  transition={springTransition}
                />
              )}

              {/* Active left bar */}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-bar"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full"
                  style={{ backgroundColor: '#0057B8' }}
                  transition={springTransition}
                />
              )}

              <Icon className="w-5 h-5 shrink-0 relative z-10" />

              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -8 }}
                    transition={springTransition}
                    className="relative z-10 whitespace-nowrap"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          );
        })}
      </nav>

      {/* Collapse Toggle */}
      <div className="p-3 border-t border-white/10">
        <motion.button
          onClick={() => setCollapsed((prev) => !prev)}
          className="w-full flex items-center justify-center p-2 rounded-xl text-white/40 hover:text-white/80 hover:bg-white/5 transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={springTransition}
        >
          {collapsed ? (
            <ChevronRightIcon className="w-5 h-5" />
          ) : (
            <ChevronLeftIcon className="w-5 h-5" />
          )}
        </motion.button>
      </div>
    </motion.aside>
  );
}
