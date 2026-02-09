import React, { useState, lazy, Suspense } from 'react';
import { Routes, Route, useLocation, NavLink } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import clsx from 'clsx';
import {
  ChartBarIcon,
  CalculatorIcon,
  ArrowTrendingUpIcon,
  GlobeAltIcon,
  BoltIcon,
  Bars3Icon,
  XMarkIcon,
  CurrencyDollarIcon,
  BellIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';

// Lazy-loaded page components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const PriceSimulator = lazy(() => import('./pages/PriceSimulator'));
const PriceWaterfall = lazy(() => import('./pages/PriceWaterfall'));
const CompetitiveLandscape = lazy(() => import('./pages/CompetitiveLandscape'));
const ExternalFactors = lazy(() => import('./pages/ExternalFactors'));

// Spring animation config used across all transitions
const springTransition = {
  type: 'spring',
  stiffness: 300,
  damping: 30,
};

// Page transition variants
const pageVariants = {
  initial: { opacity: 0, y: 20, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: -20, scale: 0.98 },
};

// Navigation items
const navItems = [
  { path: '/', label: 'Dashboard', icon: ChartBarIcon },
  { path: '/simulator', label: 'Price Simulator', icon: CalculatorIcon },
  { path: '/waterfall', label: 'Price Waterfall', icon: ArrowTrendingUpIcon },
  { path: '/competitive', label: 'Competitive Landscape', icon: GlobeAltIcon },
  { path: '/external', label: 'External Factors', icon: BoltIcon },
];

// Loading spinner for lazy-loaded pages
function PageLoader() {
  return (
    <div className="flex items-center justify-center h-full min-h-[60vh]">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        className="w-10 h-10 border-3 border-stryker-primary border-t-transparent rounded-full"
      />
    </div>
  );
}

// Sidebar component
function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={springTransition}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        className={clsx(
          'fixed top-0 left-0 z-50 h-full w-64',
          'bg-white/5 backdrop-blur-xl border-r border-white/10',
          'flex flex-col',
          'lg:translate-x-0 lg:static lg:z-auto',
          !isOpen && '-translate-x-full lg:translate-x-0'
        )}
        initial={false}
        animate={{ x: isOpen ? 0 : undefined }}
        transition={springTransition}
      >
        {/* Logo / Brand */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-white/10">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-stryker-primary to-stryker-accent flex items-center justify-center">
            <CurrencyDollarIcon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight">Stryker Pricing</h1>
            <p className="text-[10px] text-white/50 font-medium uppercase tracking-wider">Intelligence Platform</p>
          </div>
          <button
            onClick={onClose}
            className="ml-auto lg:hidden p-1 rounded-md hover:bg-white/10 transition-colors"
          >
            <XMarkIcon className="w-5 h-5 text-white/60" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-stryker-primary/20 text-stryker-primary border border-stryker-primary/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon className={clsx('w-5 h-5 flex-shrink-0', isActive ? 'text-stryker-primary' : 'text-white/40')} />
                  <span>{item.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="ml-auto w-1.5 h-1.5 rounded-full bg-stryker-primary"
                      transition={springTransition}
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Sidebar footer */}
        <div className="px-4 py-4 border-t border-white/10">
          <div className="glass-card p-3 rounded-lg">
            <p className="text-[11px] text-white/40 font-medium">Platform Version</p>
            <p className="text-xs text-white/70 font-mono mt-0.5">v1.0.0 -- Stryker MedTech</p>
          </div>
        </div>
      </motion.aside>
    </>
  );
}

// TopBar component
function TopBar({ onMenuToggle }) {
  return (
    <header className="sticky top-0 z-30 bg-white/5 backdrop-blur-xl border-b border-white/10">
      <div className="flex items-center justify-between px-4 lg:px-6 py-3">
        {/* Left: menu toggle + breadcrumb */}
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuToggle}
            className="lg:hidden p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <Bars3Icon className="w-5 h-5 text-white/70" />
          </button>
          <div className="hidden sm:block">
            <h2 className="text-sm font-semibold text-white/90">Pricing Intelligence</h2>
            <p className="text-[11px] text-white/40">Medical Devices Portfolio Analytics</p>
          </div>
        </div>

        {/* Right: actions */}
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={springTransition}
            className="relative p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <BellIcon className="w-5 h-5 text-white/60" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-stryker-accent rounded-full" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={springTransition}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-white/10 transition-colors"
          >
            <UserCircleIcon className="w-6 h-6 text-white/60" />
            <span className="hidden sm:block text-sm text-white/70 font-medium">Admin</span>
          </motion.button>
        </div>
      </div>
    </header>
  );
}

// Main App component
export default function App() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-stryker-background">
      {/* Animated gradient background */}
      <div className="animated-gradient-bg" />

      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar onMenuToggle={() => setSidebarOpen((prev) => !prev)} />

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={springTransition}
              className="h-full"
            >
              <Suspense fallback={<PageLoader />}>
                <Routes location={location}>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/simulator" element={<PriceSimulator />} />
                  <Route path="/waterfall" element={<PriceWaterfall />} />
                  <Route path="/competitive" element={<CompetitiveLandscape />} />
                  <Route path="/external" element={<ExternalFactors />} />
                </Routes>
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
