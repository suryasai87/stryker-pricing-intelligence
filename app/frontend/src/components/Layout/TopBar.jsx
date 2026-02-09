import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  BellIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * TopBar - Top navigation bar with search, notifications, and user avatar.
 *
 * Props:
 *   title           - string, page title to display (default: 'Dashboard')
 *   onSearch        - function(query), called when search input changes
 *   notificationCount - number, badge count for the notification bell
 *   userName        - string, display name for the user avatar tooltip
 *   userAvatarUrl   - string, URL for the user avatar image (falls back to icon)
 */
export default function TopBar({
  title = 'Dashboard',
  onSearch,
  notificationCount = 0,
  userName = 'User',
  userAvatarUrl,
}) {
  const [searchFocused, setSearchFocused] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
    onSearch?.(e.target.value);
  };

  return (
    <motion.header
      className="h-16 bg-slate-900/80 backdrop-blur-xl border-b border-white/10 flex items-center justify-between px-6 z-20"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      {/* Left: Page title */}
      <motion.h1
        className="text-white font-semibold text-lg"
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ ...springTransition, delay: 0.05 }}
      >
        {title}
      </motion.h1>

      {/* Right side controls */}
      <div className="flex items-center gap-4">
        {/* Search Input */}
        <motion.div
          className={`
            relative flex items-center rounded-xl overflow-hidden transition-colors
            ${searchFocused ? 'bg-white/10 border border-[#0057B8]/50' : 'bg-white/5 border border-white/10'}
          `}
          animate={{ width: searchFocused ? 280 : 220 }}
          transition={springTransition}
        >
          <MagnifyingGlassIcon className="w-4 h-4 text-white/40 ml-3 shrink-0" />
          <input
            type="text"
            placeholder="Search products, SKUs..."
            value={searchQuery}
            onChange={handleSearchChange}
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
            className="w-full bg-transparent text-white text-sm placeholder-white/30 py-2 px-2 outline-none"
          />
        </motion.div>

        {/* Notification Bell */}
        <motion.button
          className="relative p-2 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-colors"
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          transition={springTransition}
          aria-label="Notifications"
        >
          <BellIcon className="w-5 h-5" />
          <AnimatePresence>
            {notificationCount > 0 && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                transition={springTransition}
                className="absolute -top-1 -right-1 w-4.5 h-4.5 flex items-center justify-center rounded-full text-[10px] font-bold text-white"
                style={{ backgroundColor: '#f43f5e', minWidth: 18, height: 18 }}
              >
                {notificationCount > 99 ? '99+' : notificationCount}
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>

        {/* User Avatar */}
        <motion.button
          className="flex items-center gap-2 p-1.5 rounded-xl hover:bg-white/5 transition-colors"
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          transition={springTransition}
          aria-label={`User menu for ${userName}`}
        >
          {userAvatarUrl ? (
            <img
              src={userAvatarUrl}
              alt={userName}
              className="w-8 h-8 rounded-full object-cover border border-white/20"
            />
          ) : (
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-white font-semibold text-xs border border-white/20"
              style={{ backgroundColor: '#0057B8' }}
            >
              {userName
                .split(' ')
                .map((n) => n[0])
                .join('')
                .toUpperCase()
                .slice(0, 2)}
            </div>
          )}
          <span className="text-white/70 text-sm hidden md:inline">{userName}</span>
        </motion.button>
      </div>
    </motion.header>
  );
}
