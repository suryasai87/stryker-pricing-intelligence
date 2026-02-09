import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

const springTransition = { type: 'spring', stiffness: 300, damping: 30 };

/**
 * PatentTimeline - Gantt-style horizontal timeline showing patent expiry dates.
 *
 * Props:
 *   data    - array of { id, product, category, patentStart, patentEnd, description }
 *   loading - boolean
 *   title   - string (default: 'Patent Expiry Timeline')
 */
export default function PatentTimeline({
  data = null,
  loading = false,
  title = 'Patent Expiry Timeline',
}) {
  const currentYear = new Date().getFullYear();

  // Default demo data
  const patents = data || [
    { id: 'P1', product: 'Mako SmartRobotics', category: 'Joint Replacement', patentStart: 2015, patentEnd: 2029, description: 'Robotic-arm assisted surgery platform' },
    { id: 'P2', product: 'Triathlon Knee', category: 'Joint Replacement', patentStart: 2012, patentEnd: 2027, description: 'Single-radius knee system design' },
    { id: 'P3', product: 'T2 Tibial Nail', category: 'Trauma', patentStart: 2010, patentEnd: 2026, description: 'Intramedullary fixation system' },
    { id: 'P4', product: 'Spine Implant X', category: 'Spine', patentStart: 2018, patentEnd: 2033, description: 'Expandable interbody fusion device' },
    { id: 'P5', product: 'Neptune 3', category: 'Instruments', patentStart: 2016, patentEnd: 2031, description: 'Surgical waste management system' },
    { id: 'P6', product: 'System 9 Drill', category: 'Instruments', patentStart: 2020, patentEnd: 2035, description: 'Next-gen surgical power tools' },
    { id: 'P7', product: 'ProCuity ICU', category: 'Med/Surg', patentStart: 2017, patentEnd: 2028, description: 'Smart hospital bed platform' },
    { id: 'P8', product: 'Tornier Shoulder', category: 'Joint Replacement', patentStart: 2014, patentEnd: 2026, description: 'Reverse shoulder arthroplasty' },
  ];

  // Calculate timeline bounds
  const timelineConfig = useMemo(() => {
    const minYear = Math.min(...patents.map((p) => p.patentStart), currentYear - 2);
    const maxYear = Math.max(...patents.map((p) => p.patentEnd), currentYear + 10);
    const startYear = minYear;
    const endYear = maxYear + 1;
    const years = [];
    for (let y = startYear; y <= endYear; y++) years.push(y);
    return { startYear, endYear, years, span: endYear - startYear };
  }, [patents, currentYear]);

  const getUrgencyColor = (expiryYear) => {
    const yearsLeft = expiryYear - currentYear;
    if (yearsLeft <= 2) return { bar: '#f43f5e', bg: 'rgba(244, 63, 94, 0.2)', border: 'rgba(244, 63, 94, 0.4)', label: 'Critical' };
    if (yearsLeft <= 5) return { bar: '#FFB81C', bg: 'rgba(255, 184, 28, 0.2)', border: 'rgba(255, 184, 28, 0.4)', label: 'Warning' };
    return { bar: '#10b981', bg: 'rgba(16, 185, 129, 0.2)', border: 'rgba(16, 185, 129, 0.4)', label: 'Safe' };
  };

  const getXPosition = (year) => {
    return ((year - timelineConfig.startYear) / timelineConfig.span) * 100;
  };

  const svgWidth = 800;
  const rowHeight = 48;
  const headerHeight = 40;
  const leftMargin = 180;
  const rightMargin = 20;
  const plotWidth = svgWidth - leftMargin - rightMargin;
  const svgHeight = headerHeight + patents.length * rowHeight + 20;

  const getBarX = (year) => leftMargin + ((year - timelineConfig.startYear) / timelineConfig.span) * plotWidth;

  return (
    <motion.div
      className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={springTransition}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-white font-semibold text-sm">{title}</h3>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#f43f5e' }} />
            <span className="text-white/40 text-[10px]">&lt;2 years</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#FFB81C' }} />
            <span className="text-white/40 text-[10px]">2-5 years</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#10b981' }} />
            <span className="text-white/40 text-[10px]">5+ years</span>
          </div>
        </div>
      </div>
      <p className="text-white/40 text-xs mb-6">
        Patent protection runway by product - color coded by urgency
      </p>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="animate-pulse h-10 bg-white/5 rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="w-full overflow-x-auto">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full" style={{ minWidth: 600 }}>
            {/* Year markers and grid lines */}
            {timelineConfig.years
              .filter((y) => y % 2 === 0)
              .map((year) => {
                const x = getBarX(year);
                return (
                  <g key={year}>
                    <line
                      x1={x}
                      y1={headerHeight}
                      x2={x}
                      y2={svgHeight - 10}
                      stroke="rgba(255,255,255,0.05)"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={x}
                      y={headerHeight - 10}
                      textAnchor="middle"
                      fill="rgba(255,255,255,0.35)"
                      fontSize="10"
                      fontFamily="monospace"
                    >
                      {year}
                    </text>
                  </g>
                );
              })}

            {/* Current year marker */}
            <line
              x1={getBarX(currentYear)}
              y1={headerHeight}
              x2={getBarX(currentYear)}
              y2={svgHeight - 10}
              stroke="rgba(0, 87, 184, 0.5)"
              strokeWidth="2"
              strokeDasharray="6 3"
            />
            <text
              x={getBarX(currentYear)}
              y={headerHeight - 10}
              textAnchor="middle"
              fill="#0057B8"
              fontSize="10"
              fontWeight="bold"
              fontFamily="monospace"
            >
              {currentYear}
            </text>

            {/* Patent bars */}
            {patents.map((patent, index) => {
              const urgency = getUrgencyColor(patent.patentEnd);
              const barStartX = getBarX(patent.patentStart);
              const barEndX = getBarX(patent.patentEnd);
              const barY = headerHeight + index * rowHeight + 8;
              const barHeight = 28;
              const yearsLeft = patent.patentEnd - currentYear;

              return (
                <g key={patent.id}>
                  {/* Row background on hover */}
                  <rect
                    x={0}
                    y={headerHeight + index * rowHeight}
                    width={svgWidth}
                    height={rowHeight}
                    fill="transparent"
                    className="hover:fill-[rgba(255,255,255,0.02)]"
                  />

                  {/* Row separator */}
                  <line
                    x1={0}
                    y1={headerHeight + index * rowHeight}
                    x2={svgWidth}
                    y2={headerHeight + index * rowHeight}
                    stroke="rgba(255,255,255,0.03)"
                  />

                  {/* Product label */}
                  <text
                    x={10}
                    y={barY + barHeight / 2 - 4}
                    fill="rgba(255,255,255,0.8)"
                    fontSize="11"
                    fontWeight="500"
                  >
                    {patent.product}
                  </text>
                  <text
                    x={10}
                    y={barY + barHeight / 2 + 10}
                    fill="rgba(255,255,255,0.35)"
                    fontSize="9"
                  >
                    {patent.category}
                  </text>

                  {/* Patent bar */}
                  <motion.rect
                    x={barStartX}
                    y={barY}
                    rx={6}
                    ry={6}
                    height={barHeight}
                    fill={urgency.bg}
                    stroke={urgency.border}
                    strokeWidth={1}
                    initial={{ width: 0 }}
                    animate={{ width: barEndX - barStartX }}
                    transition={{ ...springTransition, delay: index * 0.08 }}
                  />

                  {/* Filled portion (elapsed) */}
                  <motion.rect
                    x={barStartX}
                    y={barY}
                    rx={6}
                    ry={6}
                    height={barHeight}
                    fill={urgency.bar}
                    fillOpacity={0.4}
                    initial={{ width: 0 }}
                    animate={{
                      width: Math.max(0, Math.min(getBarX(currentYear) - barStartX, barEndX - barStartX)),
                    }}
                    transition={{ ...springTransition, delay: index * 0.08 + 0.1 }}
                  />

                  {/* Years left label */}
                  <motion.text
                    x={barEndX + 8}
                    y={barY + barHeight / 2 + 4}
                    fill={urgency.bar}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight="600"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.08 + 0.3 }}
                  >
                    {yearsLeft > 0 ? `${yearsLeft}y` : 'EXP'}
                  </motion.text>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* Summary */}
      <motion.div
        className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-white/10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ ...springTransition, delay: 0.5 }}
      >
        <div className="text-center">
          <p className="text-[#f43f5e] font-mono text-lg font-bold">
            {patents.filter((p) => p.patentEnd - currentYear <= 2).length}
          </p>
          <p className="text-white/40 text-xs">Expiring &lt;2yr</p>
        </div>
        <div className="text-center">
          <p className="text-[#FFB81C] font-mono text-lg font-bold">
            {patents.filter((p) => {
              const y = p.patentEnd - currentYear;
              return y > 2 && y <= 5;
            }).length}
          </p>
          <p className="text-white/40 text-xs">Expiring 2-5yr</p>
        </div>
        <div className="text-center">
          <p className="text-[#10b981] font-mono text-lg font-bold">
            {patents.filter((p) => p.patentEnd - currentYear > 5).length}
          </p>
          <p className="text-white/40 text-xs">Safe 5yr+</p>
        </div>
      </motion.div>
    </motion.div>
  );
}
