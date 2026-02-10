# Stryker Pricing Intelligence Platform v2

> An enterprise-grade pricing analytics and simulation platform for medical device portfolios, built on the Databricks Lakehouse. Transforms 600K+ transaction records through a medallion architecture, trains ML models for elasticity and anomaly detection, and serves 12 interactive analytics views through a React + FastAPI application deployed as a Databricks App.

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Architecture](#2-architecture)
3. [Tab-by-Tab Presentation Talk Track](#3-tab-by-tab-presentation-talk-track)
4. [Implementation Guide](#4-implementation-guide)
5. [CREATE DDL Statements](#5-create-ddl-statements)
6. [Notebook Code Reference](#6-notebook-code-reference)
7. [API Reference](#7-api-reference)
8. [DABs Deployment Guide](#8-dabs-deployment-guide)
9. [Customization Guide](#9-customization-guide)
10. [Technology Stack](#10-technology-stack)

---

## 1. Executive Overview

### What Is It?

The Stryker Pricing Intelligence Platform is a full-stack analytics application that enables medical device organizations to **detect discount leakage, quantify price elasticity, simulate portfolio-wide uplift scenarios, and receive AI-driven pricing recommendations** -- all backed by real-time data from a Databricks Lakehouse.

The platform processes 600,000 synthetic FICM (Full Invoice-to-Cash Margin) transactions across 200 SKUs, 500 customers, 75 sales representatives, 7 product families, and 4 global regions over a 36-month window.

### Who Uses It?

| Persona | Primary Use Case |
|---------|-----------------|
| **CFO / SVP Finance** | Portfolio-level margin health, recovery opportunity quantification ($12M+), uplift scenario planning |
| **VP Pricing / Chief Pricing Officer** | Elasticity-driven price increase strategy, top 100 action list for next pricing review |
| **Commercial Operations** | Discount governance, sales rep performance benchmarking, competitive positioning intelligence |
| **Pricing Analysts** | What-if simulations, custom scenario modeling, external factor correlation analysis |
| **Sales Leadership** | Rep-level discount outlier identification, contract tier optimization, territory-level margin analysis |
| **Supply Chain / Procurement** | Tariff and commodity impact tracking, freight cost analysis, FX exposure monitoring |

### Key Value Propositions

| Value Driver | Description | Example Insight |
|-------------|-------------|-----------------|
| **Discount Leakage Detection** | Z-score-based outlier detection identifies reps discounting 2-3 standard deviations above peer norms | "7 reps are responsible for $12.3M in recoverable discount leakage across GPO and IDN segments" |
| **Price Elasticity Intelligence** | Log-log regression on 36 months of price-volume data classifies every SKU-segment pair | "65% of the Joint Replacement portfolio is highly inelastic -- safe for 2-3% price increases with <0.3% volume loss" |
| **+1% Uplift Planning** | Optimization engine that finds the minimum set of pricing actions to achieve a target uplift | "Achieve +1% portfolio revenue uplift with only 87 targeted actions and just 0.3% volume impact" |
| **AI Recommendations** | Multi-signal recommendation engine combining elasticity, outliers, competitive positioning, and margin analysis | "Top recommendation: Standardize discounting on Triathlon II Hip System -- $2.1M annual margin recovery at low competitive risk" |

---

## 2. Architecture

### System Architecture

```
+====================================================================================+
|                           DATABRICKS LAKEHOUSE PLATFORM                            |
|                                                                                    |
|  +-------------------+    +------------------------+    +-----------------------+  |
|  |  12a: Dimensions  |    |  12: FICM Fact Table   |    |  19: External Market  |  |
|  |  - dim_customers  |--->|  600K transactions     |    |  Tariffs, FX, Comm.   |  |
|  |  - dim_sales_reps |    |  Silver layer          |    |  Gold layer           |  |
|  |  - dim_products   |    +----------+-------------+    +-----------+-----------+  |
|  +-------------------+               |                              |              |
|                                      v                              |              |
|  +-------------------------------+   +---------------------------+  |              |
|  |      GOLD ANALYTICS TABLES    |   |   ML MODEL TRAINING       |  |              |
|  | 13: discount_outliers         |   | 20: Isolation Forest      |  |              |
|  | 14: price_elasticity          |   | 21: Gradient Boosting     |  |              |
|  | 15: uplift_simulation         |   |   (Point + Quantile)      |  |              |
|  | 16: pricing_recommendations   |   +----------+----------------+  |              |
|  | 17: top100_price_changes      |              |                   |              |
|  | 18: custom_pricing_scenarios  |              v                   |              |
|  +-------------------------------+   +---------------------------+  |              |
|                 |                    | MLflow Model Registry     |  |              |
|                 |                    | Unity Catalog Models      |  |              |
|                 |                    +----------+----------------+  |              |
|                 |                               |                   |              |
|  +--------------v-------------------------------v-------------------v-----------+  |
|  |                        DATABRICKS SQL WAREHOUSE                              |  |
|  |                     (Serverless, ID: 4b28691c780d9875)                       |  |
|  +-------------------------------------+----------------------------------------+  |
+====================================================================================+
                                         |
                                    SQL Queries
                                         |
                    +--------------------v---------------------+
                    |           FastAPI BACKEND (v1 + v2)       |
                    |                                           |
                    |  /api/v1/* -- Core Analytics              |
                    |    Products, Waterfall, Competitive,      |
                    |    Simulation, KPIs, External Factors     |
                    |                                           |
                    |  /api/v2/* -- Advanced Analytics          |
                    |    FICM, Outliers, Elasticity, Uplift,    |
                    |    Top100, Recommendations, Scenarios,    |
                    |    External Data                          |
                    |                                           |
                    |  Databricks SDK (auto-auth via SP)        |
                    |  5-minute cache layer for performance     |
                    +--------------------+---------------------+
                                         |
                                    REST JSON
                                         |
                    +--------------------v---------------------+
                    |         React 18 FRONTEND (SPA)          |
                    |                                           |
                    |  12 Interactive Views:                    |
                    |    Dashboard | Simulator | Waterfall      |
                    |    Competitive | External Factors         |
                    |    ----------------------------------------|
                    |    Discount Outliers | Price Elasticity    |
                    |    Uplift Simulator | Top 100 Changes     |
                    |    AI Recommendations | External Data     |
                    |    Pricing Scenarios                      |
                    |                                           |
                    |  Recharts + D3 | Framer Motion | Tailwind |
                    +------------------------------------------+
```

### Medallion Architecture Data Flow

```
  BRONZE (Raw)              SILVER (Enriched)              GOLD (Analytics-Ready)
+----------------+     +------------------------+     +----------------------------+
| Raw synthetic  |     | ficm_pricing_master    |     | discount_outliers          |
| data from      | --> | (600K rows, 47 cols)   | --> | price_elasticity           |
| notebooks      |     |                        |     | uplift_simulation          |
| 01-04          |     | dim_customers (500)    |     | pricing_recommendations    |
|                |     | dim_sales_reps (75)    |     | top100_price_changes       |
|                |     | dim_products (200)     |     | custom_pricing_scenarios   |
+----------------+     +------------------------+     | external_market_data       |
                                                      +----------------------------+
```

### Service Architecture

```
+---------------------------+      +---------------------------+
|  Databricks App Runtime   |      |  Unity Catalog            |
|  (Service Principal Auth) |      |                           |
|                           |      |  hls_amer_catalog         |
|  uvicorn                  |      |    +-- bronze.*           |
|    +-- FastAPI (8000)     |----->|    +-- silver.*           |
|        +-- /api/v1/*      |      |    +-- gold.*            |
|        +-- /api/v2/*      |      |    +-- models.*          |
|        +-- static/ (SPA)  |      |                           |
+---------------------------+      +---------------------------+
          |
          v
+---------------------------+
|  SQL Warehouse            |
|  (Serverless)             |
|  4b28691c780d9875         |
+---------------------------+
```

---

## 3. Tab-by-Tab Presentation Talk Track

### Tab 1: Dashboard -- Portfolio Health at a Glance

**Business Value**: The Dashboard is the executive landing page that answers the question every CFO asks first: "How is the portfolio performing?" It provides the 30-second health check before diving into root causes.

**What You See**:
- **4 KPI Cards**: Total Revenue, Average Margin %, YoY Growth %, Total Active Products
- **Revenue by Segment Bar Chart**: Breakdown across IDN, GPO, Direct, Distributor, Govt-VA, Academic
- **Margin by Segment Chart**: Identifies which segments are margin-dilutive
- **5-minute auto-refresh**: Data stays current without manual reload

**Key Insights**:
- Identify which segments drive the most revenue vs. which deliver the best margins
- Spot margin compression trends before they become quarterly earnings surprises
- Benchmark portfolio performance across Stryker's three business units (MedSurg, Ortho, Neuro)

**Demo Script**: "This is the portfolio command center. At a glance, you can see $2.8B in total revenue across 200 active SKUs with an average margin of 42%. Notice how GPO and IDN segments drive 50% of revenue but only deliver 38% margins -- that gap is exactly what the advanced analytics tabs will help us close."

---

### Tab 2: Price Simulator -- What-If ML-Driven Predictions

**Business Value**: Before changing any price, commercial teams need to understand the ripple effects. The Price Simulator uses three trained ML models to predict how a proposed price change will impact volume, revenue, and margin -- with confidence intervals and competitive risk scoring.

**What You See**:
- **Product Selector**: Dropdown to choose any of 200 SKUs
- **Price Change Slider**: Adjust from -20% to +20%
- **Prediction Panel**: Volume change %, revenue impact ($), margin impact ($)
- **Confidence Interval Bar**: 95% confidence bounds on the volume prediction
- **SHAP Factor Breakdown**: Top sensitivity factors driving the prediction (competitor_price_ratio, contract_tier, etc.)
- **Competitive Risk Score**: 0-100 gauge showing likelihood of competitive displacement

**Key Insights**:
- Understand the non-linear relationship between price changes and volume response
- Identify which factors (competitor proximity, contract tier, segment) amplify or dampen price sensitivity
- Quantify the risk of competitive switching before committing to a price action

**Demo Script**: "Let me show you what happens when we raise the price of the Triathlon II Knee System by 3%. The ML model predicts a volume decline of only 1.2% -- that is well within the inelastic range. The confidence interval is tight at -0.5% to -1.9%, and the competitive risk score is just 0.35. This is a safe increase that would generate an additional $1.8M in annual margin."

---

### Tab 3: Price Waterfall -- List-to-Pocket Discount Breakdown

**Business Value**: The Price Waterfall is the pricing team's X-ray machine. It exposes every layer of discount between list price and pocket price, revealing where margin erodes most -- and which layers are controllable.

**What You See**:
- **Product Selector**: Choose any SKU to analyze
- **Waterfall Chart**: Visual cascade from List Price down through Contract Discount, Invoice Adjustment, Off-Invoice Leakage, Rebates, Freight, Other Deductions to Pocket Price
- **Step Values**: Dollar amount and percentage at each waterfall step
- **Cumulative Line**: Running total showing the compounding effect of each discount layer

**Key Insights**:
- Distinguish between on-invoice discounts (controllable) and off-invoice leakage (often hidden)
- Identify products where freight costs are disproportionately eroding pocket margin
- Compare waterfall shapes across product families to establish best-practice benchmarks

**Demo Script**: "Here is the full price waterfall for our Mako Robotic Hip System. Starting at a $45,000 list price, we lose 22% at the contract level, another 3% at invoice, and then watch: 6% disappears in off-invoice leakage -- rebates, chargebacks, and SPAs that most teams do not even track. That 6% is $2,700 per unit, and at 500 units annually, that is $1.35M in hidden margin erosion on just this one SKU."

---

### Tab 4: Competitive Landscape -- Market Positioning Intelligence

**Business Value**: Pricing does not happen in a vacuum. The Competitive Landscape tab maps Stryker's position against Zimmer Biomet, DePuy Synthes, Smith+Nephew, and other competitors across every product category, enabling data-driven competitive pricing strategy.

**What You See**:
- **Category Selector**: Filter by product family (Endoscopy, Joint Replacement, Trauma, etc.)
- **Competitor ASP Comparison**: Average selling price benchmarks across competitors
- **Market Share Visualization**: Pie/bar chart showing share by competitor
- **Innovation Score Benchmarks**: Relative innovation positioning
- **ASP Trend Indicators**: Year-over-year price trajectory per competitor

**Key Insights**:
- Identify categories where Stryker is priced above or below the competitive average
- Spot competitors who are aggressively cutting prices (declining ASP trend)
- Correlate innovation scores with pricing power to justify premium positioning

**Demo Script**: "In Joint Replacement, Stryker holds a 28% market share with an average ASP 8% above the category average. That premium is justified by our innovation score of 0.87 -- highest in the category. But look at Endoscopy: Karl Storz is only 2% below us with a 0.82 innovation score. That is where we need to be strategic about any price increases."

---

### Tab 5: External Factors -- Macro-Economic Impact Awareness

**Business Value**: Healthcare pricing is subject to external forces that no internal model can ignore. This tab surfaces the macro-economic indicators that every pricing committee should review before making portfolio-wide decisions.

**What You See**:
- **Medical CPI Trend**: Consumer Price Index for medical devices over 36 months
- **Tariff Rate (Steel)**: Current import tariff rates affecting raw material costs
- **Supply Chain Pressure Index**: Measure of supply chain disruption severity
- **Hospital CapEx Index**: Hospital capital expenditure trends (demand proxy)
- **Reimbursement Trend**: CMS reimbursement rate trajectory
- **Raw Material Index**: Commodity cost trends (titanium, cobalt-chrome, polymers)

**Key Insights**:
- Correlate tariff increases with COGS pressure to justify cost-recovery price actions
- Track hospital CapEx cycles to time price increases during high-demand periods
- Monitor reimbursement trends that may constrain customer willingness to absorb increases

**Demo Script**: "The Medical CPI has risen 4.2% year-over-year, while hospital CapEx is up 6.8%. That combination -- rising costs and strong demand -- creates an ideal window for targeted price increases. Meanwhile, steel tariffs at 25% are adding $15-30 per unit in Trauma & Extremities, which we can pass through with a data-backed justification."

---

### Tab 6: Discount Outliers -- "Which reps are giving away margin?"

**Business Value**: This is the single highest-ROI tab in the platform. Discount outlier detection identifies sales representatives whose discounting behavior deviates significantly from peer norms (same SKU, same customer segment). The recovery opportunity is typically $5M-$20M annualized -- margin that can be recaptured through coaching, guardrails, and contract standardization with ZERO price increase required.

**What You See**:
- **Summary KPI Cards**: Total outliers flagged, total recovery opportunity ($12.3M), breakdown by severity (Severe/Moderate/Watch)
- **Outlier Table**: Filterable by business unit, customer segment, country, severity level
- **Z-Score Distribution**: How far each outlier deviates from peer norms (1.5x to 5x+ standard deviations)
- **By-Rep Drilldown**: Click any sales rep to see all their outlier transactions
- **Recovery by Business Unit**: Bar chart showing recovery potential per BU (MedSurg, Ortho, Neuro)
- **Recovery by Country**: Geographic breakdown of where margin leakage concentrates

**Key Insights**:
- 7 of 75 sales reps (~10%) are systematically over-discounting by 35-50% -- far above the 15-30% peer norm
- GPO and IDN segments show the widest discount variance, suggesting inconsistent contract application
- $12.3M in annualized recovery potential requires no price changes -- just bringing outlier reps to peer-group average
- Severe outliers (z-score > 3.0) represent 60% of the total recovery opportunity

**Demo Script**: "This is the money tab. We found 7 reps who are systematically giving away 35-50% discounts -- that is 15-20 points above their peer group average for the same SKUs and same customer segments. The total recovery opportunity is $12.3M annually. And here is the key insight: we do not need to raise a single price. We just need to coach these reps back to the peer norm. Three of them are in the US Southeast territory, and they account for 40% of the total leakage."

---

### Tab 7: Price Elasticity -- "Which products can absorb a price increase?"

**Business Value**: Not all price increases are created equal. The Price Elasticity tab uses log-log regression on 36 months of actual price-volume data to classify every SKU-segment combination by price sensitivity. This answers the critical question: "Where can I raise prices with minimal volume loss?"

**What You See**:
- **Elasticity Classification Grid**: SKUs classified as Highly Inelastic, Inelastic, Unit Elastic, or Elastic
- **Distribution Histogram**: Bell curve showing the spread of elasticity coefficients across the portfolio
- **Safe Increase Ranges**: For each SKU-segment, the maximum price increase at 1%, 3%, and 5% volume loss thresholds
- **Filter by Business Unit / Product Family / Customer Segment**: Drill into specific portfolio slices
- **Confidence Level Indicator**: High/Medium/Low based on R-squared and sample size

**Key Insights**:
- 30% of the portfolio is Highly Inelastic (|beta| < 0.5) -- these products can absorb 3-5% increases with <1% volume impact
- Joint Replacement is the most inelastic family (surgeon preference drives demand, not price)
- Instruments is the most elastic family (commodity-like, many substitutes)
- 65% of the portfolio can safely absorb at least a 2% price increase

**Demo Script**: "Let me show you the elasticity heat map. The green zone -- Highly Inelastic -- represents 30% of our portfolio. These are products like the Triathlon Knee System where surgeon preference, not price, drives purchasing decisions. For these 60 SKUs, we can raise prices by 3-5% with less than 1% volume loss. That translates directly to margin improvement. Contrast that with the red zone -- our Instruments portfolio -- where a 2% increase would cost us 3-4% in volume. The intelligence here is knowing exactly where to push and where to hold."

---

### Tab 8: Uplift Simulator -- "How do we achieve +1% portfolio uplift?"

**Business Value**: This is the money shot. The Uplift Simulator answers the most important question in any pricing review: "What is the minimum number of actions needed to achieve our target uplift, and what is the volume risk?" It combines elasticity data, discount outlier intelligence, and competitive positioning into a ranked action list optimized for maximum revenue impact with minimum disruption.

**What You See**:
- **Target Uplift Selector**: Set the target (default 1.0%, adjustable to 0.5%, 1.5%, 2.0%)
- **Summary KPI Cards**: Actions needed (87), total revenue impact (+$28M), total volume impact (-0.3%), average increase per SKU (1.8%)
- **Ranked Action Table**: Every recommended action sorted by composite uplift score
- **Exclusion Controls**: Exclude specific SKUs, segments, or countries from the simulation
- **Max-per-SKU Cap**: Limit the maximum increase per SKU (default 5%)
- **Cumulative Uplift Curve**: Line chart showing how uplift accumulates as actions are applied

**Key Insights**:
- Achieving +1% portfolio uplift requires only 87 targeted actions (out of 2,000+ possible combinations)
- The estimated volume impact is just 0.3% -- a 3:1 revenue-to-volume ratio
- The first 20 actions deliver 60% of the uplift (power-law distribution)
- Excluding the most elastic 10% of the portfolio has minimal impact on achievable uplift

**Demo Script**: "This is the slide your CFO will want to present to the board. To achieve a 1% portfolio revenue uplift -- roughly $28M -- we need exactly 87 targeted pricing actions. The total volume impact? Just 0.3%. That is a 3:1 ratio of revenue gain to volume risk. And look at the cumulative curve: the first 20 actions deliver 60% of the uplift. We can start with those 20 next quarter and add the remaining 67 over the following two quarters. Every action has been validated against elasticity data and competitive positioning."

---

### Tab 9: Top 100 Changes -- "What are my top 100 pricing actions for tomorrow morning?"

**Business Value**: The VP of Pricing's first stop every Monday morning. This tab curates the 100 highest-priority pricing actions from the full recommendation engine, enriched with all dimensions needed for immediate execution: product, customer, rep, territory, risk level, and expected impact.

**What You See**:
- **Fully Filterable Table**: Sort and filter by country, product family, customer segment, sales rep, business unit, risk level
- **Revenue Impact Column**: Dollar impact of each recommended action
- **Risk Level Indicator**: Low/Medium/High risk badge per action
- **Quick Action Summary**: One-sentence description of what to do (e.g., "Standardize discount on SKU-0042 for GPO segment to peer norm -- $340K recovery")
- **Pagination**: 25 rows per page with full pagination controls
- **CSV Export**: One-click download for offline analysis or integration with pricing systems
- **Multi-Select Filters**: Filter by multiple countries, segments, or families simultaneously

**Key Insights**:
- The top 10 actions alone represent $8.2M in annual impact
- 45% of the top 100 are discount standardization actions (no price increase needed)
- 35% are targeted price increases on inelastic SKUs
- 20% are margin recovery actions on products with excessive off-invoice leakage

**Demo Script**: "This is the executive action list. These 100 actions are ranked by a composite priority score that balances revenue impact, execution risk, and competitive exposure. Let me filter to just the US market and Low risk level -- now I have 38 actions worth $4.8M that we can implement this quarter with virtually no competitive risk. And I can export this to CSV right now and hand it to the pricing operations team."

---

### Tab 10: AI Recommendations -- "What does the AI suggest across all dimensions?"

**Business Value**: The AI Recommendations tab synthesizes signals from every other analytics module -- elasticity, outlier detection, competitive positioning, uplift scoring, and margin analysis -- into unified, prioritized recommendations. Each recommendation has an action type, priority score (0-100), risk level, and estimated revenue impact.

**What You See**:
- **Summary by Action Type**: Counts and total impact for each type (INCREASE_PRICE, STANDARDIZE_DISCOUNT, REVIEW_REP, COMPETITIVE_ADJUSTMENT, MARGIN_RECOVERY)
- **Filterable Recommendation List**: Filter by action type, risk level, business unit, product family
- **Priority Score Ranking**: 0-100 composite score combining all signal dimensions
- **Sortable Columns**: Sort by priority, revenue impact, risk level, or any dimension
- **Action Type Distribution**: Pie chart showing the balance of recommendation types

**Key Insights**:
- STANDARDIZE_DISCOUNT actions represent the highest total impact with the lowest risk
- INCREASE_PRICE actions are concentrated in Joint Replacement and Spine (inelastic families)
- REVIEW_REP actions flag the 7 systematic over-discounters for management intervention
- COMPETITIVE_ADJUSTMENT actions are rare (8% of total) but high-impact in Endoscopy
- Average priority score for MARGIN_RECOVERY actions is 72/100 -- actionable and well-supported

**Demo Script**: "The AI engine has generated 200+ recommendations across five action types. Let me filter to just STANDARDIZE_DISCOUNT and sort by revenue impact. These 85 actions represent $14.2M in recoverable margin, and every one of them is backed by peer-group statistical evidence. The beauty of this approach is that we are not asking customers to pay more -- we are asking our own team to discount less."

---

### Tab 11: External Data -- "How do tariffs, commodities, FX affect my portfolio?"

**Business Value**: Pricing decisions must account for external market forces. This tab enables users to upload their own external data (tariff schedules, commodity prices, FX rates, competitor intelligence) and correlate it with internal pricing data. It bridges the gap between internal analytics and the real-world market environment.

**What You See**:
- **Data Upload Panel**: Drag-and-drop file upload for Excel (.xlsx/.xls) and CSV files
- **Category Selector**: Classify uploads as tariff, commodity, fx, or competitor data
- **Current External Data Table**: All previously uploaded data with source, category, effective date, and values
- **Source History**: List of all uploaded files with metadata (filename, row count, timestamp)
- **Data Preview**: First 10 rows of any uploaded file displayed immediately after upload
- **Volume Path**: Confirmation of Unity Catalog Volume storage location

**Key Insights**:
- Track tariff rate changes (e.g., Section 301 tariffs on medical device components) and their COGS impact
- Monitor titanium and cobalt-chrome commodity prices that directly affect implant manufacturing costs
- Correlate EUR/USD exchange rate movements with EMEA pricing strategy
- Ingest competitor price list updates and benchmark against internal ASPs

**Demo Script**: "I just uploaded the latest Section 301 tariff schedule. The platform ingested 45 line items, stored the raw file in our Unity Catalog Volume for auditability, and now we can cross-reference these tariff rates against our COGS to quantify the exact cost impact per product family. This is how we build the business case for cost-recovery price increases that customers can understand."

---

### Tab 12: Pricing Scenarios -- "Let me model my own what-if scenario and submit for review"

**Business Value**: The Pricing Scenarios tab puts the power of scenario planning in the hands of every pricing analyst and commercial team member. Users can create, save, share, and submit custom pricing scenarios through a lightweight approval workflow (Draft -> Submitted -> Reviewed -> Approved), enabling collaborative pricing governance.

**What You See**:
- **Create Scenario Form**: Name, description, parameters (JSON), target uplift, selected SKUs and segments
- **My Scenarios List**: Paginated table of all scenarios owned by the current user
- **Status Workflow**: Visual status badges (Draft, Submitted, Reviewed, Approved)
- **Admin View**: Admin users can see all scenarios across all users
- **Search and Filter**: Search by name, filter by status
- **User Identity**: Current user info from Databricks OBO (On-Behalf-Of) authentication

**Key Insights**:
- Enables distributed scenario planning: regional pricing managers can model their own territory-specific scenarios
- Approval workflow prevents unauthorized price changes from going to market
- Full audit trail: every scenario records who created it, when, and what parameters were used
- Scenarios can be revisited and compared: "What did we model last quarter vs. what actually happened?"

**Demo Script**: "Let me create a new scenario. I will call it 'Q3 2025 Trauma Portfolio Uplift', set a 1.5% target, and exclude our government/VA products since those have fixed pricing. The system saves this as a draft linked to my identity. When I am ready, I submit it for review, and the VP Pricing can approve it directly from this same interface. Every scenario has a full audit trail -- we know who created what, when, and what assumptions they used."

---

## 4. Implementation Guide

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Databricks Workspace | Unity Catalog enabled | Data storage and compute |
| SQL Warehouse | Serverless or Pro | Backend query execution |
| ML Runtime Cluster | 13.3 LTS+ | Notebook execution and ML training |
| Databricks CLI | v0.200+ | App deployment and administration |
| Node.js | 18+ | Frontend build |
| npm | 9+ | Frontend dependency management |
| Python | 3.10+ | Backend and notebook development |

### Step 1: Clone and Branch

```bash
# Clone the repository
git clone <repository-url>
cd stryker-pricing-intelligence

# Create a feature branch (if modifying)
git checkout -b feature/your-feature-name
```

### Step 2: Run Notebooks (in order)

Execute each notebook on a Databricks cluster with ML Runtime 13.3 LTS+. The notebooks must be run in the following order due to data dependencies:

```
Execution Order:
================

12a  -->  12  -->  13  -->  14  -->  15  -->  16  -->  17  -->  18  -->  19  -->  20  -->  21  -->  99
 |         |       |        |        |        |        |        |        |        |        |        |
 v         v       v        v        v        v        v        v        v        v        v        v
Dims    FICM    Outliers  Elast.  Uplift   Recs    Top100  Scenarios  ExtData  Anomaly  Adv.ML  Validate
```

| Step | Notebook | Purpose | Input | Output | Est. Runtime |
|------|----------|---------|-------|--------|-------------|
| 1 | `12a_ficm_dimensions.py` | Generate dimension tables (customers, reps, products) | None | `silver.dim_customers` (500), `silver.dim_sales_reps` (75), `silver.dim_products` (200) | ~2 min |
| 2 | `12_ficm_pricing_master.py` | Generate 600K FICM fact table with full pricing waterfall | Dimensions from 12a | `silver.ficm_pricing_master` (600K) | ~10 min |
| 3 | `13_gold_discount_outliers.py` | Detect discount outliers using Z-score analysis | `silver.ficm_pricing_master` | `gold.discount_outliers` (~150-300 rows) | ~5 min |
| 4 | `14_gold_price_elasticity.py` | Compute price elasticity via log-log regression | `silver.ficm_pricing_master` | `gold.price_elasticity` (~800-1400 rows) | ~8 min |
| 5 | `15_gold_uplift_simulation.py` | Score SKU-customer-rep combinations for uplift potential | `silver.ficm_pricing_master`, `gold.price_elasticity`, `gold.discount_outliers` | `gold.uplift_simulation` (~2000-5000 rows) | ~5 min |
| 6 | `16_gold_pricing_recommendations.py` | Generate multi-signal pricing recommendations | `gold.price_elasticity`, `gold.discount_outliers`, `gold.uplift_simulation` | `gold.pricing_recommendations` (200+ rows) | ~3 min |
| 7 | `17_gold_top100_recommended_changes.py` | Curate top 100 highest-priority actions | `gold.pricing_recommendations` + upstream gold tables | `gold.top100_price_changes` (100 rows) | ~2 min |
| 8 | `18_create_custom_pricing_scenarios_table.py` | Create scenarios table with 18 synthetic scenarios | None | `gold.custom_pricing_scenarios` (18 rows) | ~1 min |
| 9 | `19_gold_external_data_integration.py` | Ingest external market data (tariffs, FX, commodities) | Volume uploads or synthetic fallback | `gold.external_market_data` (30+ rows) | ~2 min |
| 10 | `20_train_discount_anomaly_model.py` | Train Isolation Forest for discount anomaly detection | `silver.ficm_pricing_master` | MLflow model: `models.discount_anomaly_detector` | ~5 min |
| 11 | `21_train_advanced_elasticity_model.py` | Train GBR with quantile bounds for elasticity | `silver.ficm_pricing_master` | MLflow model: `models.advanced_elasticity_model` | ~8 min |
| 12 | `99_validate_end_to_end.py` | Validate all tables exist with correct schema and row counts | All tables | Validation report (pass/fail) | ~2 min |

### Step 3: Grant Service Principal Permissions

After deploying the app (Step 4), retrieve the Service Principal Client ID and grant permissions:

```bash
# Get the Service Principal Client ID
databricks apps get stryker-pricing-intel --output json --profile DEFAULT | jq '.service_principal_client_id'
# Returns: 42ccfce7-b085-401f-baf3-264dcbd01230
```

#### SQL Warehouse Access

```bash
databricks permissions update sql/warehouses 4b28691c780d9875 --json '{
  "access_control_list": [{
    "service_principal_name": "42ccfce7-b085-401f-baf3-264dcbd01230",
    "permission_level": "CAN_USE"
  }]
}' --profile DEFAULT
```

#### Unity Catalog Grants

Execute these SQL statements in a Databricks SQL editor or notebook:

```sql
-- Catalog access
GRANT USE CATALOG ON CATALOG hls_amer_catalog TO `42ccfce7-b085-401f-baf3-264dcbd01230`;

-- Schema access
GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.silver TO `42ccfce7-b085-401f-baf3-264dcbd01230`;
GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.gold TO `42ccfce7-b085-401f-baf3-264dcbd01230`;

-- Table-level SELECT
GRANT SELECT ON SCHEMA hls_amer_catalog.silver TO `42ccfce7-b085-401f-baf3-264dcbd01230`;
GRANT SELECT ON SCHEMA hls_amer_catalog.gold TO `42ccfce7-b085-401f-baf3-264dcbd01230`;

-- Write access for scenarios table (user-created scenarios)
GRANT MODIFY ON TABLE hls_amer_catalog.gold.custom_pricing_scenarios TO `42ccfce7-b085-401f-baf3-264dcbd01230`;
```

#### Model Serving Endpoint (if using ML inference)

```bash
databricks permissions update serving-endpoints/stryker-pricing-models --json '{
  "access_control_list": [{
    "service_principal_name": "apps/stryker-pricing-intel",
    "permission_level": "CAN_QUERY"
  }]
}' --profile DEFAULT
```

### Step 4: Build and Deploy

```bash
# Navigate to the app directory
cd app

# Build the frontend (installs deps, builds React, copies to static/)
python build.py

# Deploy to Databricks Apps
python deploy_to_databricks.py --app-name stryker-pricing-intel

# For a hard redeploy (deletes and recreates the app)
python deploy_to_databricks.py --app-name stryker-pricing-intel --hard-redeploy
```

### Step 5: Verify

1. **Check app status**:
   ```bash
   databricks apps get stryker-pricing-intel --profile DEFAULT
   ```

2. **Access the app URL**:
   ```
   https://stryker-pricing-intel-1602460480284688.aws.databricksapps.com
   ```

3. **Verify health endpoint**:
   ```bash
   curl https://stryker-pricing-intel-1602460480284688.aws.databricksapps.com/health
   # Expected: {"status":"healthy","version":"1.0.0"}
   ```

4. **Verify API connectivity**:
   ```bash
   curl https://stryker-pricing-intel-1602460480284688.aws.databricksapps.com/api/v1/products?limit=5
   ```

---

## 5. CREATE DDL Statements

### Silver Layer Tables

#### `hls_amer_catalog.silver.ficm_pricing_master`

The core fact table containing 600,000 FICM transactions with the complete pricing waterfall.

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.silver.ficm_pricing_master (
    -- Identifiers
    transaction_id          STRING      NOT NULL    COMMENT 'Unique transaction identifier (FICM-0000000001)',
    sku                     STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_id              STRING      NOT NULL    COMMENT 'Product dimension FK',
    product_name            STRING      NOT NULL    COMMENT 'Stryker-branded product name',
    product_family          STRING      NOT NULL    COMMENT 'Product family: Endoscopy | Joint Replacement | Trauma & Extremities | Spine | Instruments | Neurovascular | Sports Medicine',
    product_category        STRING      NOT NULL    COMMENT 'Sub-category within product family',
    business_unit           STRING      NOT NULL    COMMENT 'Business unit: MedSurg | Ortho | Neuro',

    -- Customer Dimension
    customer_id             STRING      NOT NULL    COMMENT 'Customer dimension FK',
    customer_name           STRING      NOT NULL    COMMENT 'Hospital/facility name',
    customer_segment        STRING      NOT NULL    COMMENT 'IDN | GPO | Direct | Distributor | Govt-VA | Academic',
    customer_tier           STRING      NOT NULL    COMMENT 'Customer tier: A | B | C | D',
    customer_country        STRING      NOT NULL    COMMENT 'ISO country code',
    customer_region         STRING      NOT NULL    COMMENT 'US | EMEA | APAC | LATAM',
    customer_state          STRING                  COMMENT 'US state code (null for international)',

    -- Sales Rep Dimension
    sales_rep_id            STRING      NOT NULL    COMMENT 'Sales rep dimension FK',
    sales_rep_name          STRING      NOT NULL    COMMENT 'Full name of the sales representative',
    sales_rep_territory     STRING      NOT NULL    COMMENT 'Sales territory name',
    sales_rep_region        STRING      NOT NULL    COMMENT 'Sales rep region: US | EMEA | APAC | LATAM',

    -- Price Waterfall
    list_price              DOUBLE      NOT NULL    COMMENT 'Catalog list price with escalation and period multiplier (USD)',
    contract_price          DOUBLE      NOT NULL    COMMENT 'Price after contract discount (USD)',
    invoice_price           DOUBLE      NOT NULL    COMMENT 'Price after invoice adjustment (USD)',
    pocket_price            DOUBLE      NOT NULL    COMMENT 'Final pocket price after all deductions (USD)',
    contract_discount_pct   DOUBLE      NOT NULL    COMMENT 'Contract discount as fraction (0.00-0.55)',
    invoice_discount_pct    DOUBLE      NOT NULL    COMMENT 'Invoice adjustment as fraction (0.00-0.05)',
    pocket_discount_pct     DOUBLE      NOT NULL    COMMENT 'Total list-to-pocket discount as fraction',
    off_invoice_leakage_pct DOUBLE      NOT NULL    COMMENT 'Off-invoice leakage (rebates, chargebacks) as fraction',
    rebate_amount           DOUBLE      NOT NULL    COMMENT 'Rebate amount in USD',
    freight_cost            DOUBLE      NOT NULL    COMMENT 'Freight cost per unit in USD',
    other_deductions        DOUBLE      NOT NULL    COMMENT 'Other deductions (warranty, returns, allowances) in USD',

    -- Volume & Revenue
    volume                  INT         NOT NULL    COMMENT 'Units sold (seasonally adjusted)',
    gross_revenue           DOUBLE      NOT NULL    COMMENT 'List price * volume (USD)',
    net_revenue             DOUBLE      NOT NULL    COMMENT 'Pocket price * volume (USD)',
    invoice_revenue         DOUBLE      NOT NULL    COMMENT 'Invoice price * volume (USD)',

    -- Cost & Margin
    cogs_per_unit           DOUBLE      NOT NULL    COMMENT 'Cost of goods sold per unit (USD)',
    total_cogs              DOUBLE      NOT NULL    COMMENT 'COGS per unit * volume (USD)',
    gross_margin            DOUBLE      NOT NULL    COMMENT 'Invoice revenue - COGS (USD)',
    gross_margin_pct        DOUBLE      NOT NULL    COMMENT 'Gross margin as percentage',
    pocket_margin           DOUBLE      NOT NULL    COMMENT 'Net revenue - COGS (USD)',
    pocket_margin_pct       DOUBLE      NOT NULL    COMMENT 'Pocket margin as percentage',

    -- Time Dimensions
    transaction_date        DATE        NOT NULL    COMMENT 'Transaction date (2022-01-01 to 2024-12-31)',
    transaction_month       STRING      NOT NULL    COMMENT 'Year-month (yyyy-MM)',
    transaction_quarter     STRING      NOT NULL    COMMENT 'Calendar quarter (Q1 2022)',
    transaction_year        INT         NOT NULL    COMMENT 'Calendar year',
    fiscal_quarter          STRING      NOT NULL    COMMENT 'Fiscal quarter (FQ1-2022)',

    -- Contract & Competitive
    contract_id             STRING      NOT NULL    COMMENT 'Deterministic contract ID (hash of customer+product+quarter)',
    contract_tier           STRING      NOT NULL    COMMENT 'Platinum | Gold | Silver | Bronze | Standard',
    is_competitive_deal     BOOLEAN     NOT NULL    COMMENT 'Whether transaction involved competitive bid (20% probability)',
    competitor_reference    STRING                  COMMENT 'Competitor product name (populated only for competitive deals)'
)
USING DELTA
COMMENT '600K-row FICM pricing master fact table with full invoice-to-cash margin waterfall'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
```

#### `hls_amer_catalog.silver.dim_customers`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.silver.dim_customers (
    customer_id             STRING      NOT NULL    COMMENT 'Unique customer identifier (CUST-00000)',
    customer_name           STRING      NOT NULL    COMMENT 'Hospital/facility name',
    customer_segment        STRING      NOT NULL    COMMENT 'IDN | GPO | Direct | Distributor | Govt-VA | Academic',
    customer_tier           STRING      NOT NULL    COMMENT 'Customer tier: A | B | C | D',
    customer_country        STRING      NOT NULL    COMMENT 'ISO country code (US, DE, GB, JP, etc.)',
    customer_region         STRING      NOT NULL    COMMENT 'US | EMEA | APAC | LATAM',
    customer_state          STRING                  COMMENT 'US state code (null for international)',
    annual_spend            DOUBLE      NOT NULL    COMMENT 'Annual spend in USD',
    contract_count          INT         NOT NULL    COMMENT 'Number of active contracts',
    first_purchase_date     DATE        NOT NULL    COMMENT 'Date of first purchase (2010-2023)',
    is_active               BOOLEAN     NOT NULL    COMMENT 'Currently active flag (90% true)'
)
USING DELTA
COMMENT '500-row customer dimension: hospitals and facilities across 4 regions';
```

#### `hls_amer_catalog.silver.dim_sales_reps`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.silver.dim_sales_reps (
    sales_rep_id            STRING      NOT NULL    COMMENT 'Unique rep identifier (REP-0000)',
    sales_rep_name          STRING      NOT NULL    COMMENT 'Full name',
    sales_rep_territory     STRING      NOT NULL    COMMENT 'Sales territory name',
    sales_rep_region        STRING      NOT NULL    COMMENT 'US | EMEA | APAC | LATAM',
    sales_rep_country       STRING      NOT NULL    COMMENT 'ISO country code',
    hire_date               DATE        NOT NULL    COMMENT 'Date of hire (2012-2024)',
    is_active               BOOLEAN     NOT NULL    COMMENT 'Currently active flag (93% true)',
    annual_quota            DOUBLE      NOT NULL    COMMENT 'Annual revenue quota in USD ($1.5M-$12M)',
    ytd_attainment_pct      DOUBLE      NOT NULL    COMMENT 'Year-to-date quota attainment percentage',
    is_high_discounter      BOOLEAN     NOT NULL    COMMENT 'Flagged as systematic over-discounter (7 of 75)'
)
USING DELTA
COMMENT '75-row sales rep dimension with 7 embedded high-discounter flags';
```

#### `hls_amer_catalog.silver.dim_products`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.silver.dim_products (
    product_id              STRING      NOT NULL    COMMENT 'Unique product identifier (PROD-0000)',
    sku                     STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_name            STRING      NOT NULL    COMMENT 'Stryker-branded product name',
    product_family          STRING      NOT NULL    COMMENT 'Endoscopy | Joint Replacement | Trauma & Extremities | Spine | Instruments | Neurovascular | Sports Medicine',
    product_category        STRING      NOT NULL    COMMENT 'Sub-category within product family',
    business_unit           STRING      NOT NULL    COMMENT 'MedSurg | Ortho | Neuro',
    list_price              DOUBLE      NOT NULL    COMMENT 'Catalog list price in USD',
    cogs_per_unit           DOUBLE      NOT NULL    COMMENT 'Cost of goods sold per unit in USD',
    target_margin_pct       DOUBLE      NOT NULL    COMMENT 'Target gross margin percentage',
    is_sole_source          BOOLEAN     NOT NULL    COMMENT 'Whether product is sole-sourced',
    competitor_count        INT         NOT NULL    COMMENT 'Number of direct competitors',
    launch_date             DATE        NOT NULL    COMMENT 'Product launch date',
    lifecycle_stage         STRING      NOT NULL    COMMENT 'Launch | Growth | Mature | Decline'
)
USING DELTA
COMMENT '200-row product dimension across 7 families and 3 business units';
```

### Gold Layer Tables

#### `hls_amer_catalog.gold.discount_outliers`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.discount_outliers (
    outlier_id                  STRING      NOT NULL    COMMENT 'Deterministic hash ID for idempotent writes',
    sku                         STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_name                STRING      NOT NULL    COMMENT 'Product name',
    product_family              STRING      NOT NULL    COMMENT 'Product family',
    business_unit               STRING      NOT NULL    COMMENT 'Business unit: MedSurg | Ortho | Neuro',
    customer_id                 STRING      NOT NULL    COMMENT 'Customer identifier',
    customer_name               STRING      NOT NULL    COMMENT 'Customer name',
    customer_segment            STRING      NOT NULL    COMMENT 'Customer segment',
    customer_country            STRING      NOT NULL    COMMENT 'Customer country code',
    customer_region             STRING      NOT NULL    COMMENT 'Customer region',
    sales_rep_id                STRING      NOT NULL    COMMENT 'Sales rep identifier',
    sales_rep_name              STRING      NOT NULL    COMMENT 'Sales rep name',
    sales_rep_territory         STRING      NOT NULL    COMMENT 'Sales territory',
    rep_avg_discount_pct        DOUBLE      NOT NULL    COMMENT 'Rep average discount percentage for this peer group',
    rep_median_discount_pct     DOUBLE      NOT NULL    COMMENT 'Rep median discount percentage',
    peer_avg_discount_pct       DOUBLE      NOT NULL    COMMENT 'Peer group average discount',
    peer_median_discount_pct    DOUBLE      NOT NULL    COMMENT 'Peer group median discount',
    peer_stddev_discount_pct    DOUBLE      NOT NULL    COMMENT 'Peer group standard deviation',
    peer_count                  INT         NOT NULL    COMMENT 'Number of reps in peer group',
    z_score                     DOUBLE      NOT NULL    COMMENT 'Z-score: (rep_avg - peer_avg) / peer_stddev',
    discount_gap                DOUBLE      NOT NULL    COMMENT 'Rep avg - peer avg (percentage points)',
    discount_gap_vs_median      DOUBLE      NOT NULL    COMMENT 'Rep avg - peer median (percentage points)',
    severity                    STRING      NOT NULL    COMMENT 'Severity: Severe (>3.0) | Moderate (2.0-3.0) | Watch (1.5-2.0)',
    rep_volume                  BIGINT      NOT NULL    COMMENT 'Total units sold by this rep in this peer group',
    rep_revenue                 DOUBLE      NOT NULL    COMMENT 'Total pocket revenue for this rep',
    rep_transaction_count       BIGINT      NOT NULL    COMMENT 'Number of transactions',
    recovery_amount             DOUBLE      NOT NULL    COMMENT 'Potential recovery amount in USD',
    annualized_recovery         DOUBLE      NOT NULL    COMMENT 'Annualized recovery amount in USD',
    ranking_score               DOUBLE      NOT NULL    COMMENT 'Composite ranking score for prioritization',
    computed_at                 TIMESTAMP   NOT NULL    COMMENT 'Computation timestamp'
)
USING DELTA
COMMENT 'Discount outliers detected via Z-score analysis: 150-300 rows, $5M-$20M recovery potential';
```

#### `hls_amer_catalog.gold.price_elasticity`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.price_elasticity (
    sku                         STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    customer_segment            STRING      NOT NULL    COMMENT 'Customer segment',
    product_id                  STRING      NOT NULL    COMMENT 'Product identifier',
    product_name                STRING      NOT NULL    COMMENT 'Product name',
    product_family              STRING      NOT NULL    COMMENT 'Product family',
    business_unit               STRING      NOT NULL    COMMENT 'Business unit',
    elasticity_coefficient      DOUBLE      NOT NULL    COMMENT 'Log-log regression slope (beta): price elasticity of demand',
    r_squared                   DOUBLE      NOT NULL    COMMENT 'R-squared goodness of fit',
    p_value                     DOUBLE      NOT NULL    COMMENT 'Statistical significance p-value',
    intercept                   DOUBLE      NOT NULL    COMMENT 'Regression intercept (alpha)',
    stderr                      DOUBLE      NOT NULL    COMMENT 'Standard error of the slope',
    elasticity_class            STRING      NOT NULL    COMMENT 'Highly Inelastic | Inelastic | Unit Elastic | Elastic',
    confidence                  STRING      NOT NULL    COMMENT 'High | Medium | Low (based on R-squared and sample size)',
    sample_months               INT         NOT NULL    COMMENT 'Number of monthly observations used',
    distinct_price_points       INT         NOT NULL    COMMENT 'Number of distinct price levels observed',
    overall_avg_price           DOUBLE      NOT NULL    COMMENT 'Average price across all observations',
    avg_volume_monthly          DOUBLE      NOT NULL    COMMENT 'Average monthly volume',
    total_revenue               DOUBLE      NOT NULL    COMMENT 'Total revenue across all observations',
    safe_increase_1pct          DOUBLE      NOT NULL    COMMENT 'Max price increase at 1% volume loss threshold',
    safe_increase_3pct          DOUBLE      NOT NULL    COMMENT 'Max price increase at 3% volume loss threshold',
    safe_increase_5pct          DOUBLE      NOT NULL    COMMENT 'Max price increase at 5% volume loss threshold',
    revenue_impact_at_1pct      DOUBLE      NOT NULL    COMMENT 'Revenue impact at 1% volume loss increase',
    revenue_impact_at_3pct      DOUBLE      NOT NULL    COMMENT 'Revenue impact at 3% volume loss increase',
    computed_at                 TIMESTAMP   NOT NULL    COMMENT 'Computation timestamp'
)
USING DELTA
COMMENT 'Price elasticity coefficients via log-log regression: 800-1400 rows across SKU-segment pairs';
```

#### `hls_amer_catalog.gold.uplift_simulation`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.uplift_simulation (
    sku                         STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_name                STRING      NOT NULL    COMMENT 'Product name',
    product_family              STRING      NOT NULL    COMMENT 'Product family',
    business_unit               STRING      NOT NULL    COMMENT 'Business unit',
    customer_segment            STRING      NOT NULL    COMMENT 'Customer segment',
    customer_country            STRING      NOT NULL    COMMENT 'Customer country',
    sales_rep_id                STRING                  COMMENT 'Sales rep identifier (if applicable)',
    target_uplift_pct           DOUBLE      NOT NULL    COMMENT 'Target portfolio uplift percentage',
    uplift_score                DOUBLE      NOT NULL    COMMENT 'Composite score: 0.30*inelasticity + 0.25*discount_gap + 0.20*revenue_weight + 0.15*margin_headroom - 0.10*competitive_risk',
    proposed_increase_pct       DOUBLE      NOT NULL    COMMENT 'Suggested price increase percentage',
    revenue_impact              DOUBLE      NOT NULL    COMMENT 'Expected revenue impact in USD',
    volume_impact               DOUBLE      NOT NULL    COMMENT 'Expected volume impact (units)',
    cumulative_uplift_pct       DOUBLE      NOT NULL    COMMENT 'Running cumulative portfolio uplift percentage',
    is_above_target             BOOLEAN     NOT NULL    COMMENT 'Whether cumulative uplift exceeds target',
    rationale                   STRING      NOT NULL    COMMENT 'Human-readable rationale for the recommendation',
    computed_at                 TIMESTAMP   NOT NULL    COMMENT 'Computation timestamp'
)
USING DELTA
COMMENT 'Uplift simulation results: 2000-5000 rows ranked by composite uplift score';
```

#### `hls_amer_catalog.gold.pricing_recommendations`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.pricing_recommendations (
    recommendation_id           STRING      NOT NULL    COMMENT 'Unique recommendation identifier',
    sku                         STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_name                STRING      NOT NULL    COMMENT 'Product name',
    product_family              STRING      NOT NULL    COMMENT 'Product family',
    business_unit               STRING      NOT NULL    COMMENT 'Business unit',
    customer_segment            STRING                  COMMENT 'Target customer segment (nullable for portfolio-wide actions)',
    action_type                 STRING      NOT NULL    COMMENT 'INCREASE_PRICE | STANDARDIZE_DISCOUNT | REVIEW_REP | COMPETITIVE_ADJUSTMENT | MARGIN_RECOVERY',
    priority_score              DOUBLE      NOT NULL    COMMENT 'Composite priority score (0-100)',
    risk_level                  STRING      NOT NULL    COMMENT 'Low | Medium | High',
    revenue_impact              DOUBLE      NOT NULL    COMMENT 'Expected annual revenue impact in USD',
    suggested_change_pct        DOUBLE                  COMMENT 'Suggested price or discount change percentage',
    rationale                   STRING      NOT NULL    COMMENT 'Detailed rationale text',
    supporting_evidence         STRING                  COMMENT 'JSON object with supporting data points',
    sales_rep_id                STRING                  COMMENT 'Sales rep ID (for REVIEW_REP actions)',
    computed_at                 TIMESTAMP   NOT NULL    COMMENT 'Computation timestamp'
)
USING DELTA
COMMENT 'ML-generated pricing recommendations: 200+ rows across 5 action types';
```

#### `hls_amer_catalog.gold.top100_price_changes`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.top100_price_changes (
    rank                        INT         NOT NULL    COMMENT 'Priority rank (1-100)',
    recommendation_id           STRING      NOT NULL    COMMENT 'FK to pricing_recommendations',
    sku                         STRING      NOT NULL    COMMENT 'Stock keeping unit code',
    product_name                STRING      NOT NULL    COMMENT 'Product name',
    product_family              STRING      NOT NULL    COMMENT 'Product family',
    business_unit               STRING      NOT NULL    COMMENT 'Business unit',
    customer_segment            STRING                  COMMENT 'Target customer segment',
    customer_country            STRING                  COMMENT 'Target country',
    sales_rep                   STRING                  COMMENT 'Associated sales rep name',
    action_type                 STRING      NOT NULL    COMMENT 'INCREASE_PRICE | STANDARDIZE_DISCOUNT | MARGIN_RECOVERY',
    price_change_pct            DOUBLE      NOT NULL    COMMENT 'Recommended price/discount change percentage',
    revenue_impact              DOUBLE      NOT NULL    COMMENT 'Expected annual revenue impact in USD',
    risk_level                  STRING      NOT NULL    COMMENT 'Low | Medium | High',
    priority_score              DOUBLE      NOT NULL    COMMENT 'Composite priority score',
    quick_action_summary        STRING      NOT NULL    COMMENT 'One-sentence action description',
    computed_at                 TIMESTAMP   NOT NULL    COMMENT 'Computation timestamp'
)
USING DELTA
COMMENT 'Top 100 executive action list: highest-priority pricing actions, filterable and exportable';
```

#### `hls_amer_catalog.gold.custom_pricing_scenarios`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.custom_pricing_scenarios (
    scenario_id                 STRING      NOT NULL    COMMENT 'UUID primary key',
    user_id                     STRING      NOT NULL    COMMENT 'Opaque user identifier',
    user_email                  STRING      NOT NULL    COMMENT 'Creator email address',
    scenario_name               STRING      NOT NULL    COMMENT 'Human-readable scenario title',
    description                 STRING                  COMMENT 'Free-text description',
    assumptions                 STRING                  COMMENT 'JSON object of modeling assumptions',
    target_uplift_pct           DOUBLE                  COMMENT 'Target revenue uplift percentage',
    selected_skus               STRING                  COMMENT 'JSON array of selected SKU identifiers',
    selected_segments           STRING                  COMMENT 'JSON array of selected market segments',
    simulation_results          STRING                  COMMENT 'JSON object with simulation outputs',
    status                      STRING      NOT NULL    COMMENT 'Draft | Submitted | Reviewed | Approved',
    created_at                  TIMESTAMP   NOT NULL    COMMENT 'Row creation timestamp',
    updated_at                  TIMESTAMP   NOT NULL    COMMENT 'Row last-update timestamp',
    is_deleted                  BOOLEAN     NOT NULL    COMMENT 'Soft-delete flag (default false)'
)
USING DELTA
COMMENT 'User-created pricing scenarios with approval workflow';
```

#### `hls_amer_catalog.gold.external_market_data`

```sql
CREATE TABLE IF NOT EXISTS hls_amer_catalog.gold.external_market_data (
    data_source                 STRING      NOT NULL    COMMENT 'Origin system or provider name',
    upload_timestamp            TIMESTAMP   NOT NULL    COMMENT 'When the data was ingested',
    category                    STRING      NOT NULL    COMMENT 'tariff | commodity | fx | competitor',
    item_key                    STRING      NOT NULL    COMMENT 'Unique key within category (e.g., HTS code, ticker)',
    item_description            STRING      NOT NULL    COMMENT 'Human-readable description',
    value                       DOUBLE      NOT NULL    COMMENT 'Numeric value of the data point',
    unit                        STRING      NOT NULL    COMMENT 'Unit of measure (percent, USD, ratio, etc.)',
    effective_date              DATE        NOT NULL    COMMENT 'Date when the value takes effect',
    region                      STRING                  COMMENT 'Geographic region (nullable)',
    notes                       STRING                  COMMENT 'Additional context or annotations (nullable)'
)
USING DELTA
COMMENT 'External market signals: tariffs, commodities, FX rates, competitor intelligence';
```

---

## 6. Notebook Code Reference

### Data Generation Notebooks (12a-12)

| Notebook | Purpose | Input Tables | Output Tables | Key Algorithm |
|----------|---------|-------------|---------------|---------------|
| **12a** `12a_ficm_dimensions.py` | Generate 3 dimension tables: 500 customers, 75 sales reps (7 flagged high-discounters), 200 products across 7 families | None | `silver.dim_customers`, `silver.dim_sales_reps`, `silver.dim_products` | Deterministic generation (seed=42) with weighted random distributions: 60% US, 25% EMEA, 10% APAC, 5% LATAM. Customer segments weighted: IDN 25%, GPO 25%, Direct 20%, Distributor 15%, Govt-VA 8%, Academic 7%. |
| **12** `12_ficm_pricing_master.py` | Generate 600K FICM transactions with complete pricing waterfall | `silver.dim_customers`, `silver.dim_sales_reps`, `silver.dim_products` | `silver.ficm_pricing_master` | 12-step waterfall: list price (with annual escalation 0.5-3.5% by family) -> contract discount (segment-driven 8-38% + rep behavior) -> invoice adjustment (0-5%) -> off-invoice leakage (1-10%) -> rebates (2-6%) -> freight ($8-$350/unit) -> other deductions (0.5-2.5%) -> pocket price. Volume modulated by seasonality (hospital budget cycles), annual trends, and price elasticity by family. 3-5 distinct price points per SKU over 36 months. |

### Gold Analytics Notebooks (13-19)

| Notebook | Purpose | Input Tables | Output Tables | Key Algorithm |
|----------|---------|-------------|---------------|---------------|
| **13** `13_gold_discount_outliers.py` | Detect discount outlier reps using Z-score analysis within peer groups | `silver.ficm_pricing_master` | `gold.discount_outliers` | Peer group = (SKU, customer_segment). Compute per-peer-group mean and stddev of discount %. For each rep, z_score = (rep_avg - peer_avg) / peer_stddev. Severity: Severe >3.0, Moderate 2.0-3.0, Watch 1.5-2.0. Minimum 3 reps per peer group for statistical validity. Recovery = volume * list_price_avg * discount_gap. |
| **14** `14_gold_price_elasticity.py` | Compute price elasticity via log-log regression | `silver.ficm_pricing_master` | `gold.price_elasticity` | Aggregate to (SKU, segment, month) grain. For groups with >=6 observations and >=3 distinct price points, fit OLS: ln(volume) = alpha + beta * ln(price). Beta is the elasticity coefficient. Classification: |beta| <0.5 Highly Inelastic, 0.5-1.0 Inelastic, 1.0-1.5 Unit Elastic, >=1.5 Elastic. Implemented as grouped Pandas UDF using scipy.stats.linregress. |
| **15** `15_gold_uplift_simulation.py` | Score every SKU-customer-rep combination for uplift potential | `silver.ficm_pricing_master`, `gold.price_elasticity`, `gold.discount_outliers` | `gold.uplift_simulation` | Composite uplift_score = 0.30 * inelasticity + 0.25 * discount_gap + 0.20 * revenue_weight + 0.15 * margin_headroom - 0.10 * competitive_risk. Suggested increase = MIN(safe_increase_3pct, discount_gap * 0.5, 5.0). Cumulative uplift computed by sorting by score and running sum of portfolio contribution. |
| **16** `16_gold_pricing_recommendations.py` | Generate actionable recommendations from multiple signals | `gold.price_elasticity`, `gold.discount_outliers`, `gold.uplift_simulation` | `gold.pricing_recommendations` | 5 action types: INCREASE_PRICE (inelastic products), STANDARDIZE_DISCOUNT (reps above peer norms), REVIEW_REP (severe outliers), COMPETITIVE_ADJUSTMENT (positioning gaps), MARGIN_RECOVERY (off-invoice leakage). Priority score 0-100 combining signal strength, revenue impact, and risk. |
| **17** `17_gold_top100_recommended_changes.py` | Curate top 100 highest-priority executive actions | `gold.pricing_recommendations`, `gold.price_elasticity`, `gold.discount_outliers`, `gold.uplift_simulation` | `gold.top100_price_changes` | Filter to actionable types (INCREASE_PRICE, STANDARDIZE_DISCOUNT, MARGIN_RECOVERY). Require High or Medium confidence. Rank by priority_score DESC, take top 100. Enrich with all filter dimensions and generate quick_action_summary text. |
| **18** `18_create_custom_pricing_scenarios_table.py` | Create scenarios table with synthetic seed data | None | `gold.custom_pricing_scenarios` | Creates Delta table schema and populates with 18 synthetic scenarios from 5 mock users demonstrating all 4 workflow states (Draft, Submitted, Reviewed, Approved). |
| **19** `19_gold_external_data_integration.py` | Ingest external market data from Volume or synthetic fallback | Unity Catalog Volume or synthetic | `gold.external_market_data` | Reads CSV files from `/Volumes/hls_amer_catalog/gold/external_uploads/`. Falls back to synthetically generated data with 4 categories: tariff (Section 301, HTS codes), commodity (titanium, cobalt-chrome), fx (EUR/USD, JPY/USD), competitor (price list updates). |

### ML Model Training Notebooks (20-21)

| Notebook | Purpose | Input Tables | Output | Key Algorithm |
|----------|---------|-------------|--------|---------------|
| **20** `20_train_discount_anomaly_model.py` | Train Isolation Forest for discount anomaly detection | `silver.ficm_pricing_master` | MLflow model: `models.discount_anomaly_detector` | Aggregate to per-rep features: avg_discount_pct, discount_stddev, max_discount, volume_weighted_discount, transaction_count, unique_customers. StandardScaler -> IsolationForest(contamination=0.1, seed=42). Logged to MLflow with parameters, metrics, and model artifact. |
| **21** `21_train_advanced_elasticity_model.py` | Train GBR with quantile prediction intervals | `silver.ficm_pricing_master` | MLflow model: `models.advanced_elasticity_model` | Features: log_price, log_volume, month_sin, month_cos, segment_encoded, product_family_encoded. Three GradientBoostingRegressor models: (1) point estimate (squared_error loss), (2) lower bound (quantile loss, alpha=0.05), (3) upper bound (quantile loss, alpha=0.95). 80/20 train-test split. Metrics: RMSE, MAE, R-squared. |

### Validation Notebook (99)

| Notebook | Purpose | Input Tables | Output | Key Algorithm |
|----------|---------|-------------|--------|---------------|
| **99** `99_validate_end_to_end.py` | Comprehensive validation of all tables | All Silver and Gold tables | Formatted pass/fail report | Checks table existence, row counts (min thresholds and exact counts), schema completeness (required columns exist), null percentages, and data quality invariants. Summarizes API endpoint readiness. |

---

## 7. API Reference

### v1 Endpoints (Core Analytics)

| Method | Endpoint | Parameters | Description | Response Shape |
|--------|----------|-----------|-------------|----------------|
| `GET` | `/health` | None | Health check | `{"status": "healthy", "version": "1.0.0"}` |
| `GET` | `/api/v1/products` | `?category=`, `?segment=`, `?limit=500` | List products from catalog | `[{"product_id", "product_name", "category", "sub_category", "segment", "base_asp", "cogs_pct", "innovation_tier", "market_share_pct"}]` |
| `POST` | `/api/v1/simulate-price-change` | Body: `{"product_id": "PROD-001", "price_change_pct": 5.0}` | ML-driven price change simulation | `{"predicted_volume_change_pct", "predicted_revenue_impact", "predicted_margin_impact", "confidence_interval": {"lower", "upper"}, "top_sensitivity_factors": [{"feature", "impact"}], "competitive_risk_score"}` |
| `GET` | `/api/v1/portfolio-kpis` | None | Aggregate portfolio KPIs | `{"total_revenue", "avg_margin_pct", "yoy_growth_pct", "total_products", "revenue_by_segment": [{"segment", "value"}], "margin_by_segment": [{"segment", "value"}]}` |
| `GET` | `/api/v1/price-waterfall/{product_id}` | Path: `product_id` | Price-to-pocket waterfall | `[{"name", "value", "cumulative"}]` |
| `GET` | `/api/v1/competitive-landscape/{category}` | Path: `category` | Competitor pricing data | `[{"competitor", "avg_asp", "market_share", "innovation_score", "asp_trend_pct"}]` |
| `GET` | `/api/v1/external-factors` | None | Latest macro-economic indicators | `{"month", "cpi_medical", "tariff_rate_steel", "supply_chain_pressure", "hospital_capex_index", "reimbursement_trend", "raw_material_index"}` |
| `POST` | `/api/v1/batch-scenario` | Body: `{"scenarios": [{"product_id", "price_change_pct"}]}` | Batch price simulations | `{"results": [PriceChangeResponse]}` |

### v2 Endpoints (Advanced Analytics)

| Method | Endpoint | Parameters | Description |
|--------|----------|-----------|-------------|
| `GET` | `/api/v2/ficm/summary` | None | FICM pricing master summary: row count, date range, breakdowns by country/segment/family |
| `GET` | `/api/v2/ficm/schema` | None | Full column schema of ficm_pricing_master |
| `GET` | `/api/v2/discount-outliers/` | `?business_unit=`, `?customer_segment=`, `?customer_country=`, `?min_z_score=2.0`, `?severity=`, `?limit=100` | Filtered discount outliers |
| `GET` | `/api/v2/discount-outliers/summary` | None | Aggregate stats: total outliers, recovery $, by BU/severity/country |
| `GET` | `/api/v2/discount-outliers/by-rep` | `?sales_rep_id=` (required) | All outliers for a specific sales rep |
| `GET` | `/api/v2/price-elasticity/` | `?business_unit=`, `?customer_segment=`, `?elasticity_class=`, `?product_family=`, `?confidence=` | Filtered elasticity records |
| `GET` | `/api/v2/price-elasticity/safe-ranges` | `?sku=` (required), `?customer_segment=` | Safe pricing ranges for a specific SKU |
| `GET` | `/api/v2/price-elasticity/distribution` | None | Histogram bucket data for elasticity coefficient distribution |
| `POST` | `/api/v2/uplift-simulation/` | Body: `{"target_uplift_pct", "excluded_skus": [], "excluded_segments": [], "excluded_countries": [], "max_per_sku_increase": 5.0}` | On-the-fly uplift simulation with constraints |
| `GET` | `/api/v2/uplift-simulation/precomputed` | `?target=1.0`, `?limit=100` | Retrieve precomputed uplift results |
| `GET` | `/api/v2/uplift-simulation/summary` | None | Summary KPIs: actions needed, revenue impact, volume impact per target |
| `GET` | `/api/v2/top100-price-changes/` | `?country=`, `?product_family=`, `?segment=`, `?rep=`, `?business_unit=`, `?risk_level=`, `?min_revenue_impact=`, `?sort_by=revenue_impact`, `?sort_order=desc`, `?page=1`, `?page_size=25` | Paginated, filtered, sorted top 100 actions |
| `GET` | `/api/v2/top100-price-changes/filter-options` | None | Distinct values for each filterable dimension |
| `GET` | `/api/v2/top100-price-changes/export` | Same filters as GET `/` | CSV download of filtered actions |
| `GET` | `/api/v2/pricing-recommendations/` | `?action_type=`, `?risk_level=`, `?business_unit=`, `?product_family=`, `?limit=100`, `?sort_by=priority_score` | Filtered recommendations |
| `GET` | `/api/v2/pricing-recommendations/summary` | None | Summary KPIs by action type |
| `POST` | `/api/v2/external-data/upload` | File upload + `?category=general` | Upload Excel/CSV external data |
| `GET` | `/api/v2/external-data/` | `?category=` | Retrieve external data records |
| `GET` | `/api/v2/external-data/sources` | None | List uploaded data sources with metadata |
| `GET` | `/api/v2/pricing-scenarios/user-info` | None | Current user identity from OBO headers |
| `GET` | `/api/v2/pricing-scenarios/` | `?status=`, `?search=`, `?page=1`, `?page_size=25` | List scenarios (user-scoped or admin-all) |
| `POST` | `/api/v2/pricing-scenarios/` | Body: `{"name", "description", "parameters": {}, "status": "draft"}` | Create a new scenario |
| `GET` | `/api/v2/pricing-scenarios/{scenario_id}` | Path: `scenario_id` | Retrieve a single scenario |

### Example: Uplift Simulation

```bash
curl -X POST http://localhost:8000/api/v2/uplift-simulation/ \
  -H "Content-Type: application/json" \
  -d '{
    "target_uplift_pct": 1.0,
    "excluded_skus": [],
    "excluded_segments": ["Govt-VA"],
    "excluded_countries": [],
    "max_per_sku_increase": 5.0
  }'
```

Response:

```json
{
  "target_uplift_pct": 1.0,
  "max_per_sku_increase": 5.0,
  "exclusions": {
    "skus": [],
    "segments": ["Govt-VA"],
    "countries": []
  },
  "actions_count": 87,
  "total_revenue_impact": 28340000.00,
  "total_volume_impact": -8520.00,
  "actions": [
    {
      "sku": "SKU-0042",
      "product_name": "Triathlon II Total Knee System",
      "uplift_score": 0.92,
      "proposed_increase_pct": 2.8,
      "revenue_impact": 1240000.00,
      "volume_impact": -45.0,
      "rationale": "Highly inelastic (beta=-0.28), current discount 8pts above peer norm, low competitive risk"
    }
  ]
}
```

---

## 8. DABs Deployment Guide

### Databricks Asset Bundles Configuration

For multi-workspace deployment, create a `databricks.yml` at the project root:

```yaml
bundle:
  name: stryker-pricing-intelligence

workspace:
  host: https://fe-vm-hls-amer.cloud.databricks.com

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://fe-vm-hls-amer.cloud.databricks.com
    variables:
      catalog: hls_amer_catalog
      warehouse_id: 4b28691c780d9875
      app_name: stryker-pricing-intel-dev

  staging:
    workspace:
      host: https://staging.cloud.databricks.com
    variables:
      catalog: hls_staging_catalog
      warehouse_id: <staging-warehouse-id>
      app_name: stryker-pricing-intel-staging

  prod:
    workspace:
      host: https://prod.cloud.databricks.com
    variables:
      catalog: hls_prod_catalog
      warehouse_id: <prod-warehouse-id>
      app_name: stryker-pricing-intel

resources:
  apps:
    stryker_pricing_intel:
      name: ${var.app_name}
      description: "Stryker Pricing Intelligence Platform v2"
      source_code_path: ./app
```

### Deployment Commands

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy to development
databricks bundle deploy -t dev

# Deploy to staging
databricks bundle deploy -t staging

# Deploy to production
databricks bundle deploy -t prod

# Check deployment status
databricks bundle summary -t dev
```

### Environment-Specific Configuration

Create environment-specific `app.yaml` files:

```yaml
# app/app.yaml (parameterized)
command:
  - "uvicorn"
  - "backend.main:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"
env:
  - name: DATABRICKS_SERVING_ENDPOINT
    value: "stryker-pricing-models"
  - name: CATALOG_NAME
    value: "hls_amer_catalog"
  - name: SCHEMA_NAME
    value: "gold"
  - name: DATABRICKS_WAREHOUSE_ID
    value: "4b28691c780d9875"
```

### CI/CD Pipeline Integration

```bash
# Typical CI/CD flow:
# 1. Build frontend
cd app && python build.py

# 2. Run tests
pytest tests/ -v

# 3. Deploy to target
databricks bundle deploy -t ${ENVIRONMENT}

# 4. Grant permissions (first deployment only)
databricks apps get ${APP_NAME} --output json | jq '.service_principal_client_id'
# Then run Unity Catalog GRANTs
```

---

## 9. Customization Guide

### Adapting for Other MedTech Companies

This platform is designed as a reusable template for medical device pricing intelligence. Here is how to adapt it for other organizations:

#### Step 1: Update Product Families and Business Units

In `notebooks/12a_ficm_dimensions.py`, modify the `PRODUCT_FAMILY_CONFIG` dictionary:

| Stryker | Johnson & Johnson (DePuy Synthes) | Zimmer Biomet | Smith+Nephew |
|---------|-----------------------------------|---------------|--------------|
| Endoscopy | Ethicon (Surgical Devices) | Dental | Wound Management |
| Joint Replacement | DePuy Synthes (Joint Recon) | Knee | Sports Medicine |
| Trauma & Extremities | DePuy Synthes (Trauma) | Trauma | Trauma |
| Spine | DePuy Synthes (Spine) | Spine | -- |
| Instruments | Surgical Instruments | Surgical | Advanced Wound |
| Neurovascular | Cerenovus (Neuro) | -- | -- |
| Sports Medicine | Mitek (Sports Med) | Sports Medicine | Sports Medicine |

#### Step 2: Adjust Statistical Parameters

In `notebooks/12_ficm_pricing_master.py`, customize:

- `SEGMENT_DISCOUNT_RANGES`: Adjust discount ranges per customer segment
- `FAMILY_ELASTICITY`: Set elasticity ranges per product family
- `OFF_INVOICE_LEAKAGE_RANGES`: Configure leakage by segment
- `MONTHLY_SEASONAL_MULTIPLIERS`: Adjust for different fiscal year patterns
- `COMPETITOR_REFS`: Replace with relevant competitor products

#### Step 3: Update Catalog References

In `app/backend/utils/config.py`, update:

```python
CATALOG_NAME = "your_catalog"
SCHEMA_GOLD = "gold"
SCHEMA_SILVER = "silver"
```

#### Step 4: Rebrand the Frontend

In `app/frontend/src/App.jsx`, update:

- Brand name in sidebar header
- Color theme in Tailwind config (`stryker-primary`, `stryker-accent`, `stryker-background`)
- Navigation labels if product families differ

#### Step 5: Configure Deployment

Update `app/app.yaml` with the target warehouse ID and catalog name.

#### Common Customizations

| Customization | File(s) to Modify |
|--------------|-------------------|
| Add a product family | `12a_ficm_dimensions.py`, `12_ficm_pricing_master.py` |
| Change discount ranges | `12_ficm_pricing_master.py` (SEGMENT_DISCOUNT_RANGES) |
| Add a customer segment | `12a_ficm_dimensions.py` (CUSTOMER_SEGMENTS) |
| Change outlier thresholds | `13_gold_discount_outliers.py` (Z_SCORE_SEVERE, etc.) |
| Add a new API endpoint | `app/backend/routers/` (new router file) + `app/backend/main.py` (include_router) |
| Add a new frontend tab | `app/frontend/src/pages/` (new page) + `App.jsx` (route + nav item) |
| Change uplift scoring weights | `15_gold_uplift_simulation.py` (uplift_score formula) |
| Add a recommendation action type | `16_gold_pricing_recommendations.py` (new action generator) |

---

## 10. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Data Platform** | Databricks | Lakehouse | Unified data, analytics, and AI platform |
| **Data Format** | Delta Lake | 3.x | ACID transactions, time travel, schema evolution |
| **Catalog** | Unity Catalog | -- | Centralized governance, lineage, access control |
| **Compute** | Databricks SQL Warehouse | Serverless | Sub-second query execution for API endpoints |
| **Compute** | Databricks ML Runtime | 13.3 LTS+ | Notebook execution and ML model training |
| **Pipeline** | PySpark | 3.4+ | Distributed data processing for medallion architecture |
| **ML Framework** | scikit-learn | 1.3+ | Isolation Forest, GradientBoosting, ElasticNet |
| **ML Tracking** | MLflow | 2.x | Experiment tracking, model registry, artifact management |
| **ML Explainability** | SHAP | 0.42+ | Feature importance and sensitivity analysis |
| **Statistics** | scipy | 1.11+ | Log-log regression (linregress) for elasticity |
| **Backend** | FastAPI | 0.104+ | Async REST API framework with auto-generated OpenAPI docs |
| **Backend Server** | Uvicorn | 0.24+ | ASGI server for production deployment |
| **Backend SDK** | Databricks SDK | 0.20+ | SQL statement execution, workspace client, file operations |
| **Backend Models** | Pydantic | 2.5+ | Request/response validation and serialization |
| **Backend Auth** | OBO (On-Behalf-Of) | -- | User identity propagation from Databricks App SSO |
| **Frontend** | React | 18.2 | Component-based UI framework |
| **Frontend Router** | React Router | 6.20 | Client-side routing for SPA navigation |
| **Frontend Charts** | Recharts | 2.10 | Declarative charting library for React |
| **Frontend Viz** | D3.js | 7.8 | Low-level data visualization for custom charts |
| **Frontend Animation** | Framer Motion | 10.16 | Spring-based page transitions and micro-interactions |
| **Frontend Icons** | Heroicons | 2.0 | SVG icon set for navigation and UI elements |
| **Frontend Styling** | Tailwind CSS | 3.3 | Utility-first CSS framework |
| **Frontend Build** | Vite | 5.0 | Lightning-fast HMR and optimized production builds |
| **Frontend Testing** | Vitest | 1.6 | Unit testing with React Testing Library integration |
| **Deployment** | Databricks Apps | -- | Managed app hosting with Service Principal auth |
| **Deployment CLI** | Databricks CLI | 0.278+ | App creation, deployment, and permission management |
| **File Upload** | openpyxl | 3.1+ | Excel file parsing for external data ingestion |
| **File Upload** | python-multipart | 0.0.6+ | Multipart form handling for file uploads |
| **HTTP Client** | httpx | 0.25+ | Async HTTP client for model serving endpoint calls |

---

## Project Structure

```
stryker-pricing-intelligence/
+-- README.md                                    # This file
+-- .gitignore                                   # Python, Node, Databricks ignores
+-- data/
|   +-- sample_external_data.csv                 # Sample external data for testing
+-- notebooks/
|   +-- 01_synthetic_data/                       # Group 1: Original synthetic data generation
|   +-- 02_pipeline/                             # Group 2: Original medallion pipeline
|   +-- 03_ml_models/                            # Group 3: Original ML model training
|   +-- 12a_ficm_dimensions.py                   # v2: Dimension tables (500 customers, 75 reps, 200 products)
|   +-- 12_ficm_pricing_master.py                # v2: 600K FICM fact table with full waterfall
|   +-- 13_gold_discount_outliers.py             # v2: Z-score discount outlier detection
|   +-- 14_gold_price_elasticity.py              # v2: Log-log regression elasticity
|   +-- 15_gold_uplift_simulation.py             # v2: Portfolio uplift scoring and simulation
|   +-- 16_gold_pricing_recommendations.py       # v2: Multi-signal recommendation engine
|   +-- 17_gold_top100_recommended_changes.py    # v2: Top 100 executive action list
|   +-- 18_create_custom_pricing_scenarios_table.py  # v2: Scenario workflow table
|   +-- 19_gold_external_data_integration.py     # v2: External market data ingestion
|   +-- 20_train_discount_anomaly_model.py       # v2: Isolation Forest training
|   +-- 21_train_advanced_elasticity_model.py    # v2: GBR with quantile bounds
|   +-- 99_validate_end_to_end.py                # v2: End-to-end validation
+-- app/
|   +-- app.yaml                                 # Databricks App manifest
|   +-- build.py                                 # Frontend build + copy to static/
|   +-- deploy_to_databricks.py                  # Deployment automation script
|   +-- requirements.txt                         # Python backend dependencies
|   +-- frontend/
|   |   +-- package.json                         # Node dependencies and scripts
|   |   +-- vite.config.ts                       # Vite configuration
|   |   +-- src/
|   |       +-- App.jsx                          # Main app with routing (12 tabs)
|   |       +-- main.jsx                         # React entry point
|   |       +-- pages/                           # 12 page entry points
|   |       |   +-- Dashboard.jsx
|   |       |   +-- PriceSimulator.jsx
|   |       |   +-- PriceWaterfall.jsx
|   |       |   +-- CompetitiveLandscape.jsx
|   |       |   +-- ExternalFactors.jsx
|   |       |   +-- DiscountOutliers.jsx         # v2
|   |       |   +-- PriceElasticity.jsx          # v2
|   |       |   +-- UpliftSimulator.jsx          # v2
|   |       |   +-- Top100Changes.jsx            # v2
|   |       |   +-- AIRecommendations.jsx        # v2
|   |       |   +-- ExternalData.jsx             # v2
|   |       |   +-- PricingScenarios.jsx         # v2
|   |       +-- components/                      # Reusable UI components
|   |       |   +-- Dashboard/                   # Dashboard-specific components
|   |       |   +-- Simulator/                   # Price simulator components
|   |       |   +-- Waterfall/                   # Waterfall chart components
|   |       |   +-- Competitive/                 # Competitive landscape components
|   |       |   +-- ExternalFactors/             # External factors components
|   |       |   +-- DiscountOutliers/            # v2: Outlier components
|   |       |   +-- PriceElasticity/             # v2: Elasticity components
|   |       |   +-- UpliftSimulator/             # v2: Uplift components
|   |       |   +-- Top100Changes/               # v2: Top 100 components
|   |       |   +-- AIRecommendations/           # v2: AI recommendations components
|   |       |   +-- ExternalData/                # v2: External data components
|   |       |   +-- PricingScenarios/            # v2: Scenario components
|   |       |   +-- Layout/                      # Shell layout components
|   |       |   +-- shared/                      # Shared UI elements
|   |       +-- hooks/                           # Custom React hooks
|   |       +-- styles/                          # Global CSS / Tailwind config
|   |       +-- utils/                           # Frontend utility functions
|   |       +-- __tests__/                       # Frontend test files
|   +-- backend/
|       +-- main.py                              # FastAPI application (v1 + v2 routers)
|       +-- models.py                            # Pydantic request/response models
|       +-- __init__.py
|       +-- routers/                             # v2 API route handlers
|       |   +-- v2_ficm.py                       # FICM summary and schema
|       |   +-- v2_discount_outliers.py          # Discount outlier endpoints
|       |   +-- v2_price_elasticity.py           # Elasticity endpoints
|       |   +-- v2_uplift_simulation.py          # Uplift simulation endpoints
|       |   +-- v2_top100_changes.py             # Top 100 with pagination/export
|       |   +-- v2_pricing_recommendations.py    # Recommendation endpoints
|       |   +-- v2_external_data.py              # External data upload/query
|       |   +-- v2_pricing_scenarios.py          # Scenario CRUD with OBO auth
|       +-- services/                            # Business logic layer
|       |   +-- catalog.py                       # Product/waterfall/competitive queries
|       |   +-- prediction.py                    # ML model serving integration
|       |   +-- features.py                      # Feature engineering for predictions
|       |   +-- file_ingestion.py                # Excel/CSV parsing for uploads
|       |   +-- obo_auth.py                      # On-Behalf-Of user identity extraction
|       +-- utils/                               # Infrastructure utilities
|           +-- config.py                        # Environment configuration
|           +-- databricks_client.py             # SDK client + cached SQL execution
+-- tests/
    +-- __init__.py
    +-- test_api.py                              # FastAPI endpoint tests
    +-- test_ml_models.py                        # ML model validation tests
    +-- test_synthetic_data.py                   # Synthetic data quality tests
```

---

## Configuration

Environment variables used by the application:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABRICKS_WAREHOUSE_ID` | SQL Warehouse ID for all queries | -- | Yes |
| `CATALOG_NAME` | Unity Catalog name | `hls_amer_catalog` | No |
| `SCHEMA_NAME` | Gold schema name | `gold` | No |
| `DATABRICKS_SERVING_ENDPOINT` | Model serving endpoint name | `stryker-pricing-models` | No |
| `CACHE_TTL` | Cache time-to-live in seconds | `300` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `STATIC_FILES_DIR` | Path to frontend static build | `static` | No |
| `DATABRICKS_HOST` | Workspace URL (local dev only) | Auto in Databricks App | Dev only |
| `DATABRICKS_TOKEN` | Personal access token (local dev only) | Auto in Databricks App | Dev only |

---

## Running Tests

```bash
# Backend API tests (no Databricks connection needed)
pytest tests/test_api.py -v

# ML model validation tests (requires MLflow tracking server)
pytest tests/test_ml_models.py -v

# Synthetic data quality tests (requires Spark + Unity Catalog)
pytest tests/test_synthetic_data.py -v

# All tests
pytest tests/ -v

# Frontend tests
cd app/frontend
npm run test
```

---

## Frontend Development

```bash
cd app/frontend

# Install dependencies
npm install

# Start dev server (proxied to backend at :8000)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## License

Internal use only. Proprietary to the organization.
