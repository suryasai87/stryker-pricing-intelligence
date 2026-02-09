# Databricks notebook source

# MAGIC %md
# MAGIC # 01a - Stryker Product Master: Synthetic SKU Generation
# MAGIC
# MAGIC **Purpose**: Generate a deterministic, production-quality product master table containing
# MAGIC 220 synthetic Stryker SKUs spanning all five business segments (Orthopaedics, MedSurg,
# MAGIC Neurotechnology, Capital Equipment, Consumables).
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.bronze.stryker_products` (Delta, Unity Catalog)
# MAGIC
# MAGIC **Reproducibility**: All stochastic operations use `seed=42`.
# MAGIC
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | product_id | STRING | UUID-based unique product identifier |
# MAGIC | product_name | STRING | Realistic Stryker-branded product name |
# MAGIC | category | STRING | Top-level business segment |
# MAGIC | sub_category | STRING | Product sub-category within the segment |
# MAGIC | segment | STRING | Derived market segment label |
# MAGIC | base_asp | DOUBLE | Average selling price in USD |
# MAGIC | cogs_pct | DOUBLE | Cost-of-goods-sold as a fraction of ASP |
# MAGIC | launch_year | INT | Year the product was launched |
# MAGIC | patent_expiry_year | INT | Year the core patent expires |
# MAGIC | market_share_pct | DOUBLE | Estimated market share (0-100) |
# MAGIC | innovation_tier | INT | Innovation tier (1=legacy, 5=breakthrough) |
# MAGIC | price_floor | DOUBLE | Minimum allowable selling price |
# MAGIC | price_ceiling | DOUBLE | Maximum allowable selling price |
# MAGIC | reimbursement_code | STRING | CMS/payer reimbursement code |
# MAGIC | competitor_products | STRING | JSON array of competing products |
# MAGIC | switching_cost_index | DOUBLE | Switching cost index (0=easy, 1=locked-in) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Constants

# COMMAND ----------

import numpy as np
import uuid
import json
from typing import Dict, List, Tuple, Any

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

# ---------------------------------------------------------------------------
# Deterministic seed -- every random draw in this notebook is anchored here.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Unity Catalog target
# ---------------------------------------------------------------------------
TARGET_CATALOG: str = "hls_amer_catalog"
TARGET_SCHEMA: str = "bronze"
TARGET_TABLE: str = "stryker_products"
FULLY_QUALIFIED_TABLE: str = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.{TARGET_TABLE}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Product Configuration
# MAGIC
# MAGIC Each entry defines the sub-category count, ASP range, and COGS percentage range.
# MAGIC The sum of all `count` values equals **220 SKUs**.

# COMMAND ----------

PRODUCT_CONFIG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Orthopaedics": {
        "Joint Replacement - Hip": {
            "count": 20,
            "asp_range": (4500, 7500),
            "cogs_pct_range": (0.22, 0.35),
        },
        "Joint Replacement - Knee": {
            "count": 22,
            "asp_range": (5000, 8000),
            "cogs_pct_range": (0.20, 0.32),
        },
        "Joint Replacement - Shoulder": {
            "count": 12,
            "asp_range": (4000, 6500),
            "cogs_pct_range": (0.25, 0.38),
        },
        "Trauma - Plates & Screws": {
            "count": 25,
            "asp_range": (500, 3500),
            "cogs_pct_range": (0.15, 0.28),
        },
        "Trauma - IM Nails": {
            "count": 15,
            "asp_range": (1200, 4000),
            "cogs_pct_range": (0.18, 0.30),
        },
        "Extremities": {
            "count": 18,
            "asp_range": (800, 3000),
            "cogs_pct_range": (0.20, 0.35),
        },
    },
    "MedSurg": {
        "Instruments - Power Tools": {
            "count": 15,
            "asp_range": (8000, 45000),
            "cogs_pct_range": (0.30, 0.45),
        },
        "Endoscopy - Visualization": {
            "count": 10,
            "asp_range": (25000, 350000),
            "cogs_pct_range": (0.35, 0.50),
        },
        "Medical - Beds": {
            "count": 12,
            "asp_range": (18000, 55000),
            "cogs_pct_range": (0.40, 0.55),
        },
        "Medical - Stretchers": {
            "count": 10,
            "asp_range": (12000, 28000),
            "cogs_pct_range": (0.38, 0.52),
        },
        "Medical - Emergency": {
            "count": 8,
            "asp_range": (8000, 18000),
            "cogs_pct_range": (0.35, 0.48),
        },
    },
    "Neurotechnology": {
        "Neurovascular": {
            "count": 12,
            "asp_range": (800, 3500),
            "cogs_pct_range": (0.18, 0.30),
        },
        "Cranial & CMF": {
            "count": 8,
            "asp_range": (1500, 6000),
            "cogs_pct_range": (0.22, 0.35),
        },
    },
    "Capital Equipment": {
        "Robotics - Mako": {
            "count": 3,
            "asp_range": (1000000, 1500000),
            "cogs_pct_range": (0.45, 0.55),
        },
        "Navigation Systems": {
            "count": 5,
            "asp_range": (250000, 500000),
            "cogs_pct_range": (0.40, 0.52),
        },
    },
    "Consumables": {
        "Disposables": {
            "count": 15,
            "asp_range": (500, 1500),
            "cogs_pct_range": (0.12, 0.25),
        },
        "Biologics": {
            "count": 10,
            "asp_range": (1200, 4500),
            "cogs_pct_range": (0.20, 0.35),
        },
    },
}

# Validation: assert total SKU count matches expected configuration
EXPECTED_SKU_COUNT: int = 220
_total_skus = sum(
    cfg["count"]
    for segment in PRODUCT_CONFIG.values()
    for cfg in segment.values()
)
assert _total_skus == EXPECTED_SKU_COUNT, f"Expected {EXPECTED_SKU_COUNT} SKUs, got {_total_skus}"
print(f"Product config validated: {_total_skus} SKUs across {len(PRODUCT_CONFIG)} segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product Name Patterns & Reference Data
# MAGIC
# MAGIC Realistic product-name templates, competitor lists, and reimbursement-code
# MAGIC prefixes drawn from the actual Stryker portfolio and CMS code books.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Product name components -- assembled per sub-category to produce realistic
# Stryker-branded names.  Format: "Stryker {brand} {model} {descriptor}"
# ---------------------------------------------------------------------------
PRODUCT_NAME_PATTERNS: Dict[str, Dict[str, Any]] = {
    "Joint Replacement - Hip": {
        "brands": ["Accolade", "Trident", "LFit", "Anato", "SecurFit"],
        "models": ["II", "III", "Plus", "Advanced", "HA", "PSL", "X3"],
        "descriptors": [
            "Cemented Hip Stem",
            "Cementless Hip Stem",
            "Acetabular Shell",
            "Acetabular Liner",
            "Femoral Head",
            "Total Hip System",
            "Bipolar Hip System",
            "Revision Hip System",
        ],
    },
    "Joint Replacement - Knee": {
        "brands": ["Triathlon", "Scorpio", "Duracon", "NRG", "GetAroundKnee"],
        "models": [
            "CR",
            "PS",
            "CS",
            "TS",
            "PKR",
            "Tritanium",
            "X3",
            "StabilPost",
        ],
        "descriptors": [
            "Total Knee System",
            "Cruciate Retaining Knee",
            "Posterior Stabilized Knee",
            "Revision Knee System",
            "Partial Knee Replacement",
            "Unicompartmental Knee",
            "Tibial Baseplate",
            "Femoral Component",
        ],
    },
    "Joint Replacement - Shoulder": {
        "brands": ["ReUnion", "Aequalis", "Solar", "Ascend"],
        "models": ["Flex", "RSA", "TSA", "Perform", "Mini"],
        "descriptors": [
            "Reverse Shoulder Arthroplasty",
            "Total Shoulder Arthroplasty",
            "Humeral Stem",
            "Glenoid Baseplate",
            "Shoulder Fracture System",
            "Anatomic Shoulder System",
        ],
    },
    "Trauma - Plates & Screws": {
        "brands": ["VariAx", "AxSOS", "Peri-Loc", "Locking", "SolidLoc"],
        "models": [
            "2",
            "3",
            "Distal",
            "Proximal",
            "Periarticular",
            "LCP",
            "Mini",
        ],
        "descriptors": [
            "Distal Radius Plate",
            "Proximal Humerus Plate",
            "Distal Femur Plate",
            "Proximal Tibia Plate",
            "Clavicle Plate",
            "Pelvic Plate",
            "Olecranon Plate",
            "Locking Screw Set",
            "Cortical Screw System",
            "Cannulated Screw Set",
        ],
    },
    "Trauma - IM Nails": {
        "brands": ["T2", "Gamma", "Expert", "S2"],
        "models": [
            "Alpha",
            "Beta",
            "Lateral",
            "Supracondylar",
            "Retrograde",
            "Antegrade",
        ],
        "descriptors": [
            "Femoral Nail",
            "Tibial Nail",
            "Humeral Nail",
            "Cephalomedullary Nail",
            "Supracondylar Nail",
            "Retrograde Nail System",
        ],
    },
    "Extremities": {
        "brands": ["Acumed", "SBI", "Integra", "Trident"],
        "models": ["Polarus", "Acutrak", "Modular", "Universal"],
        "descriptors": [
            "Wrist Fusion Plate",
            "Headless Compression Screw",
            "Foot & Ankle Plating System",
            "Hand Fracture System",
            "Elbow Prosthesis",
            "Small Bone Fixation System",
            "Metatarsal Osteotomy System",
            "Distal Fibula Plate",
        ],
    },
    "Instruments - Power Tools": {
        "brands": ["System", "Stryker"],
        "models": ["8", "9", "7+", "Sabo", "Core", "TPS", "RemB"],
        "descriptors": [
            "Sagittal Saw",
            "Oscillating Saw",
            "Reciprocating Saw",
            "Pneumatic Drill",
            "Battery-Powered Drill",
            "Reamer",
            "Wire Driver",
            "Sternum Saw",
            "Craniotomy Perforator",
        ],
    },
    "Endoscopy - Visualization": {
        "brands": ["1688", "1788", "ProCare", "CrossFire"],
        "models": [
            "AIM 4K",
            "AIM Platform",
            "i-Suite",
            "Elite",
            "HD",
            "Fluorescence",
        ],
        "descriptors": [
            "4K Camera System",
            "Camera Head",
            "Light Source",
            "Image Management System",
            "Laparoscope",
            "Arthroscope",
            "Insufflator",
            "Video Tower",
        ],
    },
    "Medical - Beds": {
        "brands": ["InTouch", "S3", "Secure", "ProCuity"],
        "models": [
            "Critical Care",
            "Med/Surg",
            "Bariatric",
            "Low",
            "Epic II",
        ],
        "descriptors": [
            "ICU Bed",
            "Med-Surg Bed",
            "Bariatric Bed",
            "Low Hospital Bed",
            "Critical Care Bed Frame",
            "Bed Surface System",
        ],
    },
    "Medical - Stretchers": {
        "brands": ["Power-PRO", "Stryker", "MX-PRO"],
        "models": ["XT", "IT", "2", "XPS"],
        "descriptors": [
            "Powered Ambulance Cot",
            "Manual Ambulance Cot",
            "Bariatric Transport",
            "Stair Chair",
            "Emergency Stretcher",
            "Fluoroscopy Stretcher",
        ],
    },
    "Medical - Emergency": {
        "brands": ["LIFEPAK", "LUCAS", "Physio-Control"],
        "models": ["15", "20e", "3", "CR2", "1000"],
        "descriptors": [
            "Monitor/Defibrillator",
            "Chest Compression System",
            "AED System",
            "Transport Defibrillator",
            "Defibrillator with SpO2",
        ],
    },
    "Neurovascular": {
        "brands": ["Target", "Excelsior", "Surpass", "Neuroform"],
        "models": [
            "360",
            "Ultra",
            "SL-10",
            "Evolve",
            "Atlas",
            "Nano",
            "Helical",
        ],
        "descriptors": [
            "Detachable Coil",
            "Microcatheter",
            "Flow Diverter Stent",
            "Stent Retriever",
            "Intracranial Stent",
            "Aspiration Catheter",
        ],
    },
    "Cranial & CMF": {
        "brands": ["Medtronic", "Stryker CMF", "Leibinger"],
        "models": ["Universal", "MatrixMIDFACE", "Mini", "Mandible"],
        "descriptors": [
            "Cranial Fixation System",
            "Midface Plating System",
            "Mandible Reconstruction Plate",
            "Mesh Cranioplasty Implant",
            "Resorbable Fixation System",
        ],
    },
    "Robotics - Mako": {
        "brands": ["Mako"],
        "models": ["SmartRobotics", "TKA", "THA"],
        "descriptors": [
            "Robotic-Arm Assisted Total Knee System",
            "Robotic-Arm Assisted Total Hip System",
            "Robotic-Arm Assisted Partial Knee System",
        ],
    },
    "Navigation Systems": {
        "brands": ["NAV3i", "Scopis", "Stryker"],
        "models": ["Platform", "ENT", "Spine", "Cranial", "Hybrid"],
        "descriptors": [
            "Surgical Navigation Platform",
            "ENT Navigation System",
            "Spine Navigation System",
            "Cranial Navigation System",
            "Hybrid OR Navigation",
        ],
    },
    "Disposables": {
        "brands": ["Neptune", "PulseLavage", "Stryker"],
        "models": ["3", "Plus", "Ultra", "Solo", "Dual"],
        "descriptors": [
            "Waste Management System Cartridge",
            "Pulse Lavage Tip",
            "Surgical Blade",
            "Suction Irrigation Disposable",
            "Cast Padding",
            "Casting Tape",
            "Skin Closure Strip",
            "Cement Mixing Kit",
        ],
    },
    "Biologics": {
        "brands": ["OP-1", "Vitoss", "Stryker"],
        "models": ["BA", "Scaffold", "Foam", "Flow", "Morsels"],
        "descriptors": [
            "Bone Graft Substitute",
            "Demineralized Bone Matrix",
            "Bone Void Filler",
            "Osteobiologic Scaffold",
            "Bone Morphogenetic Protein",
            "Collagen Bone Graft",
        ],
    },
}

# ---------------------------------------------------------------------------
# Segment-level market position assumptions drive market share distributions.
# ---------------------------------------------------------------------------
MARKET_SHARE_PARAMS: Dict[str, Tuple[float, float]] = {
    "Orthopaedics": (18.0, 8.0),        # Strong leader; mean 18%, std 8%
    "MedSurg": (22.0, 10.0),             # Market-leading segment
    "Neurotechnology": (14.0, 6.0),      # Competitive niche
    "Capital Equipment": (35.0, 12.0),   # Dominant in robotics
    "Consumables": (10.0, 5.0),          # Fragmented market
}

# ---------------------------------------------------------------------------
# Reimbursement code prefixes per sub-category.
# Follows CMS HCPCS Level II and CPT patterns.
# ---------------------------------------------------------------------------
REIMBURSEMENT_PREFIXES: Dict[str, str] = {
    "Joint Replacement - Hip": "L8630",
    "Joint Replacement - Knee": "L8641",
    "Joint Replacement - Shoulder": "L8650",
    "Trauma - Plates & Screws": "L8680",
    "Trauma - IM Nails": "L8683",
    "Extremities": "L8699",
    "Instruments - Power Tools": "L8689",
    "Endoscopy - Visualization": "C1748",
    "Medical - Beds": "E0301",
    "Medical - Stretchers": "E0148",
    "Medical - Emergency": "E0617",
    "Neurovascular": "C1768",
    "Cranial & CMF": "L8699",
    "Robotics - Mako": "S2900",
    "Navigation Systems": "S2901",
    "Disposables": "A4649",
    "Biologics": "C9359",
}

# ---------------------------------------------------------------------------
# Competitor products per sub-category -- realistic market landscape.
# ---------------------------------------------------------------------------
COMPETITOR_PRODUCTS: Dict[str, List[str]] = {
    "Joint Replacement - Hip": [
        "Zimmer Biomet Taperloc",
        "DePuy Synthes Corail",
        "Smith+Nephew Anthology",
        "DePuy Synthes Pinnacle",
    ],
    "Joint Replacement - Knee": [
        "Zimmer Biomet Persona",
        "DePuy Synthes Attune",
        "Smith+Nephew JOURNEY II",
        "Medacta GMK Sphere",
    ],
    "Joint Replacement - Shoulder": [
        "Zimmer Biomet Comprehensive",
        "DePuy Synthes GLOBAL",
        "Exactech Equinoxe",
        "Lima SMR",
    ],
    "Trauma - Plates & Screws": [
        "DePuy Synthes LCP",
        "Zimmer Biomet NCB",
        "Smith+Nephew PERI-LOC",
        "Acumed Acu-Loc",
    ],
    "Trauma - IM Nails": [
        "DePuy Synthes Expert",
        "Zimmer Biomet Natural Nail",
        "Smith+Nephew TRIGEN",
    ],
    "Extremities": [
        "Acumed Polarus",
        "Wright Medical ORTHOLOC",
        "Integra LifeSciences",
        "Arthrex",
    ],
    "Instruments - Power Tools": [
        "DePuy Synthes Power Pro",
        "Zimmer Biomet Micro 100",
        "Conmed PRO6200",
        "Arthrex SynergyHD3",
    ],
    "Endoscopy - Visualization": [
        "Karl Storz IMAGE1 S",
        "Olympus VISERA ELITE III",
        "Arthrex SynergyUHD4",
        "Smith+Nephew LENS",
    ],
    "Medical - Beds": [
        "Hill-Rom Centrella",
        "Hillrom Progressa",
        "Linet Eleganza",
        "Arjo Enterprise",
    ],
    "Medical - Stretchers": [
        "Ferno iN/X",
        "Hill-Rom TransStar",
        "Pedigo Transport",
    ],
    "Medical - Emergency": [
        "Philips HeartStart",
        "ZOLL X Series",
        "Nihon Kohden",
        "Mindray BeneHeart",
    ],
    "Neurovascular": [
        "Medtronic Pipeline",
        "MicroVention FRED",
        "Cerenovus Embotrap",
        "Penumbra Jet",
    ],
    "Cranial & CMF": [
        "DePuy Synthes ProPlan",
        "KLS Martin IPS",
        "Medtronic Kanit",
        "OssDsign OSSA",
    ],
    "Robotics - Mako": [
        "Zimmer Biomet ROSA",
        "Smith+Nephew CORI",
        "Think Surgical TSolution One",
    ],
    "Navigation Systems": [
        "Medtronic StealthStation",
        "Brainlab Curve",
        "Zimmer Biomet iASSIST",
    ],
    "Disposables": [
        "Medline",
        "Cardinal Health",
        "Halyard",
        "Molnlycke",
        "3M",
    ],
    "Biologics": [
        "Medtronic Infuse",
        "Zimmer Biomet iFactor",
        "SeaSpine OsteoAMP",
        "NovaBone",
    ],
}

print("Reference data loaded: name patterns, market share params, reimbursement codes, competitor lists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Row-Generation Logic
# MAGIC
# MAGIC A pure-Python generator builds all 220 rows on the driver.  At this scale the
# MAGIC driver-side generation is negligible; the heavy lifting (Delta write, schema
# MAGIC registration) happens in Spark.

# COMMAND ----------

def _generate_product_rows(seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate all product master rows deterministically.

    The function walks *PRODUCT_CONFIG* in sorted key order so that insertion
    order is reproducible across runs.  A dedicated ``numpy.random.RandomState``
    seeded with *seed* guarantees identical output regardless of any external
    random state.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict maps column name to its scalar value, ready for Spark
        ``createDataFrame``.
    """
    rng = np.random.RandomState(seed)
    # Seed Python's uuid with a deterministic namespace so product_ids are stable.
    # uuid5 (SHA-1 based) is deterministic given a namespace + name.
    _uuid_namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace

    rows: List[Dict[str, Any]] = []
    global_idx: int = 0  # monotonic counter for uuid stability

    for category in sorted(PRODUCT_CONFIG.keys()):
        sub_categories = PRODUCT_CONFIG[category]
        for sub_category in sorted(sub_categories.keys()):
            cfg = sub_categories[sub_category]
            count: int = cfg["count"]
            asp_lo, asp_hi = cfg["asp_range"]
            cogs_lo, cogs_hi = cfg["cogs_pct_range"]

            name_cfg = PRODUCT_NAME_PATTERNS[sub_category]
            brands = name_cfg["brands"]
            models = name_cfg["models"]
            descriptors = name_cfg["descriptors"]

            mkt_mean, mkt_std = MARKET_SHARE_PARAMS[category]
            reimb_prefix = REIMBURSEMENT_PREFIXES[sub_category]
            competitors = COMPETITOR_PRODUCTS[sub_category]

            for i in range(count):
                global_idx += 1

                # --- Product ID (deterministic UUID5) ---
                product_id = str(
                    uuid.uuid5(_uuid_namespace, f"stryker-sku-{global_idx:04d}")
                )

                # --- Product Name ---
                brand = brands[rng.randint(0, len(brands))]
                model = models[rng.randint(0, len(models))]
                descriptor = descriptors[rng.randint(0, len(descriptors))]
                product_name = f"Stryker {brand} {model} {descriptor}"

                # --- Segment label (marketing-friendly) ---
                segment = f"{category} - {sub_category.split(' - ')[0]}"

                # --- ASP (log-uniform for wide ranges, uniform for narrow) ---
                if asp_hi / max(asp_lo, 1) > 10:
                    base_asp = float(
                        np.exp(rng.uniform(np.log(asp_lo), np.log(asp_hi)))
                    )
                else:
                    base_asp = float(rng.uniform(asp_lo, asp_hi))
                base_asp = round(base_asp, 2)

                # --- COGS % ---
                cogs_pct = round(float(rng.uniform(cogs_lo, cogs_hi)), 4)

                # --- Launch year (2005-2024, newer products more likely) ---
                launch_year = int(
                    np.clip(
                        rng.normal(loc=2018, scale=4),
                        2005,
                        2024,
                    )
                )

                # --- Patent expiry (launch + 12-20 years) ---
                patent_expiry_year = launch_year + int(rng.randint(12, 21))

                # --- Market share % ---
                market_share_pct = round(
                    float(np.clip(rng.normal(mkt_mean, mkt_std), 1.0, 55.0)),
                    2,
                )

                # --- Innovation tier (1-5) ---
                # Newer launches and higher-ASP products skew toward higher tiers.
                recency_bonus = max(0, (launch_year - 2015)) / 9.0  # 0..1
                asp_bonus = (base_asp - asp_lo) / max(asp_hi - asp_lo, 1)
                tier_mean = 1.5 + 2.0 * recency_bonus + 1.0 * asp_bonus
                innovation_tier = int(np.clip(round(rng.normal(tier_mean, 0.7)), 1, 5))

                # --- Price floor / ceiling ---
                price_floor = round(base_asp * rng.uniform(0.70, 0.85), 2)
                price_ceiling = round(base_asp * rng.uniform(1.15, 1.40), 2)

                # --- Reimbursement code ---
                reimb_suffix = f"{rng.randint(0, 100):02d}"
                reimbursement_code = f"{reimb_prefix}-{reimb_suffix}"

                # --- Competitor products (2-4 random competitors as JSON) ---
                n_competitors = rng.randint(2, min(5, len(competitors) + 1))
                chosen = rng.choice(
                    competitors, size=n_competitors, replace=False
                ).tolist()
                competitor_products = json.dumps(chosen)

                # --- Switching cost index (0-1) ---
                # Capital equipment and robotics have the highest switching costs.
                if category == "Capital Equipment":
                    sci_mean = 0.85
                elif category == "Consumables":
                    sci_mean = 0.20
                else:
                    sci_mean = 0.50 + 0.15 * recency_bonus
                switching_cost_index = round(
                    float(np.clip(rng.normal(sci_mean, 0.12), 0.0, 1.0)),
                    4,
                )

                rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "sub_category": sub_category,
                        "segment": segment,
                        "base_asp": base_asp,
                        "cogs_pct": cogs_pct,
                        "launch_year": launch_year,
                        "patent_expiry_year": patent_expiry_year,
                        "market_share_pct": market_share_pct,
                        "innovation_tier": innovation_tier,
                        "price_floor": price_floor,
                        "price_ceiling": price_ceiling,
                        "reimbursement_code": reimbursement_code,
                        "competitor_products": competitor_products,
                        "switching_cost_index": switching_cost_index,
                    }
                )

    return rows


product_rows = _generate_product_rows()
print(f"Generated {len(product_rows)} product rows")
print(f"Sample row:\n{json.dumps(product_rows[0], indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build PySpark DataFrame with Explicit Schema

# COMMAND ----------

PRODUCT_SCHEMA = StructType(
    [
        StructField("product_id", StringType(), nullable=False),
        StructField("product_name", StringType(), nullable=False),
        StructField("category", StringType(), nullable=False),
        StructField("sub_category", StringType(), nullable=False),
        StructField("segment", StringType(), nullable=False),
        StructField("base_asp", DoubleType(), nullable=False),
        StructField("cogs_pct", DoubleType(), nullable=False),
        StructField("launch_year", IntegerType(), nullable=False),
        StructField("patent_expiry_year", IntegerType(), nullable=False),
        StructField("market_share_pct", DoubleType(), nullable=False),
        StructField("innovation_tier", IntegerType(), nullable=False),
        StructField("price_floor", DoubleType(), nullable=False),
        StructField("price_ceiling", DoubleType(), nullable=False),
        StructField("reimbursement_code", StringType(), nullable=False),
        StructField("competitor_products", StringType(), nullable=False),
        StructField("switching_cost_index", DoubleType(), nullable=False),
    ]
)

spark = SparkSession.builder.getOrCreate()

df_products = spark.createDataFrame(product_rows, schema=PRODUCT_SCHEMA)

print(f"DataFrame created: {df_products.count()} rows, {len(df_products.columns)} columns")
df_products.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Quality Checks
# MAGIC
# MAGIC Validate the generated data before writing to Delta.

# COMMAND ----------

from pyspark.sql import functions as F

# --- 6a. Row count ---
row_count = df_products.count()
assert row_count == EXPECTED_SKU_COUNT, f"Expected {EXPECTED_SKU_COUNT} rows, got {row_count}"
print(f"[CHECK] Row count: {row_count} == {EXPECTED_SKU_COUNT}")

# --- 6b. No null values in any column ---
null_counts = df_products.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df_products.columns]
).collect()[0]
for col_name in df_products.columns:
    assert null_counts[col_name] == 0, f"Null values found in column: {col_name}"
print("[CHECK] No null values in any column")

# --- 6c. Unique product_ids ---
unique_ids = df_products.select("product_id").distinct().count()
assert unique_ids == EXPECTED_SKU_COUNT, f"Expected {EXPECTED_SKU_COUNT} unique product_ids, got {unique_ids}"
print(f"[CHECK] Unique product_ids: {unique_ids} == {EXPECTED_SKU_COUNT}")

# --- 6d. Segment distribution ---
print("\n[CHECK] SKU distribution by category:")
df_products.groupBy("category").count().orderBy("category").show(truncate=False)

# --- 6e. ASP range sanity ---
asp_stats = df_products.select(
    F.min("base_asp").alias("min_asp"),
    F.max("base_asp").alias("max_asp"),
    F.mean("base_asp").alias("avg_asp"),
).collect()[0]
print(f"[CHECK] ASP range: ${asp_stats['min_asp']:,.2f} - ${asp_stats['max_asp']:,.2f} (avg: ${asp_stats['avg_asp']:,.2f})")

# --- 6f. Price floor < base_asp < price ceiling ---
invalid_pricing = df_products.filter(
    (F.col("price_floor") >= F.col("base_asp"))
    | (F.col("base_asp") >= F.col("price_ceiling"))
).count()
assert invalid_pricing == 0, f"Found {invalid_pricing} rows with invalid floor/ceiling relationship"
print(f"[CHECK] Price floor < ASP < Price ceiling: all {row_count} rows valid")

# --- 6g. COGS percentage in (0, 1) ---
invalid_cogs = df_products.filter(
    (F.col("cogs_pct") <= 0) | (F.col("cogs_pct") >= 1)
).count()
assert invalid_cogs == 0, f"Found {invalid_cogs} rows with COGS % outside (0,1)"
print("[CHECK] COGS % in valid range (0, 1)")

# --- 6h. Innovation tier in [1, 5] ---
invalid_tier = df_products.filter(
    (F.col("innovation_tier") < 1) | (F.col("innovation_tier") > 5)
).count()
assert invalid_tier == 0, f"Found {invalid_tier} rows with innovation_tier outside [1,5]"
print("[CHECK] Innovation tier in [1, 5]")

# --- 6i. Switching cost index in [0, 1] ---
invalid_sci = df_products.filter(
    (F.col("switching_cost_index") < 0) | (F.col("switching_cost_index") > 1)
).count()
assert invalid_sci == 0, f"Found {invalid_sci} rows with switching_cost_index outside [0,1]"
print("[CHECK] Switching cost index in [0, 1]")

print("\n=== All data quality checks passed ===")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write to Delta Lake (Unity Catalog)

# COMMAND ----------

# Ensure the target schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_CATALOG}.{TARGET_SCHEMA}")
print(f"Schema ready: {TARGET_CATALOG}.{TARGET_SCHEMA}")

# COMMAND ----------

# Write the DataFrame as a managed Delta table with overwrite semantics.
# This is idempotent: re-running the notebook fully replaces the table.
(
    df_products
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULLY_QUALIFIED_TABLE)
)

print(f"Delta table written: {FULLY_QUALIFIED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Post-Write Validation & Table Metadata

# COMMAND ----------

# --- 8a. Add table comment ---
spark.sql(f"""
    COMMENT ON TABLE {FULLY_QUALIFIED_TABLE} IS
    'Synthetic Stryker product master with {EXPECTED_SKU_COUNT} SKUs across 5 segments. '
    'Generated deterministically (seed=42) for pricing intelligence analytics. '
    'Source notebook: 01a_product_master.py'
""")

# --- 8b. Add column comments ---
column_comments = {
    "product_id": "UUID-based unique product identifier",
    "product_name": "Realistic Stryker-branded product name",
    "category": "Top-level business segment (Orthopaedics, MedSurg, etc.)",
    "sub_category": "Product sub-category within the segment",
    "segment": "Derived market segment label for analytics grouping",
    "base_asp": "Average selling price in USD",
    "cogs_pct": "Cost-of-goods-sold as a fraction of ASP (0.0 to 1.0)",
    "launch_year": "Year the product was launched (2005-2024)",
    "patent_expiry_year": "Year the core patent expires",
    "market_share_pct": "Estimated market share percentage (0-55)",
    "innovation_tier": "Innovation tier rating (1=legacy, 5=breakthrough)",
    "price_floor": "Minimum allowable selling price in USD",
    "price_ceiling": "Maximum allowable selling price in USD",
    "reimbursement_code": "CMS/payer reimbursement code (HCPCS/CPT format)",
    "competitor_products": "JSON array of competing product names",
    "switching_cost_index": "Switching cost index (0=easy switch, 1=locked-in)",
}

for col_name, comment in column_comments.items():
    spark.sql(
        f"ALTER TABLE {FULLY_QUALIFIED_TABLE} ALTER COLUMN {col_name} COMMENT '{comment}'"
    )

print("Table and column comments applied")

# COMMAND ----------

# --- 8c. Read back and verify ---
df_verify = spark.table(FULLY_QUALIFIED_TABLE)
verify_count = df_verify.count()
assert verify_count == EXPECTED_SKU_COUNT, f"Post-write verification failed: expected {EXPECTED_SKU_COUNT}, got {verify_count}"

print(f"\n{'='*80}")
print(f"SUCCESS: {FULLY_QUALIFIED_TABLE}")
print(f"{'='*80}")
print(f"  Rows:       {verify_count}")
print(f"  Columns:    {len(df_verify.columns)}")
print(f"  Categories: {df_verify.select('category').distinct().count()}")
print(f"  Sub-cats:   {df_verify.select('sub_category').distinct().count()}")
print(f"{'='*80}")

# --- 8d. Show sample rows from each segment ---
print("\nSample products per segment:")
df_verify.groupBy("category", "sub_category").agg(
    F.count("*").alias("sku_count"),
    F.round(F.min("base_asp"), 2).alias("min_asp"),
    F.round(F.max("base_asp"), 2).alias("max_asp"),
    F.round(F.avg("market_share_pct"), 1).alias("avg_mkt_share"),
    F.round(F.avg("switching_cost_index"), 2).alias("avg_switch_cost"),
).orderBy("category", "sub_category").show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Total SKUs | 220 |
# MAGIC | Segments | 5 (Orthopaedics, MedSurg, Neurotechnology, Capital Equipment, Consumables) |
# MAGIC | Sub-categories | 17 |
# MAGIC | ASP range | ~$500 - ~$1.5M |
# MAGIC | Target table | `hls_amer_catalog.bronze.stryker_products` |
# MAGIC | Format | Delta (managed, Unity Catalog) |
# MAGIC | Reproducible | Yes (seed=42) |
# MAGIC
# MAGIC **Next notebook**: `01b_contract_master.py` -- generates synthetic contract/GPO data linked to these products.
