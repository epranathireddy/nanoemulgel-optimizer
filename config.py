"""
Configuration Module for Nanoemulgel Formulation Optimization System.
Defines constants, standard excipient lists, default factor and response templates,
statistical thresholds, and educational glossaries for pharmaceutical research.
"""

import os
import tempfile
from pathlib import Path

# Set Matplotlib config directory to avoid permission warnings on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_nanoemulgel")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "nanoemulgel.db"
REPORTS_DIR = BASE_DIR / "reports"
EXPORTS_DIR = BASE_DIR / "exports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_TITLE = "PharmaQbD: Nanoemulgel Optimization Platform"
APP_SUBTITLE = "Quality by Design (QbD) Powered Formulation Engineering & Statistical Modeling"
APP_VERSION = "2.0.0"
AUTHOR_INFO = "Pharmaceutical Research & Formulation Development"

# Standard Pharmaceutical Excipients Catalogue
STANDARD_EXCIPIENTS = {
    "oils": [
        "Clove Oil",
        "Eucalyptus Oil",
        "Oleic Acid",
        "Isopropyl Myristate (IPM)",
        "Olive Oil",
        "Castor Oil",
        "Peppermint Oil",
        "Capryol 90",
        "Labrafac PG",
        "Medium Chain Triglycerides (MCT)"
    ],
    "surfactants": [
        "Tween 80 (Polysorbate 80)",
        "Tween 20 (Polysorbate 20)",
        "Span 80 (Sorbitan monooleate)",
        "Span 20 (Sorbitan monolaurate)",
        "Labrasol",
        "Kolliphor EL (Cremophor EL)",
        "Kolliphor RH 40 (Cremophor RH 40)",
        "Sodium Lauryl Sulfate (SLS)"
    ],
    "cosurfactants": [
        "Transcutol P (Diethylene glycol monoethyl ether)",
        "PEG 400 (Polyethylene glycol 400)",
        "Propylene Glycol (PG)",
        "Ethanol (Absolute)",
        "Glycerol",
        "Isopropanol",
        "Plurol Oleique CC 497"
    ],
    "gelling_agents": [
        "Carbopol 940 (0.5 - 2.0% w/w)",
        "Carbopol 934 (0.5 - 2.0% w/w)",
        "Carbopol 980 (0.5 - 2.0% w/w)",
        "Pluronic F127 (Poloxamer 407)",
        "Sodium Carboxymethylcellulose (Na-CMC)",
        "Hydroxypropyl Methylcellulose (HPMC K4M)",
        "Xanthan Gum (0.5 - 1.5% w/w)",
        "Chitosan (1.0 - 2.5% w/w)"
    ]
}

# Default Factor Profiles (from SAMPLE PYTHON.docx and QbD literature)
DEFAULT_FACTORS = [
    {
        "code": "A",
        "name": "Oil concentration",
        "unit": "% w/w",
        "low": 5.0,
        "center": 10.0,
        "high": 15.0,
        "description": "Internal lipid phase concentration forming the core nanoemulsion droplets."
    },
    {
        "code": "B",
        "name": "Surfactant concentration",
        "unit": "% w/w",
        "low": 2.0,
        "center": 4.0,
        "high": 6.0,
        "description": "Primary surfactant lowering interfacial tension and providing steric/electrostatic stabilization."
    },
    {
        "code": "C",
        "name": "Co-surfactant concentration",
        "unit": "% w/w",
        "low": 1.0,
        "center": 2.0,
        "high": 3.0,
        "description": "Fluidizes interfacial film and enhances curvature for sub-micron droplet formation."
    }
]

# Default Dependent Responses (Critical Quality Attributes - CQAs)
DEFAULT_RESPONSES = [
    {
        "name": "Particle Size",
        "unit": "nm",
        "goal": "minimize",
        "target": None,
        "low": 20.0,
        "high": 250.0,
        "importance": 5,
        "enabled": True,
        "description": "Hydrodynamic droplet diameter measured by Dynamic Light Scattering (DLS). Smaller droplets enhance cutaneous penetration."
    },
    {
        "name": "Polydispersity Index (PDI)",
        "unit": "dimensionless",
        "goal": "minimize",
        "target": None,
        "low": 0.05,
        "high": 0.50,
        "importance": 4,
        "enabled": True,
        "description": "Measurement of droplet size distribution homogeneity. PDI < 0.3 indicates narrow, monodisperse globule distribution."
    },
    {
        "name": "Zeta Potential",
        "unit": "mV",
        "goal": "range",
        "target": -30.0,
        "low": -50.0,
        "high": -20.0,
        "importance": 3,
        "enabled": True,
        "description": "Electrokinetic surface charge. Absolute magnitude > |25-30 mV| prevents droplet flocculation and coalescence."
    },
    {
        "name": "Entrapment Efficiency",
        "unit": "%",
        "goal": "maximize",
        "target": None,
        "low": 60.0,
        "high": 100.0,
        "importance": 5,
        "enabled": True,
        "description": "Percentage of active drug successfully encapsulated within lipid nano-droplets versus free drug."
    },
    {
        "name": "Drug Content",
        "unit": "%",
        "goal": "target",
        "target": 100.0,
        "low": 95.0,
        "high": 105.0,
        "importance": 4,
        "enabled": True,
        "description": "Assay percentage of drug in the final nanoemulgel matrix conforming to pharmacopeial (USP/IP) limits (95-105%)."
    },
    {
        "name": "Viscosity",
        "unit": "cP",
        "goal": "range",
        "target": 18000.0,
        "low": 8000.0,
        "high": 35000.0,
        "importance": 3,
        "enabled": True,
        "description": "Rheological resistance to shear. Governs formulation retention on skin surface and extrusion from collapsible tube."
    },
    {
        "name": "Spreadability",
        "unit": "cm",
        "goal": "range",
        "target": 5.5,
        "low": 4.0,
        "high": 7.0,
        "importance": 3,
        "enabled": True,
        "description": "Spreading diameter under standardized weight. Higher spreadability ensures patient compliance and uniform topical film."
    },
    {
        "name": "In-vitro Drug Release (8h)",
        "unit": "%",
        "goal": "maximize",
        "target": None,
        "low": 65.0,
        "high": 100.0,
        "importance": 5,
        "enabled": True,
        "description": "Cumulative percentage of drug permeated across synthetic/biological membrane over 8 hours using Franz diffusion cells."
    }
]

# Statistical Quality Thresholds
STAT_THRESHOLDS = {
    "alpha": 0.05,
    "lack_of_fit_alpha": 0.05,  # Non-significant (p > 0.05) is desirable
    "max_r2_pred_diff": 0.20,    # |Adj R2 - Pred R2| should be <= 0.20
    "min_adeq_precision": 4.0,   # Signal-to-noise ratio > 4 is desirable
    "max_cv_percent": 10.0       # Precision threshold
}

# Educational Concept Explanations for B.Pharm Students
EDUCATIONAL_GLOSSARY = {
    "Box-Behnken Design (BBD)": (
        "An independent quadratic design requiring only 3 levels (-1, 0, +1) per factor. "
        "Unlike Central Composite Design, BBD does NOT contain combinations where all factors "
        "are simultaneously at their extreme low or extreme high levels. This avoids harsh, "
        "physically unstable formulation conditions (such as phase inversion or precipitation) "
        "common when preparing delicate nanoemulsions."
    ),
    "Central Composite Design (CCD)": (
        "A response surface design comprising factorial points, center points, and axial (star) "
        "points at distance alpha. CCD is well-suited for comprehensive sequential experimentation "
        "when extreme formulation operating regions can safely be manufactured without cracking the emulsion."
    ),
    "Mixture Design": (
        "A specialized design where ingredient proportions are constrained to sum to a fixed total (e.g. 100%). "
        "Standard BBD or CCD are NOT suitable for strictly constrained mixtures because changing one ingredient "
        "inherently forces another to change, violating variable independence."
    ),
    "Response Surface Methodology (RSM)": (
        "A mathematical and statistical technique that fits experimental data to polynomial equations "
        "to map the relationship between formulation variables (excipients) and responses (droplet size, release). "
        "It generates 2D contour and 3D surface topologies to pinpoint optimal formulation recipes."
    ),
    "ANOVA (Analysis of Variance)": (
        "A statistical procedure that partitions the total observed variation into explained model components "
        "(linear, interaction, quadratic) and unexplained experimental residual error. It determines whether "
        "formulation changes cause statistically real differences in quality attributes."
    ),
    "F-Value": (
        "The test statistic comparing model variance to residual variance. A large F-value implies that "
        "the variation caused by the formulation factors is significantly greater than random laboratory noise."
    ),
    "P-Value": (
        "The probability of obtaining the observed result by pure chance under the null hypothesis. "
        "In pharmaceutical research, p < 0.05 signifies a statistically significant factor or model (95% confidence level)."
    ),
    "R-Squared (R²)": (
        "The coefficient of determination, measuring the proportion of total response variance explained by the model. "
        "Values close to 1.0 indicate excellent model fit."
    ),
    "Adjusted R-Squared": (
        "R² penalized for the number of terms in the model. It increases only if new terms truly improve the model "
        "more than expected by chance, preventing over-fitting."
    ),
    "Predicted R-Squared": (
        "Calculated using PRESS (Predicted Residual Error Sum of Squares) by systematically leaving out one observation "
        "at a time. It measures how well the model predicts response values for new formulation batches. "
        "A difference between Adjusted R² and Predicted R² of less than 0.20 indicates good model consistency."
    ),
    "Adequate Precision": (
        "A signal-to-noise ratio comparing the predicted response range at design points to average prediction error. "
        "A ratio greater than 4.0 indicates that the model can be safely utilized to navigate the design space."
    ),
    "Lack of Fit": (
        "Compares the variation of the model around replicate observations (pure experimental error at center points). "
        "For a reliable model, Lack of Fit should be NON-SIGNIFICANT (p > 0.05), proving that the polynomial equation "
        "adequately captures the underlying surface curvature."
    ),
    "Derringer-Suich Desirability Function": (
        "A multi-criteria optimization methodology that transforms each response into an individual desirability (d) "
        "ranging from 0 (completely unacceptable) to 1 (ideal target). The geometric mean yields an overall composite "
        "desirability (D), identifying the balanced compromise recipe across all competing quality attributes."
    ),
    "Prediction Interval (95% PI)": (
        "The statistical range within which an individual future laboratory batch prepared at the optimal conditions "
        "is expected to fall with 95% confidence. Unlike a Confidence Interval (which bounds the mean), the PI accounts "
        "for both model uncertainty and individual batch-to-batch manufacturing variation."
    )
}
