"""
Design of Experiments (DoE) Module for Nanoemulgel Formulation Optimization.
Implements Box-Behnken Design (BBD), Central Composite Design (CCD),
and Simplex Mixture Design algorithms with coded/actual level conversion,
reproducible randomization, and scientific design selection guidance.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

def generate_box_behnken(num_factors: int, num_center: int = 5) -> np.ndarray:
    """
    Generates Box-Behnken Design in coded units (-1, 0, +1).
    - 3 factors: 12 edge points + num_center center points (Default: 17 runs)
    - 4 factors: 24 edge points + num_center center points (Default: 27-29 runs)
    """
    if num_factors not in [3, 4]:
        raise ValueError(f"Box-Behnken design is standardly supported for 3 or 4 factors. Received: {num_factors}")

    design = []
    factor_indices = list(range(num_factors))

    # All combinations of 2 factors taking values in {-1, +1} while remaining factors are 0
    for active_pair in itertools.combinations(factor_indices, 2):
        for combo in itertools.product([-1.0, 1.0], repeat=2):
            point = [0.0] * num_factors
            point[active_pair[0]] = combo[0]
            point[active_pair[1]] = combo[1]
            design.append(point)

    # Center points (0, 0, ... 0)
    for _ in range(num_center):
        design.append([0.0] * num_factors)

    return np.array(design, dtype=float)

def generate_central_composite(num_factors: int, num_center: int = 6, alpha_type: str = "face_centered") -> Tuple[np.ndarray, float]:
    """
    Generates Central Composite Design (CCD) in coded units.
    Consists of:
    1. Factorial points: 2^k full factorial combinations in {-1, +1}^k
    2. Axial (star) points: 2k points at distance +/- alpha on factor axes
    3. Center points: num_center runs at origin (0, 0, ... 0)
    """
    if num_factors not in [2, 3, 4]:
        raise ValueError(f"Central Composite Design supports 2, 3, or 4 factors. Received: {num_factors}")

    # Determine alpha
    if alpha_type == "face_centered":
        # CCF: alpha = 1.0 (axial points lie on the faces of the cube, no extrapolation)
        alpha = 1.0
    elif alpha_type == "rotatable":
        # CCR: alpha = (2^k)^(1/4)
        alpha = float((2 ** num_factors) ** 0.25)
    elif alpha_type == "spherical":
        # alpha = sqrt(k)
        alpha = float(np.sqrt(num_factors))
    else:
        alpha = 1.0

    design = []

    # 1. Factorial points
    for combo in itertools.product([-1.0, 1.0], repeat=num_factors):
        design.append(list(combo))

    # 2. Axial points
    for i in range(num_factors):
        for sign in [-1.0, 1.0]:
            axial_point = [0.0] * num_factors
            axial_point[i] = sign * alpha
            design.append(axial_point)

    # 3. Center points
    for _ in range(num_center):
        design.append([0.0] * num_factors)

    return np.array(design, dtype=float), alpha

def generate_simplex_lattice_mixture(num_components: int, degree: int = 2) -> np.ndarray:
    """
    Generates Simplex Lattice points for mixture design where sum(x_i) = 1.0.
    For degree=2 (quadratic), generates pure components, binary blends, and overall centroid.
    """
    if num_components not in [3, 4]:
        raise ValueError(f"Mixture design supports 3 or 4 components. Received: {num_components}")

    points = []
    # Generate all integer partitions that sum to degree
    def partitions(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for p in partitions(n - i, k - 1):
                    yield (i,) + p

    for part in partitions(degree, num_components):
        coords = [val / float(degree) for val in part]
        points.append(coords)

    # Add overall centroid (e.g., 1/3, 1/3, 1/3) if not present
    centroid = [1.0 / num_components] * num_components
    if not any(np.allclose(centroid, p) for p in points):
        points.append(centroid)

    return np.array(points, dtype=float)

def code_to_actual(coded: float, low: float, center: float, high: float) -> float:
    """
    Converts a coded factor value into its actual physical concentration.
    Supports linear scaling for symmetric ranges as well as piecewise conversion.
    """
    if np.isclose(coded, 0.0):
        return float(center)
    elif coded > 0:
        return float(center + coded * (high - center))
    else:
        return float(center + coded * (center - low))

def actual_to_code(actual: float, low: float, center: float, high: float) -> float:
    """
    Converts actual physical concentration into coded variable (-1, 0, +1).
    """
    if np.isclose(actual, center):
        return 0.0
    elif actual > center:
        denom = high - center
        return float((actual - center) / denom) if denom != 0 else 0.0
    else:
        denom = center - low
        return float((actual - center) / denom) if denom != 0 else 0.0

def create_design_matrix(
    design_type: str,
    factors: List[Dict[str, Any]],
    num_center: int = 5,
    random_seed: int = 42,
    alpha_type: str = "face_centered"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Creates a full DoE matrix including coded levels, actual concentrations,
    and randomized laboratory execution order.

    Returns:
    - df_design: Formatted pandas DataFrame for display and data entry
    - coded_matrix: np.ndarray of coded levels in randomized order
    - actual_matrix: np.ndarray of actual levels in randomized order
    """
    num_factors = len(factors)
    if num_factors < 2:
        raise ValueError("At least 2 factors are required to construct an experimental design.")

    if design_type == "Box-Behnken Design (BBD)":
        coded_base = generate_box_behnken(num_factors=num_factors, num_center=num_center)
    elif design_type == "Central Composite Design (CCD)":
        coded_base, _ = generate_central_composite(num_factors=num_factors, num_center=num_center, alpha_type=alpha_type)
    elif design_type == "Simplex Lattice Mixture Design":
        coded_base = generate_simplex_lattice_mixture(num_components=num_factors, degree=2)
    else:
        raise ValueError(f"Unsupported design type: {design_type}")

    n_runs = len(coded_base)
    std_order = np.arange(1, n_runs + 1)

    # Randomize run order using reproducible seed (as in original SAMPLE PYTHON.docx)
    rng = np.random.RandomState(random_seed)
    random_perm = rng.permutation(n_runs)
    run_order = np.empty_like(random_perm)
    run_order[random_perm] = np.arange(1, n_runs + 1)

    # Randomized coded matrix
    coded_randomized = coded_base[random_perm]

    # Convert coded values to actual concentrations
    actual_randomized = []
    for row in coded_randomized:
        actual_row = []
        for i, f in enumerate(factors):
            if design_type == "Simplex Lattice Mixture Design":
                # For mixture, row[i] is proportion in [0, 1]
                val = f["low"] + row[i] * (f["high"] - f["low"])
            else:
                val = code_to_actual(row[i], f["low"], f["center"], f["high"])
            actual_row.append(round(val, 4))
        actual_randomized.append(actual_row)

    actual_randomized = np.array(actual_randomized, dtype=float)

    # Build DataFrame
    df = pd.DataFrame()
    df["Std_Order"] = std_order[random_perm]
    df["Run_Order"] = np.arange(1, n_runs + 1)

    # Coded columns
    for i, f in enumerate(factors):
        code_col = f"Factor_{f['code']}_Coded"
        df[code_col] = np.round(coded_randomized[:, i], 3)

    # Actual columns
    for i, f in enumerate(factors):
        actual_col = f"{f['code']}_{f['name']} ({f['unit']})"
        df[actual_col] = actual_randomized[:, i]

    # Sort display by Run_Order
    df = df.sort_values(by="Run_Order").reset_index(drop=True)

    return df, coded_randomized, actual_randomized

def get_design_guidance(num_factors: int, is_mixture_constrained: bool = False, can_test_extremes: bool = False) -> Dict[str, Any]:
    """
    Expert system offering scientific selection advice between BBD, CCD, and Mixture designs.
    """
    if is_mixture_constrained:
        return {
            "recommended": "Simplex Lattice Mixture Design",
            "warning": (
                "CRITICAL PHARMACEUTICAL NOTICE: The ingredients have constrained proportions (e.g. sum = 100%). "
                "Applying standard Box-Behnken or CCD is mathematically invalid because the factors are not independent. "
                "A Simplex Mixture Design must be employed to properly evaluate Scheffe canonical polynomial models."
            ),
            "suitable_designs": ["Simplex Lattice Mixture Design"],
            "rationale": "Ingredient proportions sum to unity; collinearity violates standard factorial orthogonality."
        }

    if num_factors == 3:
        if not can_test_extremes:
            return {
                "recommended": "Box-Behnken Design (BBD)",
                "warning": None,
                "suitable_designs": ["Box-Behnken Design (BBD)", "Central Composite Design (CCD)"],
                "rationale": (
                    "RECOMMENDED FOR NANOEMULSIONS: 3-factor Box-Behnken Design requires only 17 runs (with 5 center points) "
                    "and completely avoids harsh simultaneous extreme factor combinations (e.g. highest oil with lowest surfactant). "
                    "This protects delicate nanoemulsion batches from phase cracking during experimental preparation."
                )
            }
        else:
            return {
                "recommended": "Central Composite Design (CCD)",
                "warning": None,
                "suitable_designs": ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"],
                "rationale": (
                    "CCD provides rotatability and uniform spherical estimation variance, but requires 20 runs for 3 factors "
                    "and tests axial points that may extend beyond standard rectangular concentration limits."
                )
            }
    elif num_factors == 4:
        if not can_test_extremes:
            return {
                "recommended": "Box-Behnken Design (BBD)",
                "warning": None,
                "suitable_designs": ["Box-Behnken Design (BBD)", "Central Composite Design (CCD)"],
                "rationale": (
                    "For 4 factors, BBD requires 27-29 runs, which is more economical than 4-factor CCD (30-31 runs) "
                    "and avoids multi-factor extreme vertices."
                )
            }
        else:
            return {
                "recommended": "Central Composite Design (CCD)",
                "warning": None,
                "suitable_designs": ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"],
                "rationale": "4-factor CCD provides comprehensive surface exploration if extreme axial points are physically feasible."
            }
    else:
        return {
            "recommended": "Central Composite Design (CCD)",
            "warning": f"BBD is only available for 3 or 4 factors. For {num_factors} factors, CCD or Factorial design is required.",
            "suitable_designs": ["Central Composite Design (CCD)"],
            "rationale": f"Factor count ({num_factors}) requires CCD configuration."
        }
