"""
Database Management Module for Nanoemulgel Optimization System.
Provides persistent SQLite storage for projects, experimental designs,
laboratory data entry, fitted statistical models, multi-criteria optimization,
and laboratory confirmation validations.
"""

import sqlite3
import json
import io
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from config import DATABASE_PATH

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Projects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        drug_name TEXT NOT NULL,
        dosage_form TEXT NOT NULL,
        route TEXT NOT NULL,
        oil_phase TEXT NOT NULL,
        surfactant TEXT NOT NULL,
        cosurfactant TEXT NOT NULL,
        gelling_agent TEXT NOT NULL,
        other_excipients TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Factors Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS factors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        factor_code TEXT NOT NULL,
        factor_name TEXT NOT NULL,
        unit TEXT NOT NULL,
        low_val REAL NOT NULL,
        center_val REAL NOT NULL,
        high_val REAL NOT NULL,
        description TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Responses Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        response_name TEXT NOT NULL,
        unit TEXT NOT NULL,
        goal TEXT NOT NULL,
        target_val REAL,
        low_val REAL NOT NULL,
        high_val REAL NOT NULL,
        importance INTEGER DEFAULT 3,
        enabled INTEGER DEFAULT 1,
        description TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Designs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS designs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        design_type TEXT NOT NULL,
        num_factors INTEGER NOT NULL,
        num_center_points INTEGER NOT NULL,
        random_seed INTEGER NOT NULL,
        design_table_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Experimental Data Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experimental_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        design_id INTEGER NOT NULL,
        data_table_json TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (design_id) REFERENCES designs(id) ON DELETE CASCADE
    );
    """)

    # Models Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        response_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        equation TEXT NOT NULL,
        coefficients_json TEXT NOT NULL,
        anova_json TEXT NOT NULL,
        fit_statistics_json TEXT NOT NULL,
        diagnostics_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Optimization Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimization_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        composite_desirability REAL NOT NULL,
        optimal_factors_json TEXT NOT NULL,
        predicted_responses_json TEXT NOT NULL,
        candidate_solutions_json TEXT NOT NULL,
        criteria_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    # Validation Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS validation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        predicted_json TEXT NOT NULL,
        observed_json TEXT NOT NULL,
        error_table_json TEXT NOT NULL,
        status TEXT NOT NULL,
        scientific_assessment TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

def save_project(project_data: Dict[str, Any], factors: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> int:
    """Creates or updates a project with its factors and responses."""
    conn = get_connection()
    cursor = conn.cursor()

    project_id = project_data.get("id")
    if project_id:
        cursor.execute("""
            UPDATE projects SET
                name=?, drug_name=?, dosage_form=?, route=?, oil_phase=?,
                surfactant=?, cosurfactant=?, gelling_agent=?, other_excipients=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (
            project_data["name"], project_data["drug_name"], project_data["dosage_form"],
            project_data["route"], project_data["oil_phase"], project_data["surfactant"],
            project_data["cosurfactant"], project_data["gelling_agent"],
            project_data.get("other_excipients", ""), project_id
        ))
        cursor.execute("DELETE FROM factors WHERE project_id=?", (project_id,))
        cursor.execute("DELETE FROM responses WHERE project_id=?", (project_id,))
    else:
        cursor.execute("""
            INSERT INTO projects (
                name, drug_name, dosage_form, route, oil_phase,
                surfactant, cosurfactant, gelling_agent, other_excipients
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_data["name"], project_data["drug_name"], project_data["dosage_form"],
            project_data["route"], project_data["oil_phase"], project_data["surfactant"],
            project_data["cosurfactant"], project_data["gelling_agent"],
            project_data.get("other_excipients", "")
        ))
        project_id = cursor.lastrowid

    for f in factors:
        cursor.execute("""
            INSERT INTO factors (project_id, factor_code, factor_name, unit, low_val, center_val, high_val, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id, f["code"], f["name"], f["unit"],
            float(f["low"]), float(f["center"]), float(f["high"]), f.get("description", "")
        ))

    for r in responses:
        cursor.execute("""
            INSERT INTO responses (
                project_id, response_name, unit, goal, target_val,
                low_val, high_val, importance, enabled, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id, r["name"], r["unit"], r["goal"],
            float(r["target"]) if r.get("target") is not None else None,
            float(r["low"]), float(r["high"]), int(r.get("importance", 3)),
            1 if r.get("enabled", True) else 0, r.get("description", "")
        ))

    conn.commit()
    conn.close()
    return project_id

def load_project(project_id: int) -> Optional[Dict[str, Any]]:
    """Loads a project along with its factors and responses."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects WHERE id=?", (project_id,))
    p_row = cursor.fetchone()
    if not p_row:
        conn.close()
        return None

    project = dict(p_row)

    cursor.execute("SELECT * FROM factors WHERE project_id=? ORDER BY factor_code", (project_id,))
    factors = [
        {
            "code": row["factor_code"],
            "name": row["factor_name"],
            "unit": row["unit"],
            "low": row["low_val"],
            "center": row["center_val"],
            "high": row["high_val"],
            "description": row["description"]
        }
        for row in cursor.fetchall()
    ]

    cursor.execute("SELECT * FROM responses WHERE project_id=? ORDER BY id", (project_id,))
    responses = [
        {
            "name": row["response_name"],
            "unit": row["unit"],
            "goal": row["goal"],
            "target": row["target_val"],
            "low": row["low_val"],
            "high": row["high_val"],
            "importance": row["importance"],
            "enabled": bool(row["enabled"]),
            "description": row["description"]
        }
        for row in cursor.fetchall()
    ]

    project["factors"] = factors
    project["responses"] = responses
    conn.close()
    return project

def list_projects() -> List[Dict[str, Any]]:
    """Lists all saved formulation projects."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, drug_name, dosage_form, created_at, updated_at FROM projects ORDER BY updated_at DESC")
    projects = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return projects

def delete_project(project_id: int):
    """Deletes a project and all associated cascades."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id=?", (project_id,))
    cursor.execute("DELETE FROM factors WHERE project_id=?", (project_id,))
    cursor.execute("DELETE FROM responses WHERE project_id=?", (project_id,))
    cursor.execute("DELETE FROM models WHERE project_id=?", (project_id,))
    cursor.execute("DELETE FROM optimization_results WHERE project_id=?", (project_id,))
    cursor.execute("DELETE FROM validation_results WHERE project_id=?", (project_id,))
    conn.commit()
    conn.close()

def save_design(project_id: int, design_type: str, num_factors: int, num_center: int, seed: int, df_design: pd.DataFrame) -> int:
    """Saves or updates the DoE matrix for a project."""
    conn = get_connection()
    cursor = conn.cursor()
    design_json = df_design.to_json(orient="split")

    cursor.execute("SELECT id FROM designs WHERE project_id=?", (project_id,))
    row = cursor.fetchone()
    if row:
        design_id = row["id"]
        cursor.execute("""
            UPDATE designs SET
                design_type=?, num_factors=?, num_center_points=?, random_seed=?, design_table_json=?
            WHERE id=?
        """, (design_type, num_factors, num_center, seed, design_json, design_id))
    else:
        cursor.execute("""
            INSERT INTO designs (project_id, design_type, num_factors, num_center_points, random_seed, design_table_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (project_id, design_type, num_factors, num_center, seed, design_json))
        design_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return design_id

def load_design(project_id: int) -> Optional[Dict[str, Any]]:
    """Loads the DoE matrix for a project."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM designs WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    design = dict(row)
    design["dataframe"] = pd.read_json(io.StringIO(design["design_table_json"]), orient="split")
    return design

def save_experimental_data(design_id: int, df_data: pd.DataFrame):
    """Saves measured experimental observations."""
    conn = get_connection()
    cursor = conn.cursor()
    data_json = df_data.to_json(orient="split")

    cursor.execute("SELECT id FROM experimental_data WHERE design_id=?", (design_id,))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE experimental_data SET data_table_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (data_json, row["id"]))
    else:
        cursor.execute("INSERT INTO experimental_data (design_id, data_table_json) VALUES (?, ?)", (design_id, data_json))

    conn.commit()
    conn.close()

def load_experimental_data(design_id: int) -> Optional[pd.DataFrame]:
    """Loads experimental data for a given design."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT data_table_json FROM experimental_data WHERE design_id=? ORDER BY id DESC LIMIT 1", (design_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return pd.read_json(io.StringIO(row["data_table_json"]), orient="split")

def save_model(project_id: int, response_name: str, model_type: str, equation: str,
               coefficients: Dict, anova: Dict, stats: Dict, diagnostics: Dict):
    """Saves regression and ANOVA statistics for a response."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM models WHERE project_id=? AND response_name=?", (project_id, response_name))
    cursor.execute("""
        INSERT INTO models (
            project_id, response_name, model_type, equation,
            coefficients_json, anova_json, fit_statistics_json, diagnostics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        project_id, response_name, model_type, equation,
        json.dumps(coefficients), json.dumps(anova), json.dumps(stats), json.dumps(diagnostics)
    ))
    conn.commit()
    conn.close()

def load_models(project_id: int) -> Dict[str, Dict[str, Any]]:
    """Loads all fitted models for a project."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE project_id=?", (project_id,))
    rows = cursor.fetchall()
    conn.close()

    models = {}
    for r in rows:
        models[r["response_name"]] = {
            "model_type": r["model_type"],
            "equation": r["equation"],
            "coefficients": json.loads(r["coefficients_json"]),
            "anova": json.loads(r["anova_json"]),
            "fit_statistics": json.loads(r["fit_statistics_json"]),
            "diagnostics": json.loads(r["diagnostics_json"])
        }
    return models

def save_optimization(project_id: int, desirability: float, optimal_factors: Dict,
                      predicted_responses: Dict, candidate_solutions: List[Dict], criteria: Dict):
    """Saves the optimal formulation solution."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM optimization_results WHERE project_id=?", (project_id,))
    cursor.execute("""
        INSERT INTO optimization_results (
            project_id, composite_desirability, optimal_factors_json,
            predicted_responses_json, candidate_solutions_json, criteria_json
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        project_id, desirability, json.dumps(optimal_factors),
        json.dumps(predicted_responses), json.dumps(candidate_solutions), json.dumps(criteria)
    ))
    conn.commit()
    conn.close()

def load_optimization(project_id: int) -> Optional[Dict[str, Any]]:
    """Loads saved optimization results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM optimization_results WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "composite_desirability": row["composite_desirability"],
        "optimal_factors": json.loads(row["optimal_factors_json"]),
        "predicted_responses": json.loads(row["predicted_responses_json"]),
        "candidate_solutions": json.loads(row["candidate_solutions_json"]),
        "criteria": json.loads(row["criteria_json"]),
        "created_at": row["created_at"]
    }

def save_validation(project_id: int, predicted: Dict, observed: Dict, error_table: List[Dict], status: str, assessment: str):
    """Saves laboratory validation results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM validation_results WHERE project_id=?", (project_id,))
    cursor.execute("""
        INSERT INTO validation_results (
            project_id, predicted_json, observed_json, error_table_json, status, scientific_assessment
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        project_id, json.dumps(predicted), json.dumps(observed), json.dumps(error_table), status, assessment
    ))
    conn.commit()
    conn.close()

def load_validation(project_id: int) -> Optional[Dict[str, Any]]:
    """Loads saved validation results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM validation_results WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "predicted": json.loads(row["predicted_json"]),
        "observed": json.loads(row["observed_json"]),
        "error_table": json.loads(row["error_table_json"]),
        "status": row["status"],
        "scientific_assessment": row["scientific_assessment"],
        "created_at": row["created_at"]
    }

# Automatically initialize database upon import
init_db()
