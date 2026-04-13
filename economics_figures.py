from __future__ import annotations

"""
CHP Steenwijk — local techno-economic figure generator
======================================================

Purpose
-------
Generate the same core techno-economic PNG figures as the earlier generator, now expanded to 27 figures,
but with scenario power outputs sourced from the improved plant-performance
model used in the Streamlit app, not from the workbook.

This script is meant for local use. It supports:
- pulling scenario defaults from thermo_engine_improved.py
- overriding power output, CAPEX, OPEX, and financial assumptions from JSON
- generating 27 high-resolution PNG figures plus CSV outputs and a ZIP pack

Typical usage
-------------
1) Install dependencies locally:
   pip install numpy pandas matplotlib CoolProp

2) Keep this file in the same folder as thermo_engine_improved.py

3) Optional: edit the config template JSON
   python chp_technoeconomic_figure_generator_local.py --write-config-template

4) Generate figures
   python chp_technoeconomic_figure_generator_local.py --config chp_local_config_template.json

You can also directly override the scenario power and economics in the config file.
"""

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "chp_requested_figures_local"

DEFAULT_FINANCIALS: Dict[str, Any] = {
    "electricity_price_eur_per_mwh": 130.0,
    "heat_price_eur_per_mwhth": 35.0,
    "biomass_price_eur_per_t": 65.0,
    "npv_horizon_years": 10,
    "cashflow_years": 20,
    "discount_rate": 0.10,
    "electricity_escalation": 0.00,
    "heat_escalation": 0.00,
    "biomass_escalation": 0.00,
    "om_escalation": 0.025,
}

DEFAULT_OM: Dict[str, Any] = {
    "true_fixed_base_kEUR": 210.0,
    "runtime_rate_eur_per_hr": 5.6,
    "biomass_throughput_eur_per_t": 1.0,
    "power_throughput_eur_per_mwh": 0.45,
    "start_cost_kEUR": 1.667,
    "starts_per_year": {
        "S1": 6, "S2": 6, "S3": 6, "S4": 18, "S5": 20,
        "S6": 6, "S7": 6, "S8": 6, "C1": 6, "C2": 6, "C3": 18,
    },
    "control_incremental_om_kEUR": {
        "S2": 8.0,
        "S5": 10.0,
        "C3": 10.0,
    },
    "filter_incremental_om_kEUR": 17.0,
    "fan_incremental_om_kEUR": 12.0,
    "lp_incremental_om_kEUR": 14.0,
}

DEFAULT_TECHNICALS: Dict[str, Any] = {
    "live_steam_pressure_bar": 50.0,
    "live_steam_temperature_C": 444.0,
    "extraction_pressure_bar": 2.5,
    "condenser_pressure_bar": 0.20,
    "feedwater_temperature_C": 130.0,
    "eta_hp": 0.80,
    "eta_lp": 0.78,
    "eta_exp": 0.75,
    "eta_gen": 0.94,
    "transformer_limit_kW": 2520.0,
    "biomass_lhv_mj_per_kg": 10.5,
    "boiler_efficiency": 0.85,
}

DEFAULT_SCENARIO_OVERRIDES: Dict[str, Any] = {
    # Example structure. Leave empty or edit locally.
    # "S8": {
    #     "P_export_kW": 1775.0,
    #     "AnnualMWh": 14300.0,
    #     "CAPEX_kEUR": 325.0,
    #     "HeatMWhth": 18200.0,
    #     "Biomass_tpy": 25000.0,
    #     "ScenarioIncrementalOM_kEUR": 18.0,
    # }
}

DEFAULT_CONFIG = {
    "financial_assumptions": DEFAULT_FINANCIALS,
    "om_assumptions": DEFAULT_OM,
    "technical_assumptions": DEFAULT_TECHNICALS,
    "scenario_overrides": DEFAULT_SCENARIO_OVERRIDES,
}

SORT_MAP = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6, "S7": 7, "S8": 8, "C1": 9, "C2": 10, "C3": 11}

FILTER_UPGRADE_IDS = {"S6", "S7", "C1", "C2"}
FAN_UPGRADE_IDS = {"S7", "C2"}
LP_EXPANDER_IDS = {"S8", "C1", "C2"}

PATHWAY_IDS = ["S1", "S2", "S6", "C1", "S7", "C2"]

DEFAULT_PHASE_TIMELINE = [
    {
        "label": "Phase 1: Quick Wins",
        "months": (0, 6),
        "capex_kEUR": 32,
        "delta_margin_kEUR": -10,
        "color": "#3C8D5A",
        "actions": "FGR tuning + O₂ optimisation + price dispatch",
        "payback_label": "n/a (heat offset)",
    },
    {
        "label": "Phase 2: Filter+DCS",
        "months": (4, 11),
        "capex_kEUR": 145,
        "delta_margin_kEUR": 85,
        "color": "#0B6FA4",
        "actions": "Filter bag replacement + DCS/EPEX API",
    },
    {
        "label": "Phase 3: Debottleneck",
        "months": (8, 16),
        "capex_kEUR": 520,
        "delta_margin_kEUR": 77,
        "color": "#D98C0B",
        "actions": "Filter expansion (40k) + ECO2 reinstatement",
    },
    {
        "label": "Phase 4: LP Exp.+Fan",
        "months": (14, 24),
        "capex_kEUR": 460,
        "delta_margin_kEUR": 227,
        "color": "#C0392B",
        "actions": "LP expander + FG fan upgrade (optional)",
    },
]

BAG_UTIL_DEFAULT = {
    "S1": 91.0, "S2": 97.0, "S3": 97.0, "S4": 96.0, "S5": 97.0,
    "S6": 77.0, "S7": 71.0, "S8": 97.0, "C1": 77.0, "C2": 71.0, "C3": 97.0,
}
FAN_UTIL_DEFAULT = {
    "S1": 84.0, "S2": 91.0, "S3": 91.0, "S4": 90.0, "S5": 91.0,
    "S6": 103.0, "S7": 77.0, "S8": 91.0, "C1": 103.0, "C2": 92.0, "C3": 91.0,
}
FG_NM3PH_DEFAULT = {
    "S1": 26200.0, "S2": 28400.0, "S3": 28400.0, "S4": 28100.0, "S5": 28400.0,
    "S6": 32600.0, "S7": 35400.0, "S8": 28400.0, "C1": 32600.0, "C2": 35400.0, "C3": 28400.0,
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = deep_update(cfg, user_cfg)
    return cfg


def write_config_template(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)


# -----------------------------------------------------------------------------
# Build scenario dataset from improved plant model
# -----------------------------------------------------------------------------
def build_scenario_dataframe_from_engine(cfg: Dict[str, Any]) -> pd.DataFrame:
    try:
        from thermo_engine_tespy import analyse_cycle, compute_scenario, SCENARIO_DEFS
    except Exception as exc:
        raise RuntimeError(
            "Could not import thermo_engine_improved.py. Keep it in the same folder and install its dependencies, especially CoolProp."
        ) from exc

    tech = cfg["technical_assumptions"]
    cycle = analyse_cycle(
        P_live_bar=tech["live_steam_pressure_bar"],
        T_live_C=tech["live_steam_temperature_C"],
        P_extraction_bar=tech["extraction_pressure_bar"],
        P_condenser_bar=tech["condenser_pressure_bar"],
        T_feedwater_C=tech["feedwater_temperature_C"],
        eta_HP=tech["eta_hp"],
        eta_LP=tech["eta_lp"],
        eta_exp=tech["eta_exp"],
        eta_gen=tech["eta_gen"],
    )

    results = [
        compute_scenario(
            cycle=cycle,
            transformer_limit_kW=tech["transformer_limit_kW"],
            LHV_MJkg=tech["biomass_lhv_mj_per_kg"],
            eta_boiler=tech["boiler_efficiency"],
            **sd,
        )
        for sd in SCENARIO_DEFS
    ]

    records: List[Dict[str, Any]] = []
    for r in results:
        records.append({
            "ID": r.sid,
            "Scenario": r.name,
            "Category": "Scenario" if r.sid.startswith("S") else "Case",
            "Steam_tph": float(r.steam_tph),
            "Extraction_tph": float(r.extraction_tph),
            "Condensing_tph": float(r.condensing_tph),
            "Availability": float(r.availability_pct) / 100.0,
            "CAPEX_kEUR": float(r.capex_kEUR),
            "LPExpander": "Y" if r.has_lp_expander else "N",
            "FilterUpgrade": "Y" if r.sid in FILTER_UPGRADE_IDS else "N",
            "FanUpgrade": "Y" if r.sid in FAN_UPGRADE_IDS else "N",
            "FG_Nm3ph": FG_NM3PH_DEFAULT.get(r.sid, np.nan),
            "BagUtil": BAG_UTIL_DEFAULT.get(r.sid, np.nan),
            "FanUtil": FAN_UTIL_DEFAULT.get(r.sid, np.nan),
            "HPPower_kW": float(r.P_HP),
            "LPPower_kW": float(r.P_LP),
            "Expander_kW": float(r.P_exp),
            "Gross_kW": float(r.P_gross),
            "Aux_kW": float(r.P_aux),
            "Net_kW": float(r.P_net),
            "Export_kW": float(r.P_export),
            "Curtailed_kW": float(r.P_curtailed),
            "AnnualMWh": float(r.annual_MWh_export),
            "AnnualMWhGross": float(r.annual_MWh_gross),
            "HeatMWhth": float(r.heat_output_kW * r.hours_yr / 1000.0),
            "Biomass_tpy": float(r.biomass_tpy),
            "ElecEff": float(r.elec_eff_pct_net) / 100.0,
            "CHPEff": float(r.chp_eff_pct) / 100.0,
            "Status": "WARN" if (FAN_UTIL_DEFAULT.get(r.sid, 0.0) > 100.0 or BAG_UTIL_DEFAULT.get(r.sid, 0.0) > 100.0) else "OK",
        })
    df = pd.DataFrame(records).sort_values("ID", key=lambda s: s.map(SORT_MAP)).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Scenario overrides
# -----------------------------------------------------------------------------
COLUMN_MAP = {
    "P_export_kW": "Export_kW",
    "P_net_kW": "Net_kW",
    "P_gross_kW": "Gross_kW",
    "P_aux_kW": "Aux_kW",
    "P_hp_kW": "HPPower_kW",
    "P_lp_kW": "LPPower_kW",
    "P_exp_kW": "Expander_kW",
    "annual_mwh": "AnnualMWh",
    "annual_mwh_export": "AnnualMWh",
    "annual_mwh_gross": "AnnualMWhGross",
    "heat_mwhth": "HeatMWhth",
    "biomass_tpy": "Biomass_tpy",
    "capex_keur": "CAPEX_kEUR",
    "availability": "Availability",
    "steam_tph": "Steam_tph",
    "extraction_tph": "Extraction_tph",
    "condensing_tph": "Condensing_tph",
    "ScenarioIncrementalOM_kEUR": "ScenarioIncrementalOM_kEUR_Override",
    "true_fixed_base_kEUR": "TrueFixedBase_kEUR_Override",
}


def apply_scenario_overrides(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    overrides = cfg.get("scenario_overrides", {}) or {}
    for sid, changes in overrides.items():
        if sid not in set(out["ID"]):
            continue
        idx = out.index[out["ID"] == sid][0]
        for k, v in changes.items():
            col = COLUMN_MAP.get(k, k)
            if col not in out.columns:
                out[col] = np.nan
            out.loc[idx, col] = v

        hours = 8760.0 * float(out.loc[idx, "Availability"])
        if pd.notna(out.loc[idx, "Export_kW"]) and ("annual_mwh" not in changes and "annual_mwh_export" not in changes):
            out.loc[idx, "AnnualMWh"] = float(out.loc[idx, "Export_kW"]) * hours / 1000.0
        if pd.notna(out.loc[idx, "Gross_kW"]) and ("annual_mwh_gross" not in changes):
            out.loc[idx, "AnnualMWhGross"] = float(out.loc[idx, "Gross_kW"]) * hours / 1000.0
        if pd.notna(out.loc[idx, "Net_kW"]) and pd.notna(out.loc[idx, "Export_kW"]):
            out.loc[idx, "Curtailed_kW"] = max(float(out.loc[idx, "Net_kW"]) - float(out.loc[idx, "Export_kW"]), 0.0)
    return out


# -----------------------------------------------------------------------------
# Economic model
# -----------------------------------------------------------------------------


def row_scenario_id(row: pd.Series) -> str:
    """Return scenario ID whether it is stored as a column or as the Series index name."""
    if isinstance(row, pd.Series):
        if "ID" in row.index and pd.notna(row["ID"]):
            return str(row["ID"])
        name = row.name
        if isinstance(name, str) and name:
            return name
    raise KeyError("Could not determine scenario ID from row; expected an 'ID' column or Series.name.")

def scenario_incremental_om_kEUR(row: pd.Series, om_cfg: Dict[str, Any]) -> float:
    sid = row_scenario_id(row)
    if pd.notna(row.get("ScenarioIncrementalOM_kEUR_Override", np.nan)):
        return float(row["ScenarioIncrementalOM_kEUR_Override"])

    val = float(om_cfg["control_incremental_om_kEUR"].get(sid, 0.0))
    if str(row["FilterUpgrade"]).upper() == "Y":
        val += float(om_cfg["filter_incremental_om_kEUR"])
    if str(row["FanUpgrade"]).upper() == "Y":
        val += float(om_cfg["fan_incremental_om_kEUR"])
    if str(row["LPExpander"]).upper() == "Y":
        val += float(om_cfg["lp_incremental_om_kEUR"])
    return val


def om_components_baseyear(row: pd.Series, om_cfg: Dict[str, Any]) -> Dict[str, float]:
    hours = 8760.0 * float(row["Availability"])
    true_fixed_base = float(row.get("TrueFixedBase_kEUR_Override", np.nan))
    if not np.isfinite(true_fixed_base):
        true_fixed_base = float(om_cfg["true_fixed_base_kEUR"])
    runtime = hours * float(om_cfg["runtime_rate_eur_per_hr"]) / 1000.0
    throughput = (
        float(row["Biomass_tpy"]) * float(om_cfg["biomass_throughput_eur_per_t"]) / 1000.0 +
        float(row["AnnualMWh"]) * float(om_cfg["power_throughput_eur_per_mwh"]) / 1000.0
    )
    starts = int(om_cfg["starts_per_year"].get(row_scenario_id(row), 6))
    start_stop = starts * float(om_cfg["start_cost_kEUR"])
    incremental = scenario_incremental_om_kEUR(row, om_cfg)
    fixed_like = true_fixed_base + start_stop + incremental
    variable_like = runtime + throughput
    total = fixed_like + variable_like
    return {
        "OperatingHours": hours,
        "TrueFixedBase_kEUR": true_fixed_base,
        "RuntimeOM_kEUR": runtime,
        "ThroughputOM_kEUR": throughput,
        "StartStopOM_kEUR": start_stop,
        "ScenarioIncrementalOM_kEUR": incremental,
        "FixedOMSharpened_kEUR": fixed_like,
        "VarOMSharpened_kEUR": variable_like,
        "TotalOMSharpened_kEUR": total,
    }


def om_components_year(row: pd.Series, year: int, om_cfg: Dict[str, Any], fin: Dict[str, Any]) -> Dict[str, float]:
    base = om_components_baseyear(row, om_cfg)
    mult = (1 + float(fin["om_escalation"])) ** (year - 1)
    out = {}
    for k, v in base.items():
        out[k] = v if k == "OperatingHours" else v * mult
    return out


def annual_revenue_costs_kEUR(row: pd.Series, fin: Dict[str, Any], om_cfg: Dict[str, Any], year: int = 1,
                              elec_price: Optional[float] = None,
                              biomass_price: Optional[float] = None,
                              heat_value: Optional[float] = None) -> Dict[str, float]:
    elec_price = float(fin["electricity_price_eur_per_mwh"] if elec_price is None else elec_price)
    biomass_price = float(fin["biomass_price_eur_per_t"] if biomass_price is None else biomass_price)
    heat_value = float(fin["heat_price_eur_per_mwhth"] if heat_value is None else heat_value)
    elec_rev = float(row["AnnualMWh"]) * elec_price * ((1 + float(fin["electricity_escalation"])) ** (year - 1)) / 1000.0
    heat_rev = float(row["HeatMWhth"]) * heat_value * ((1 + float(fin["heat_escalation"])) ** (year - 1)) / 1000.0
    biomass_cost = float(row["Biomass_tpy"]) * biomass_price * ((1 + float(fin["biomass_escalation"])) ** (year - 1)) / 1000.0
    om = om_components_year(row, year, om_cfg, fin)
    return {
        "ElecRevenue_kEUR": elec_rev,
        "HeatRevenue_kEUR": heat_rev,
        "TotalRevenue_kEUR": elec_rev + heat_rev,
        "BiomassCost_kEUR": biomass_cost,
        **om,
    }


def annual_margin_series(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any],
                         elec_price: Optional[float] = None,
                         biomass_price: Optional[float] = None,
                         heat_value: Optional[float] = None,
                         ids: Optional[Iterable[str]] = None) -> pd.Series:
    d = df if ids is None else df[df["ID"].isin(list(ids))]
    vals = []
    for _, row in d.iterrows():
        rc = annual_revenue_costs_kEUR(row, fin, om_cfg, year=1, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value)
        vals.append(rc["TotalRevenue_kEUR"] - rc["BiomassCost_kEUR"] - rc["TotalOMSharpened_kEUR"])
    return pd.Series(vals, index=d["ID"].values)


def revenue_series(df: pd.DataFrame, fin: Dict[str, Any], elec_price: Optional[float] = None,
                   heat_value: Optional[float] = None, ids: Optional[Iterable[str]] = None) -> pd.Series:
    elec_price = float(fin["electricity_price_eur_per_mwh"] if elec_price is None else elec_price)
    heat_value = float(fin["heat_price_eur_per_mwhth"] if heat_value is None else heat_value)
    d = df if ids is None else df[df["ID"].isin(list(ids))]
    vals = (d["AnnualMWh"] * elec_price + d["HeatMWhth"] * heat_value) / 1000.0
    return pd.Series(vals.values, index=d["ID"].values)


def delta_margin_series(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any],
                        elec_price: Optional[float] = None,
                        biomass_price: Optional[float] = None,
                        heat_value: Optional[float] = None,
                        ids: Optional[Iterable[str]] = None) -> pd.Series:
    m = annual_margin_series(df, fin, om_cfg, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value, ids=ids)
    baseline = annual_margin_series(df, fin, om_cfg, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value, ids=["S1"]).iloc[0]
    return m - baseline


def incremental_cashflow_table(df: pd.DataFrame, sid: str, fin: Dict[str, Any], om_cfg: Dict[str, Any],
                               capex_mult: float = 1.0,
                               elec_price: Optional[float] = None,
                               biomass_price: Optional[float] = None,
                               heat_value: Optional[float] = None,
                               years: Optional[int] = None) -> pd.DataFrame:
    years = int(fin["cashflow_years"] if years is None else years)
    row = df.set_index("ID").loc[sid]
    base = df.set_index("ID").loc["S1"]
    capex = float(row["CAPEX_kEUR"]) * capex_mult
    wacc = float(fin["discount_rate"])
    rows = [{
        "Year": 0,
        "IncrementalFCF_kEUR": -capex,
        "DiscountedIncrementalFCF_kEUR": -capex,
        "CumDiscountedIncrementalFCF_kEUR": -capex,
        "ScenarioFCF_kEUR": -capex,
    }]
    cum_disc = -capex
    for y in range(1, years + 1):
        rc_s = annual_revenue_costs_kEUR(row, fin, om_cfg, year=y, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value)
        rc_b = annual_revenue_costs_kEUR(base, fin, om_cfg, year=y, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value)
        fcf_s = rc_s["TotalRevenue_kEUR"] - rc_s["BiomassCost_kEUR"] - rc_s["TotalOMSharpened_kEUR"]
        fcf_b = rc_b["TotalRevenue_kEUR"] - rc_b["BiomassCost_kEUR"] - rc_b["TotalOMSharpened_kEUR"]
        incr = fcf_s - fcf_b
        disc = incr / ((1 + wacc) ** y)
        cum_disc += disc
        rows.append({
            "Year": y,
            "IncrementalFCF_kEUR": incr,
            "DiscountedIncrementalFCF_kEUR": disc,
            "CumDiscountedIncrementalFCF_kEUR": cum_disc,
            "ScenarioFCF_kEUR": fcf_s,
        })
    return pd.DataFrame(rows)


def npv_series(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any],
               capex_mult: float = 1.0,
               elec_price: Optional[float] = None,
               biomass_price: Optional[float] = None,
               heat_value: Optional[float] = None,
               ids: Optional[Iterable[str]] = None) -> pd.Series:
    horizon = int(fin["npv_horizon_years"])
    d = df if ids is None else df[df["ID"].isin(list(ids))]
    vals = []
    for sid in d["ID"]:
        cft = incremental_cashflow_table(df, sid, fin, om_cfg, capex_mult=capex_mult, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value, years=horizon)
        vals.append(cft["DiscountedIncrementalFCF_kEUR"].sum())
    return pd.Series(vals, index=d["ID"].values)


def discounted_payback_series(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any],
                              years: Optional[int] = None,
                              capex_mult: float = 1.0,
                              elec_price: Optional[float] = None,
                              biomass_price: Optional[float] = None,
                              heat_value: Optional[float] = None,
                              ids: Optional[Iterable[str]] = None) -> pd.Series:
    years = int(fin["cashflow_years"] if years is None else years)
    d = df if ids is None else df[df["ID"].isin(list(ids))]
    out = []
    for sid in d["ID"]:
        if float(df.set_index("ID").loc[sid, "CAPEX_kEUR"]) <= 0:
            out.append(np.nan)
            continue
        cft = incremental_cashflow_table(df, sid, fin, om_cfg, years=years, capex_mult=capex_mult, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value)
        cum = cft["CumDiscountedIncrementalFCF_kEUR"].values
        if np.all(cum < 0):
            out.append(np.nan)
            continue
        idx = int(np.argmax(cum >= 0))
        if idx == 0:
            out.append(0.0)
        else:
            prev = cum[idx - 1]
            curr = cum[idx]
            frac = 0.0 if curr == prev else (-prev) / (curr - prev)
            out.append((idx - 1) + frac)
    return pd.Series(out, index=d["ID"].values)


def annualized_value_series(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any],
                            capex_mult: float = 1.0,
                            elec_price: Optional[float] = None,
                            biomass_price: Optional[float] = None,
                            heat_value: Optional[float] = None,
                            ids: Optional[Iterable[str]] = None) -> pd.Series:
    annuity_factor = sum(1 / ((1 + float(fin["discount_rate"])) ** y) for y in range(1, int(fin["npv_horizon_years"]) + 1))
    d = df if ids is None else df[df["ID"].isin(list(ids))]
    delta = delta_margin_series(df, fin, om_cfg, elec_price=elec_price, biomass_price=biomass_price, heat_value=heat_value, ids=d["ID"].tolist())
    vals = delta.values - d["CAPEX_kEUR"].values * capex_mult / annuity_factor
    return pd.Series(vals, index=d["ID"].values)


def enrich_dataframe(df: pd.DataFrame, fin: Dict[str, Any], om_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    base_components = []
    for _, row in out.iterrows():
        comp = om_components_baseyear(row, om_cfg)
        comp["ID"] = row["ID"]
        base_components.append(comp)
    base_comp = pd.DataFrame(base_components).set_index("ID")
    for col in base_comp.columns:
        out[col] = out["ID"].map(base_comp[col])
    out["TotalRevenue_kEUR"] = out["ID"].map(revenue_series(out, fin))
    baseline_revenue = revenue_series(out, fin, ids=["S1"]).iloc[0]
    out["RevenueChange_kEUR"] = out["TotalRevenue_kEUR"] - baseline_revenue
    out["Margin_kEUR"] = out["ID"].map(annual_margin_series(out, fin, om_cfg))
    out["DeltaMargin_kEUR"] = out["ID"].map(delta_margin_series(out, fin, om_cfg))
    out["NPV_kEUR"] = out["ID"].map(npv_series(out, fin, om_cfg))
    out["PBP_yr"] = out["ID"].map(discounted_payback_series(out, fin, om_cfg))
    out["InvestmentAdjustedAnnualValue_kEUR"] = out["ID"].map(annualized_value_series(out, fin, om_cfg))
    return out


# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
BLUE = "#0B6FA4"
MID_BLUE = "#5F9BD1"
DARK_BLUE = "#2F4E85"
AMBER = "#D98C0B"
GREEN = "#3C8D5A"
RED = "#C0392B"
LIGHT_RED = "#D8745C"
GREY = "#8F8F8F"
DARK_GREY = "#4E4E4E"
LIGHT_GREY = "#D9D9D9"
TEAL = "#6FAFC8"
ORANGE = "#D59B3D"

SCENARIO_COLORS = {
    "S1": BLUE, "S2": MID_BLUE, "S3": "#7EB0C8", "S4": "#9FC2CF", "S5": "#97B9C6",
    "S6": AMBER, "S7": "#C26B1D", "S8": "#A3530E",
    "C1": DARK_BLUE, "C2": "#566EA5", "C3": "#7A8CC0",
}
LINE_STYLES = ['-', '--', ':', '-.', (0,(3,1,1,1)), (0,(5,1)), (0,(1,1)), (0,(5,2,1,2)), (0,(3,2)), (0,(2,1)), (0,(4,1,1,1))]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "legend.title_fontsize": 13,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 1.2,
    "axes.labelcolor": "black",
    "axes.labelweight": "bold",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.titlecolor": BLUE,
    "figure.dpi": 120,
    "savefig.dpi": 360,
})


def setup_canvas(title: str = "", subtitle: Optional[str] = None, figsize=(16, 9)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    return fig


def add_footer(fig, note: str = ""):
    return None


def wrap_label(text: str, width: int = 16) -> str:
    words = str(text).replace("/", "/ ").split()
    if not words:
        return str(text)
    lines = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if len(candidate) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)


def scenario_ticklabels(df: pd.DataFrame, width: int = 16) -> list[str]:
    return [f"{sid}\n{wrap_label(name, width)}" for sid, name in zip(df["ID"], df["Scenario"])]


def scenario_legend_label(sid: str, scenario_name: str) -> str:
    return f"{sid}: {scenario_name}"


def phase_payback_label(phase: Dict[str, Any]) -> str:
    explicit = phase.get("payback_label")
    if explicit:
        return str(explicit)
    delta = float(phase.get("delta_margin_kEUR", 0) or 0)
    capex = float(phase.get("capex_kEUR", 0) or 0)
    if delta <= 0:
        return "n/a"
    years = capex / delta
    if years < 1.0:
        months = int(round(years * 12))
        return f"~{months} months"
    return f"~{years:.1f} yr"


def savefig(fig, out_dir: Path, name: str) -> Path:
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def format_axes(ax):
    ax.grid(axis="y", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_spread_legend(ax, n_items: int, yshift: float = -0.30, fontsize: float = 11.5):
    ncol = 4 if n_items >= 8 else min(3, max(1, n_items))
    ax.legend(
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.0, yshift, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
        handlelength=2.8,
        columnspacing=1.2,
        handletextpad=0.5,
        labelspacing=0.6,
    )


SCATTER_LABEL_OFFSETS = {
    "S1": (8, 14),
    "S2": (18, -12),
    "S3": (8, -18),
    "S4": (18, 12),
    "S5": (8, 12),
    "C3": (18, -14),
    "C1": (8, -14),
    "C2": (8, 10),
    "S6": (8, 10),
    "S7": (8, 10),
    "S8": (8, 10),
}


def annotate_scenario_point(ax, sid: str, x0: float, y0: float, label: str, fontsize: float = 10.5):
    dx, dy = SCATTER_LABEL_OFFSETS.get(sid, (8, 8))
    ax.annotate(
        label,
        xy=(x0, y0),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=DARK_GREY,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        arrowprops=dict(arrowstyle="-", color=LIGHT_GREY, lw=0.8, alpha=0.8),
    )


def add_end_of_line_labels(ax, xvals, curve_map, label_map, color_map, fontsize: float = 12.0, min_gap_frac: float = 0.055):
    xvals = list(xvals)
    if not curve_map:
        return
    ymin, ymax = ax.get_ylim()
    yrng = max(ymax - ymin, 1e-9)
    min_gap = yrng * min_gap_frac
    items = []
    for sid, yvals in curve_map.items():
        y_end = float(yvals[-1])
        items.append([sid, y_end, y_end])
    items.sort(key=lambda t: t[1])
    prev = None
    for item in items:
        if prev is not None and item[2] - prev < min_gap:
            item[2] = prev + min_gap
        prev = item[2]
    upper = ymax - 0.02 * yrng
    for i in range(len(items) - 1, -1, -1):
        if items[i][2] > upper:
            items[i][2] = upper
            upper -= min_gap
    x_last = float(xvals[-1])
    x_first = float(xvals[0])
    xpad = (x_last - x_first) * 0.03 if x_last != x_first else 1.0
    ax.set_xlim(x_first, x_last + xpad * 2.2)
    for sid, y_true, y_lab in items:
        ax.annotate(
            label_map.get(sid, sid),
            xy=(x_last, y_true),
            xytext=(x_last + xpad, y_lab),
            textcoords='data',
            fontsize=fontsize,
            fontweight='bold',
            color=color_map.get(sid, DARK_GREY),
            ha='left',
            va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75),
            arrowprops=dict(arrowstyle='-', color=color_map.get(sid, LIGHT_GREY), lw=0.9, alpha=0.9),
            clip_on=False,
        )


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------

def generate_figures(df_in: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fin = cfg["financial_assumptions"]
    om_cfg = cfg["om_assumptions"]
    df = enrich_dataframe(df_in, fin, om_cfg)
    id_to_name = dict(zip(df["ID"], df["Scenario"]))
    investable = df[df["CAPEX_kEUR"] > 0].copy()
    top4_by_npv = investable.sort_values("NPV_kEUR", ascending=False)["ID"].head(4).tolist()
    files: List[Path] = []

    # 01 Electrical power output
    fig = setup_canvas("How Do the Scenarios Compare on Power Output?", "Export power defaults come from the improved app/engine unless overridden locally")
    ax = fig.add_axes([0.06, 0.16, 0.9, 0.66])
    x = np.arange(len(df))
    bars = ax.bar(x, df["Export_kW"], color=[SCENARIO_COLORS[i] for i in df["ID"]], width=0.64)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ticklabels(df), fontsize=11)
    ax.axhline(2505, color=LIGHT_RED, linestyle="--", linewidth=1.5, label="Nameplate capacity (2,505 kW)")
    ax.axhline(float(cfg["technical_assumptions"]["transformer_limit_kW"]), color=DARK_GREY, linestyle=":", linewidth=1.3, label="Transformer limit")
    ax.axhline(float(df.set_index("ID").loc["S1", "Export_kW"]), color=TEAL, linestyle="--", linewidth=1.2, alpha=0.75, label="Current baseline")
    ax.set_ylabel("Electrical Power Output (kW)", fontsize=15)
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for rect, val in zip(bars, df["Export_kW"]):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+12, f"{int(round(val)):,.0f}", ha="center", va="bottom", fontsize=13, weight="bold", color=DARK_GREY)
    ax.legend(loc="upper left", fontsize=15, frameon=True)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "01_electrical_power_output"))

    # 02 Production gap waterfall
    fig = setup_canvas("Where Is the Energy Being Lost?", "Decomposing the 9,953 MWh/yr production gap into four root causes")
    ax = fig.add_axes([0.06, 0.18, 0.9, 0.60])
    components = [("Theoretical\nMaximum", 20089), ("Steam Flow\nGap", -5808), ("Heat\nExtraction", -2010), ("SH Temp &\nVacuum", -758), ("Downtime\n(243 hrs)", -1377), ("Actual\nOutput", 10136)]
    x = np.arange(len(components))
    ax.bar(x[0], components[0][1], color=BLUE, width=0.52)
    curr = components[0][1]
    for i, (_, val) in enumerate(components[1:-1], start=1):
        new = curr + val
        bottom = min(curr, new)
        ax.bar(x[i], abs(val), bottom=bottom, color=LIGHT_RED, width=0.52)
        ax.text(x[i], bottom + abs(val) / 2, f"{val:,}", ha="center", va="center", fontsize=13, weight="bold", color="white")
        curr = new
    ax.bar(x[-1], components[-1][1], color=GREEN, width=0.52)
    ax.text(2.35, 16000, "Total gap: 9,953 MWh/yr (50%)", color=RED, fontsize=15, weight="bold", bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7E8E5", edgecolor=LIGHT_RED, alpha=0.9))
    ax.set_ylabel("Annual Generation (MWh/yr)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in components], fontsize=12.5)
    ax.grid(alpha=0.14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "02_production_gap_waterfall"))

    # 03 Constraints graph
    fig = setup_canvas("Five Constraints Limit Electrical Output", "Ranked by impact; the flue gas system dominates")
    ax = fig.add_axes([0.08, 0.18, 0.86, 0.60])
    labels = ["Bag filter", "FG fan", "FG temp window", "Condenser (summer)", "Availability", "SH temperature"]
    vals = [91, 84, 97, 100, 96.7, 98.4]
    colors = [ORANGE, "#6AAF84", "#C35D53", "#C35D53", "#C35D53", "#C35D53"]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.46)
    ax.axvline(100, color=LIGHT_RED, linestyle="--", linewidth=1.5, label="Limit (100%)")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Current as % of Limit", fontsize=15)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16)
    for b, v in zip(bars, vals):
        txt = f"{v:.1f}%" if abs(v-round(v)) > 1e-6 else f"{int(v)}%"
        ax.text(v + 1.2, b.get_y() + b.get_height()/2, txt, va="center", ha="left", fontsize=15, weight="bold", color=DARK_GREY)
    ax.legend(loc="lower right", fontsize=14.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "03_constraints_graph"))

    # 04 FG bottleneck
    fig = setup_canvas("The FG Bottleneck: Filter Saturates Before Fan", "The bag filter reaches capacity at ~10.7 t/h steam; before the fan at ~11.6 t/h")
    ax = fig.add_axes([0.06, 0.17, 0.9, 0.62])
    x = np.linspace(7, 14, 200)
    y = 2777.77777778 * x - 730.55555556
    ax.plot(x, y, color="#0A567F", linewidth=2.2, label="FG volume vs steam rate")
    ax.axhline(29000, color=RED, linestyle="--", linewidth=1.6, label="Filter limit (29,000 Nm³/h)")
    ax.axhline(31500, color=AMBER, linestyle="--", linewidth=1.6, label="Fan limit (31,500 Nm³/h)")
    ax.axhline(40000, color=GREEN, linestyle=":", linewidth=1.5, label="Upgrade target (40,000 Nm³/h)")
    for xpos, text, color in [(9.7, "Current\n(9.7 t/h)", "#0A567F"), (10.7, "Filter\nsaturates", RED), (11.6, "Fan\nsaturates", AMBER), (13.0, "Design\n(13 t/h)", GREEN)]:
        ypos = 2777.77777778 * xpos - 730.55555556
        ax.axvline(xpos, color=color, linestyle=":", linewidth=0.8, alpha=0.35)
        ax.annotate(text, xy=(xpos, ypos), xytext=(xpos + 0.25, ypos + 2200), arrowprops=dict(arrowstyle="->", color=color, lw=1.2), fontsize=14.5, color=color, weight="bold")
    ax.set_xlim(7, 14); ax.set_ylim(15000, 42000)
    ax.set_xlabel("Steam Production Rate (t/h)", fontsize=15)
    ax.set_ylabel("Flue Gas Volume (Nm³/h)", fontsize=15)
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=14.5)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "04_fg_bottleneck"))

    # 05 Delta margin comparison
    fig = setup_canvas("Delta Margin Comparison")
    ax = fig.add_axes([0.06, 0.16, 0.9, 0.64])
    x = np.arange(len(df))
    bars = ax.bar(x, df["DeltaMargin_kEUR"], color=[SCENARIO_COLORS[s] for s in df["ID"]], width=0.60)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ticklabels(df), fontsize=11)
    ax.axhline(0, color=TEAL, linestyle="--", linewidth=1.2, alpha=0.8, label="Baseline (S1 = 0)")
    ax.set_ylabel("Δ Margin (kEUR/yr)", fontsize=15)
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for rect, val in zip(bars, df["DeltaMargin_kEUR"]):
        ypos = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width()/2,
            ypos + (12 if val >= 0 else -12),
            f"€{int(round(val)):+}k",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=12.5,
            weight="bold",
            color=DARK_GREY
        )

    ax.legend(loc="upper right", fontsize=14.5)
    add_footer(fig, "Δ Margin = scenario margin − S1 margin.")

    files.append(savefig(fig, out_dir, "05_delta_margin_comparison"))
    

    # 06 NPV and payback analysis
    fig = setup_canvas("Investment Analysis: NPV and Discounted Payback")
    ax1 = fig.add_axes([0.06, 0.20, 0.48, 0.58])
    ax2 = fig.add_axes([0.58, 0.20, 0.36, 0.58])
    npv_vals = investable.set_index("ID")["NPV_kEUR"].reindex(investable["ID"])
    colors = [GREEN if v > 0 else RED for v in npv_vals]
    x1 = np.arange(len(investable))
    bars = ax1.bar(x1, npv_vals.values, color=colors, width=0.62)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(scenario_ticklabels(investable, width=14), fontsize=11.5)
    ax1.axhline(0, color=DARK_GREY, lw=1)
    ax1.set_ylabel("NPV (kEUR)", fontsize=15)
    ax1.grid(axis="y", alpha=0.16)
    for rect, sid, val in zip(bars, investable["ID"], npv_vals.values):
        pb = float(investable.set_index("ID").loc[sid, "PBP_yr"])
        label = f"€{int(round(val)):,}k"
        label += f"\n({pb:.1f} yr)" if np.isfinite(pb) else "\n(n/a)"
        ax1.text(rect.get_x()+rect.get_width()/2, val + (28 if val >= 0 else 50), label, ha="center", va="bottom", fontsize=14.8, weight="bold", color=DARK_GREY)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for sid in df["ID"]:
        x0 = float(df.set_index("ID").loc[sid, "CAPEX_kEUR"])
        y0 = float(df.set_index("ID").loc[sid, "NPV_kEUR"])
        ax2.scatter([x0], [y0], s=70 if sid in top4_by_npv else 52, color=SCENARIO_COLORS[sid], alpha=0.82, edgecolors="white", linewidths=0.8)
        annotate_scenario_point(ax2, sid, x0, y0, f"{sid}\n{wrap_label(id_to_name[sid], 14)}", fontsize=13.9)
    ax2.axhline(0, color=GREY, lw=1)
    ax2.axvline(0, color=GREY, lw=0.8)
    ax2.set_xlabel("CAPEX (kEUR)", fontsize=15)
    ax2.set_ylabel("NPV (kEUR)", fontsize=15)
    ax2.grid(alpha=0.14)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    add_footer(fig, f"NPV uses {int(fin['npv_horizon_years'])}-year discounted incremental cash flow vs S1. Payback is discounted payback over {int(fin['cashflow_years'])} years.")
    files.append(savefig(fig, out_dir, "06_npv_pbp_analysis"))

    # 07 standalone NPV view
    fig = setup_canvas("NPV View: All Scenarios and Cases", "CAPEX versus NPV using discounted incremental cash flow versus S1")
    ax = fig.add_axes([0.08, 0.24, 0.84, 0.54])
    for sid in df["ID"]:
        x0 = float(df.set_index("ID").loc[sid, "CAPEX_kEUR"])
        y0 = float(df.set_index("ID").loc[sid, "NPV_kEUR"])
        ax.scatter([x0], [y0], s=85 if sid in top4_by_npv else 62, color=SCENARIO_COLORS[sid], alpha=0.85, edgecolors="white", linewidths=0.8)
        annotate_scenario_point(ax, sid, x0, y0, f"{sid}\n{wrap_label(id_to_name[sid], 14)}", fontsize=14.2)
    ax.axhline(0, color=GREY, lw=1)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("CAPEX (kEUR)", fontsize=14)
    ax.set_ylabel("NPV (kEUR)", fontsize=14)
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "07_npv_vs_capex_all_scenarios"))

    def line_sensitivity_chart(filename, title, subtitle, xvals, value_func, xlabel, ylabel, ylim=None, direct_labels: bool = False, label_fontsize: float = 7.8):
        fig = setup_canvas(title, subtitle)
        ax = fig.add_axes([0.08, 0.30, 0.84 if not direct_labels else 0.78, 0.48])
        curve_map = {}
        for sid, ls in zip(df["ID"], LINE_STYLES):
            yvals = [value_func(sid, x) for x in xvals]
            curve_map[sid] = yvals
            ax.plot(xvals, yvals, linestyle=ls, linewidth=2.0, color=SCENARIO_COLORS[sid], label=scenario_legend_label(sid, id_to_name[sid]))
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ymin, ymax = ax.get_ylim()
        if ymin < 0 < ymax:
            ax.axhspan(ymin, 0, facecolor="#FDECEC", alpha=0.55, zorder=0)
        ax.axhline(0, color=GREY, lw=1.0, alpha=0.9)
        ax.grid(alpha=0.16)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if direct_labels:
            add_end_of_line_labels(ax, xvals, curve_map, {sid: sid for sid in df["ID"]}, SCENARIO_COLORS, fontsize=label_fontsize)
        else:
            add_spread_legend(ax, len(df["ID"]), yshift=-0.33, fontsize=13.9)
        add_footer(fig)
        return savefig(fig, out_dir, filename)

    capex_mults = np.linspace(0.7, 1.3, 13)
    biomass_prices = np.linspace(float(fin["biomass_price_eur_per_t"]) * 0.7, float(fin["biomass_price_eur_per_t"]) * 1.3, 13)
    elec_prices = np.linspace(float(fin["electricity_price_eur_per_mwh"]) * 0.7, float(fin["electricity_price_eur_per_mwh"]) * 1.3, 13)

    files.append(line_sensitivity_chart("08_sensitivity_capex_margin_all_options", "CAPEX Sensitivity: Investment-adjusted Annual Value", "CAPEX does not change operating margin directly; chart shows annual value after CAPEX burden for all options", capex_mults, lambda sid, x: annualized_value_series(df, fin, om_cfg, capex_mult=x, ids=[sid]).iloc[0], "CAPEX multiplier (x)", "Investment-adjusted annual value (kEUR/yr)"))
    files.append(line_sensitivity_chart("09_sensitivity_biomass_margin_all_options", "Biomass Price Sensitivity: Margin", "All scenarios and cases", biomass_prices, lambda sid, x: annual_margin_series(df, fin, om_cfg, biomass_price=x, ids=[sid]).iloc[0], "Biomass price (EUR/t)", "Margin (kEUR/yr)"))
    files.append(line_sensitivity_chart("10_sensitivity_electricity_margin_all_options", "Electricity Price Sensitivity: Margin", "All scenarios and cases", elec_prices, lambda sid, x: annual_margin_series(df, fin, om_cfg, elec_price=x, ids=[sid]).iloc[0], "Electricity price (EUR/MWh)", "Margin (kEUR/yr)"))
    files.append(line_sensitivity_chart("11_sensitivity_capex_npv_all_options", "CAPEX Sensitivity: NPV", "NPV versus baseline for all options", capex_mults, lambda sid, x: npv_series(df, fin, om_cfg, capex_mult=x, ids=[sid]).iloc[0], "CAPEX multiplier (x)", "NPV (kEUR)"))
    files.append(line_sensitivity_chart("12_sensitivity_biomass_npv_all_options", "Biomass Price Sensitivity: NPV", "NPV versus baseline for all options", biomass_prices, lambda sid, x: npv_series(df, fin, om_cfg, biomass_price=x, ids=[sid]).iloc[0], "Biomass price (EUR/t)", "NPV (kEUR)"))
    files.append(line_sensitivity_chart("13_sensitivity_electricity_npv_all_options", "Electricity Price Sensitivity: NPV", "NPV versus baseline for all options", elec_prices, lambda sid, x: npv_series(df, fin, om_cfg, elec_price=x, ids=[sid]).iloc[0], "Electricity price (EUR/MWh)", "NPV (kEUR)", direct_labels=True, label_fontsize=13.9))
    files.append(line_sensitivity_chart("14_sensitivity_capex_pbp_all_options", "CAPEX Sensitivity: Discounted Payback", f"Any option clipped at the upper boundary does not achieve discounted payback within {int(fin['cashflow_years'])} years", capex_mults, lambda sid, x: discounted_payback_series(df, fin, om_cfg, capex_mult=x, ids=[sid]).iloc[0] if np.isfinite(discounted_payback_series(df, fin, om_cfg, capex_mult=x, ids=[sid]).iloc[0]) else float(fin["cashflow_years"]), "CAPEX multiplier (x)", f"Discounted payback (years; clipped at {int(fin['cashflow_years'])} if not recovered)", ylim=(0, float(fin["cashflow_years"])), direct_labels=True, label_fontsize=13.8))
    files.append(line_sensitivity_chart("15_sensitivity_biomass_pbp_all_options", "Biomass Price Sensitivity: Discounted Payback", f"Any option clipped at the upper boundary does not achieve discounted payback within {int(fin['cashflow_years'])} years", biomass_prices, lambda sid, x: discounted_payback_series(df, fin, om_cfg, biomass_price=x, ids=[sid]).iloc[0] if np.isfinite(discounted_payback_series(df, fin, om_cfg, biomass_price=x, ids=[sid]).iloc[0]) else float(fin["cashflow_years"]), "Biomass price (EUR/t)", f"Discounted payback (years; clipped at {int(fin['cashflow_years'])} if not recovered)", ylim=(0, float(fin["cashflow_years"])), direct_labels=True, label_fontsize=13.8))
    files.append(line_sensitivity_chart("16_sensitivity_electricity_pbp_all_options", "Electricity Price Sensitivity: Discounted Payback", f"Any option clipped at the upper boundary does not achieve discounted payback within {int(fin['cashflow_years'])} years", elec_prices, lambda sid, x: discounted_payback_series(df, fin, om_cfg, elec_price=x, ids=[sid]).iloc[0] if np.isfinite(discounted_payback_series(df, fin, om_cfg, elec_price=x, ids=[sid]).iloc[0]) else float(fin["cashflow_years"]), "Electricity price (EUR/MWh)", f"Discounted payback (years; clipped at {int(fin['cashflow_years'])} if not recovered)", ylim=(0, float(fin["cashflow_years"])), direct_labels=True, label_fontsize=13.8))

    # 17 small multiples cashflow
    fig = setup_canvas("20-year Cash Flow Diagram: All Options", "Cumulative discounted incremental cash flow versus baseline (S1)", figsize=(16, 10))
    nrows, ncols = 3, 4
    for idx, sid in enumerate(df["ID"]):
        ax = fig.add_axes([0.05 + (idx % ncols) * 0.23, 0.66 - (idx // ncols) * 0.20, 0.19, 0.135])
        cft = incremental_cashflow_table(df, sid, fin, om_cfg, years=int(fin["cashflow_years"]))
        ax.plot(cft["Year"], cft["CumDiscountedIncrementalFCF_kEUR"], color=SCENARIO_COLORS[sid], linewidth=2.0)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.12)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if idx % ncols == 0:
            ax.set_ylabel("kEUR", fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Year", fontsize=10)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "17_cashflow_diagram_all_options_20yr"))

    # 18 all options one chart
    fig = setup_canvas("20-year Cash Flow: All Options in One Chart", "Cumulative discounted incremental cash flow versus baseline (S1)")
    ax = fig.add_axes([0.08, 0.30, 0.84, 0.48])
    for sid, ls in zip(df["ID"], LINE_STYLES):
        cft = incremental_cashflow_table(df, sid, fin, om_cfg, years=int(fin["cashflow_years"]))
        ax.plot(cft["Year"], cft["CumDiscountedIncrementalFCF_kEUR"], linestyle=ls, linewidth=2.2, color=SCENARIO_COLORS[sid], label=scenario_legend_label(sid, id_to_name[sid]))
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Cumulative discounted incremental cash flow (kEUR)", fontsize=14)
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_spread_legend(ax, len(df["ID"]), yshift=-0.33, fontsize=13.9)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "18_cashflow_all_options_one_chart"))

    # 19 top4 cashflow
    fig = setup_canvas("20-year Cash Flow: Best Four Options", f"Top 4 by NPV: {', '.join(top4_by_npv)}")
    ax = fig.add_axes([0.08, 0.26, 0.84, 0.52])
    for sid in top4_by_npv:
        cft = incremental_cashflow_table(df, sid, fin, om_cfg, years=int(fin["cashflow_years"]))
        ax.plot(cft["Year"], cft["CumDiscountedIncrementalFCF_kEUR"], linewidth=2.4, color=SCENARIO_COLORS[sid], label=f"{sid}: {id_to_name[sid]}")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Cumulative discounted incremental cash flow (kEUR)", fontsize=14)
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    add_spread_legend(ax, len(top4_by_npv), yshift=-0.28, fontsize=14.4)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "19_cashflow_top4_one_chart"))

    # Heatmaps
    elec_grid = np.linspace(float(fin["electricity_price_eur_per_mwh"]) * 0.7, float(fin["electricity_price_eur_per_mwh"]) * 1.3, 10)
    bio_grid = np.linspace(float(fin["biomass_price_eur_per_t"]) * 0.7, float(fin["biomass_price_eur_per_t"]) * 1.3, 10)

    def heatmap_grid(metric: str, sid: str) -> np.ndarray:
        arr = np.zeros((len(bio_grid), len(elec_grid)))
        for i, bp in enumerate(bio_grid):
            for j, ep in enumerate(elec_grid):
                if metric == "npv":
                    arr[i, j] = npv_series(df, fin, om_cfg, elec_price=ep, biomass_price=bp, ids=[sid]).iloc[0]
                elif metric == "margin":
                    arr[i, j] = annual_margin_series(df, fin, om_cfg, elec_price=ep, biomass_price=bp, ids=[sid]).iloc[0]
                elif metric == "pbp":
                    val = discounted_payback_series(df, fin, om_cfg, elec_price=ep, biomass_price=bp, ids=[sid]).iloc[0]
                    arr[i, j] = float(fin["cashflow_years"]) if not np.isfinite(val) else val
                else:
                    raise ValueError(metric)
        return arr

    def four_panel_heatmap(filename: str, title: str, metric: str, cbar_label: str):
        fig = setup_canvas(title, f"Electricity and biomass price sensitivity for top 4 options: {', '.join(top4_by_npv)}")
        for idx, sid in enumerate(top4_by_npv):
            left = 0.06 + (idx % 2) * 0.44
            bottom = 0.52 - (idx // 2) * 0.34
            ax = fig.add_axes([left, bottom, 0.34, 0.22])
            arr = heatmap_grid(metric, sid)
            im = ax.imshow(arr, aspect="auto", origin="lower")
            ax.set_xticks(range(len(elec_grid)))
            ax.set_xticklabels([f"{x:.0f}" for x in elec_grid], rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(bio_grid)))
            ax.set_yticklabels([f"{x:.0f}" for x in bio_grid], fontsize=9)
            ax.set_xlabel("Electricity price (€/MWh)", fontsize=10)
            ax.set_ylabel("Biomass price (€/t)", fontsize=10)
            cax = fig.add_axes([left + 0.35, bottom, 0.015, 0.22])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=8)
            cb.set_label(cbar_label, fontsize=10)
        add_footer(fig)
        return savefig(fig, out_dir, filename)

    files.append(four_panel_heatmap("20_heatmap_electricity_biomass_npv_top4", "Heatmap: Electricity and Biomass Sensitivity on NPV", "npv", "NPV (kEUR)"))
    files.append(four_panel_heatmap("21_heatmap_electricity_biomass_margin_top4", "Heatmap: Electricity and Biomass Sensitivity on Margin", "margin", "Margin (kEUR/yr)"))
    files.append(four_panel_heatmap("22_heatmap_electricity_biomass_pbp_top4", "Heatmap: Electricity and Biomass Sensitivity on Discounted Payback", "pbp", "Years"))

    # 23 revenue change
    fig = setup_canvas("Revenue Change Across All Studied Options", "Change in annual revenue versus baseline (electricity + heat revenue only)")
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.62])
    x = np.arange(len(df))
    bars = ax.bar(x, df["RevenueChange_kEUR"], color=[SCENARIO_COLORS[s] for s in df["ID"]], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ticklabels(df), fontsize=11)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylabel("Revenue change vs S1 (kEUR/yr)", fontsize=14)
    format_axes(ax)
    for rect, val in zip(bars, df["RevenueChange_kEUR"]):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + (6 if val >= 0 else -12), f"€{int(round(val))}k", ha="center", va="bottom" if val >= 0 else "top", fontsize=14, weight="bold", color=DARK_GREY)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "23_revenue_change_all_options"))

    # 24 Step-by-Step Pathway to Maximum Power
    pathway = df.set_index("ID").reindex(PATHWAY_IDS).reset_index()
    fig = setup_canvas("Step-by-Step Pathway to Maximum Power")
    ax = fig.add_axes([0.08, 0.16, 0.84, 0.68])
    x = np.arange(len(pathway))
    bar_colors = [SCENARIO_COLORS[sid] for sid in pathway["ID"]]
    bars = ax.bar(x, pathway["Export_kW"], color=bar_colors, width=0.54, edgecolor="white")
    ax.set_xticks(x)
    short_labels = {"S1": "Current", "S2": "Optimised", "S6": "Filter", "C1": "Filt.+LP", "S7": "Full", "C2": "Full+LP"}
    ax.set_xticklabels([f"{sid}\n{short_labels.get(sid, sid)}" for sid in pathway["ID"]], fontsize=13)
    tlim = float(cfg["technical_assumptions"]["transformer_limit_kW"])
    ax.axhline(tlim, color=LIGHT_RED, linestyle="--", linewidth=1.5)
    ax.text(len(pathway)-0.4, tlim+18, "Transformer limit", color=LIGHT_RED, fontsize=15, ha="right", weight="bold")
    ax.text(0.9, tlim+130, "Operational", color=GREY, fontsize=13, ha="center")
    ax.text(4.0, tlim+130, "Capital Investment", color=GREY, fontsize=13, ha="center")
    ax.axvline(1.5, color=LIGHT_GREY, lw=1.0, alpha=0.7)
    ax.axvline(3.5, color=LIGHT_GREY, lw=1.0, alpha=0.7)
    ax.set_ylabel("Power (kW)", fontsize=15)
    ax.grid(axis="y", alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    prev = None
    for i, (rect, sid, val, cap) in enumerate(zip(bars, pathway["ID"], pathway["Export_kW"], pathway["CAPEX_kEUR"])):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()-32, f"{int(round(val)):,.0f} kW", ha="center", va="top", fontsize=14, color="white", weight="bold")
        if cap > 0:
            ax.text(rect.get_x()+rect.get_width()/2, 40, f"€{int(round(cap))}k", ha="center", va="bottom", fontsize=13, color=GREY, weight="bold")
        if prev is not None:
            delta = val - prev
            ax.annotate(f"+{int(round(delta))}", xy=(i, val+10), xytext=(i-0.18, val+90), textcoords="data", fontsize=13, color=GREEN, weight="bold", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
        prev = val
    files.append(savefig(fig, out_dir, "24_step_by_step_pathway_to_maximum_power"))

    # 25 Phased Implementation Timeline
    fig = setup_canvas("Phased Implementation Timeline")
    ax = fig.add_axes([0.08, 0.12, 0.76, 0.80])
    ax2 = ax.twinx()

    # ── bars: stacked top-to-bottom, Phase 1 at top, Phase 4 at bottom ──────
    # We want them visible across the FULL height without clipping.
    # Use fixed row centres at y = 3.5, 2.5, 1.5, 0.5  (height 0.70)
    # ylim = [-0.1, 4.6] gives equal padding top and bottom
    y_centres = [3.5, 2.5, 1.5, 0.5]
    height = 0.70
    cum_margin = 0
    xmids = [0]
    ycums = [0]
    for idx, phase in enumerate(DEFAULT_PHASE_TIMELINE):
        start_m, end_m = phase["months"]
        width = end_m - start_m
        yc = y_centres[idx]
        ybot = yc - height / 2
        ax.broken_barh([(start_m, width)], (ybot, height),
                       facecolors=phase["color"], edgecolors="white",
                       linewidth=1.5, alpha=0.94)
        dm = float(phase["delta_margin_kEUR"])
        sign = "+" if dm >= 0 else "−"
        dm_str = f"{sign}€{abs(int(round(dm)))}k/yr"
        label = f"{phase['label']}\n€{int(round(phase['capex_kEUR']))}k → {dm_str}"
        ax.text(start_m + width / 2, yc, label,
                ha="center", va="center", fontsize=12.5,
                color="white", weight="bold")
        cum_margin += dm
        xmids.append(end_m)
        ycums.append(cum_margin)

    ax.set_xlim(0, 26)
    ax.set_ylim(-0.1, 4.1)          # exactly fits y_centres 0.5–3.5 ± 0.35
    ax.set_xlabel("Month", fontsize=14)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── cumulative line on right axis ─────────────────────────────────────
    # Phase 4 end point is the highest; line must reach near top of chart.
    # Keep ax2 ylim symmetric so the line occupies the full vertical space.
    margin_vals = [0] + [
        sum(float(p["delta_margin_kEUR"]) for p in DEFAULT_PHASE_TIMELINE[:i+1])
        for i in range(len(DEFAULT_PHASE_TIMELINE))
    ]
    y_lo = min(margin_vals) - 25
    y_hi = max(margin_vals) + 60
    ax2.plot(xmids, ycums, color="#C92D23", marker="o",
             linewidth=2.2, markersize=7, zorder=5)
    ax2.axhline(0, color="#C92D23", linewidth=0.8, linestyle=":", alpha=0.45)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_ylabel("Cumul. Δ Margin (kEUR/yr)", fontsize=14, color="#C92D23")
    ax2.tick_params(axis="y", colors="#C92D23", labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    files.append(savefig(fig, out_dir, "25_phased_implementation_timeline"))

    # 26 LP expander power output vs flowrate
    fig = setup_canvas("LP Expander: Power Output vs Flow Rate")
    ax = fig.add_axes([0.10, 0.18, 0.82, 0.64])
    eta_ref = float(cfg["technical_assumptions"].get("eta_exp", 0.75))
    try:
        s8_row = df.set_index("ID").loc["S8"]
        base_specific_kw_per_tph = float(s8_row["Expander_kW"]) / max(float(s8_row["Extraction_tph"]), 0.1)
    except Exception:
        base_specific_kw_per_tph = 67.0
    flow = np.linspace(0.5, 8.0, 100)
    screw_kw = flow * base_specific_kw_per_tph
    micro_kw = screw_kw * (0.80 / max(eta_ref, 1e-6))
    ax.plot(flow, screw_kw, color=BLUE, linewidth=2.0, label=f"Screw expander (η={eta_ref:.0%})")
    ax.plot(flow, micro_kw, color=GREEN, linewidth=1.6, linestyle="--", label="Micro turbine (η=80%)")
    marker_flows = [2.0, 3.5, 5.0]
    marker_labels = ["Reduced", "Current", "High"]
    for f, lbl in zip(marker_flows, marker_labels):
        y = f * base_specific_kw_per_tph
        ax.scatter([f], [y], color=AMBER, s=34, zorder=3)
        ax.annotate(f"{lbl}; {f:.1f} t/h\n{int(round(y))} kW", xy=(f, y), xytext=(6, 10), textcoords="offset points", fontsize=13, color=DARK_GREY, arrowprops=dict(arrowstyle="-", color=LIGHT_GREY, lw=0.8))
    ax.axhline(500, color=LIGHT_GREY, linestyle=":", linewidth=1.0)
    ax.text(flow[-1], 505, "500-kW target", color=GREY, fontsize=14.9, ha="right")
    ax.set_xlabel("Flow Through Expander (t/h)", fontsize=14)
    ax.set_ylabel("Output (kWe)", fontsize=14)
    ax.set_xlim(0, 8.1)
    ax.set_ylim(0, max(600, float(micro_kw.max()) + 20))
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=14.5, frameon=True)
    files.append(savefig(fig, out_dir, "26_lp_expander_power_output_vs_flowrate"))

    # 27 Incremental NPV bar chart for all investable options
    fig = setup_canvas("Incremental NPV: All Studied Options")
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.66])
    npv_vals = investable.set_index("ID")["NPV_kEUR"].reindex(investable["ID"])
    x = np.arange(len(investable))
    colors = [GREEN if v > 0 else RED for v in npv_vals.values]
    bars = ax.bar(x, npv_vals.values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ticklabels(investable, width=14), fontsize=11.5)
    ax.axhline(0, color=DARK_GREY, lw=1)
    ax.set_ylabel("NPV (kEUR)", fontsize=14)
    ax.grid(axis="y", alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for rect, sid, val in zip(bars, investable["ID"], npv_vals.values):
        pb = float(investable.set_index("ID").loc[sid, "PBP_yr"])
        txt = f"€{int(round(val)):,}k" + (f"\n({pb:.1f} yr)" if np.isfinite(pb) else "\n(n/a)")
        if val >= 0:
            ax.text(rect.get_x()+rect.get_width()/2, val + 28, txt, ha="center", va="bottom", fontsize=14.8, weight="bold", color=DARK_GREY)
        else:
            ax.text(rect.get_x()+rect.get_width()/2, val + 45, txt, ha="center", va="bottom", fontsize=14.8, weight="bold", color=BLUE)
    files.append(savefig(fig, out_dir, "27_incremental_npv_all_options"))

    # 28 Margin evolution over years (with escalation)
    fig = setup_canvas("Annual Margin Evolution Over Time",
                       f"Elec +{float(fin.get('electricity_escalation',0))*100:.1f}%/yr, Bio +{float(fin.get('biomass_escalation',0))*100:.1f}%/yr, O&M +{float(fin.get('om_escalation',0))*100:.1f}%/yr")
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.62])
    cf_years = int(fin.get("cashflow_years", 20))
    show_ids = ["S1", "S6", "S7", "S8", "C1", "C2"]
    for sid in show_ids:
        if sid not in df["ID"].values:
            continue
        row = df.set_index("ID").loc[sid]
        yearly = []
        for y in range(1, cf_years + 1):
            rc = annual_revenue_costs_kEUR(row, fin, om_cfg, year=y)
            m = rc["TotalRevenue_kEUR"] - rc["BiomassCost_kEUR"] - rc["TotalOMSharpened_kEUR"]
            yearly.append(m)
        ax.plot(range(1, cf_years + 1), yearly,
                color=SCENARIO_COLORS.get(sid, GREY), linewidth=2.2,
                label=f"{sid}: {id_to_name.get(sid, sid)}")
    ax.axhline(0, color=DARK_GREY, linewidth=0.8)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Annual Margin (kEUR/yr)", fontsize=14)
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=11, frameon=True)
    add_footer(fig)
    files.append(savefig(fig, out_dir, "28_margin_evolution_over_years"))

    manifest_rows = [{"filename": p.name, "bytes": p.stat().st_size} for p in sorted(files)]
    pd.DataFrame(manifest_rows).to_csv(out_dir / "figure_manifest.csv", index=False)

    econ_cols = [
        "ID", "Scenario", "CAPEX_kEUR", "Export_kW", "AnnualMWh", "HeatMWhth", "Biomass_tpy",
        "TrueFixedBase_kEUR", "RuntimeOM_kEUR", "ThroughputOM_kEUR", "StartStopOM_kEUR",
        "ScenarioIncrementalOM_kEUR", "TotalOMSharpened_kEUR", "TotalRevenue_kEUR", "RevenueChange_kEUR",
        "Margin_kEUR", "DeltaMargin_kEUR", "NPV_kEUR", "PBP_yr"
    ]
    df[econ_cols].to_csv(out_dir / "scenario_economic_table.csv", index=False)
    df.to_csv(out_dir / "scenario_snapshot_used.csv", index=False)

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files):
            zf.write(p, arcname=p.name)
        zf.write(out_dir / "figure_manifest.csv", arcname="figure_manifest.csv")
        zf.write(out_dir / "scenario_economic_table.csv", arcname="scenario_economic_table.csv")
        zf.write(out_dir / "scenario_snapshot_used.csv", arcname="scenario_snapshot_used.csv")
    return files


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CHP Steenwijk 23 techno-economic figures for local use."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file with technical, financial, O&M, and scenario overrides."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIR,
        help="Output folder for PNGs and CSVs."
    )
    parser.add_argument(
        "--write-config-template",
        action="store_true",
        help="Write a config template JSON and exit."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_config_template:
        path = args.config or (Path(__file__).resolve().parent / "chp_local_config_template.json")
        write_config_template(path)
        print(f"Wrote config template: {path}")
        return

    cfg = load_config(args.config)
    df = build_scenario_dataframe_from_engine(cfg)
    df = apply_scenario_overrides(df, cfg)
    files = generate_figures(df, cfg, args.outdir)
    print(f"Wrote {len(files)} PNG figures to {args.outdir}")
    print(f"ZIP: {args.outdir}.zip")


if __name__ == "__main__":
    main()
