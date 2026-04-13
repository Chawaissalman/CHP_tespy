"""
CHP Steenwijk — TESPy Plant Performance App (v3: Editable + Economics)
Run with:
    pip install streamlit tespy CoolProp plotly pandas numpy matplotlib
    streamlit run app_tespy_v3.py

All v2 features (editable scenario tabs, methodology, S8 detail) PLUS
full economics tab with editable financial inputs, all 27 figures from
the techno-economic report, and ZIP download.
"""
from __future__ import annotations
import copy
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from pathlib import Path

from thermo_engine_tespy import analyse_cycle, compute_scenario, SCENARIO_DEFS
from economics_figures import (
    build_scenario_dataframe_from_engine, enrich_dataframe, generate_figures,
    DEFAULT_FINANCIALS, DEFAULT_OM, DEFAULT_CONFIG,
    annual_margin_series, annual_revenue_costs_kEUR, revenue_series, delta_margin_series,
    npv_series, discounted_payback_series, incremental_cashflow_table,
    om_components_baseyear,
    SCENARIO_COLORS, SORT_MAP, FILTER_UPGRADE_IDS, FAN_UPGRADE_IDS, LP_EXPANDER_IDS,
    FG_NM3PH_DEFAULT, BAG_UTIL_DEFAULT, FAN_UTIL_DEFAULT,
)

st.set_page_config(page_title="CHP Steenwijk — TESPy v2", layout="wide", page_icon="🔥")

# ═══════════════════════════════════════════════
# SIDEBAR: Global cycle parameters
# ═══════════════════════════════════════════════
st.sidebar.title("🔧 Global Cycle Parameters")
st.sidebar.caption("These apply to ALL scenarios as the base thermodynamic cycle.")
st.sidebar.header("Steam Cycle")
P_live = st.sidebar.number_input("Live steam pressure (bar)", value=50.0, min_value=20.0, max_value=100.0, step=1.0)
T_live = st.sidebar.number_input("Live steam temperature (°C)", value=444.0, min_value=350.0, max_value=550.0, step=1.0)
P_ext = st.sidebar.number_input("Extraction pressure (bar)", value=2.5, min_value=0.5, max_value=10.0, step=0.1)
P_cond_ref = st.sidebar.number_input("Reference condenser pressure (bar)", value=0.20, min_value=0.03, max_value=1.0, step=0.01, format="%.2f")
T_fw = st.sidebar.number_input("Feedwater temperature (°C)", value=130.0, min_value=50.0, max_value=200.0, step=5.0)
st.sidebar.header("Design Efficiencies")
eta_HP = st.sidebar.slider("η HP turbine (design)", 0.60, 0.95, 0.80, 0.01)
eta_LP = st.sidebar.slider("η LP turbine (design)", 0.60, 0.95, 0.78, 0.01)
eta_exp = st.sidebar.slider("η LP expander (design)", 0.50, 0.90, 0.75, 0.01)
eta_gen = st.sidebar.slider("η generator + mechanical", 0.85, 0.98, 0.94, 0.01)
st.sidebar.header("Plant Limits & Fuel")
trafo = st.sidebar.number_input("Transformer limit (kW)", value=2520.0, step=10.0)
LHV = st.sidebar.number_input("Biomass LHV (MJ/kg)", value=10.5, min_value=5.0, max_value=20.0, step=0.5)
eta_boiler = st.sidebar.slider("Boiler efficiency", 0.70, 0.95, 0.85, 0.01)
biomass_ratio_override = st.sidebar.number_input(
    "Biomass ratio (t_bio/t_steam)", value=0.388, min_value=0.0, max_value=1.0,
    step=0.001, format="%.3f",
    help="Historical: 0.388. Thermodynamic: ~0.31. Set to 0 to use thermodynamic calc.")

# ═══════════════════════════════════════════════
# BASE CYCLE (shared by all scenarios)
# ═══════════════════════════════════════════════
base_cycle = analyse_cycle(P_live_bar=P_live, T_live_C=T_live, P_extraction_bar=P_ext,
    P_condenser_bar=P_cond_ref, T_feedwater_C=T_fw,
    eta_HP=eta_HP, eta_LP=eta_LP, eta_exp=eta_exp, eta_gen=eta_gen)

# ═══════════════════════════════════════════════
# INITIALISE SESSION STATE with defaults
# ═══════════════════════════════════════════════
if "scenario_params" not in st.session_state:
    st.session_state.scenario_params = {}
    for sd in SCENARIO_DEFS:
        st.session_state.scenario_params[sd["sid"]] = {
            "steam_tph": sd["steam_tph"],
            "extraction_tph": sd["extraction_tph"],
            "availability_pct": sd["availability_pct"],
            "capex_kEUR": float(sd["capex_kEUR"]),
            "has_lp_expander": sd["has_lp_expander"],
            "condenser_pressure_bar": sd["condenser_pressure_bar"],
            "summer_full_condenser_fans": sd["summer_full_condenser_fans"],
            "fg_fan_limit_tph": sd["fg_fan_limit_tph"],
            "fg_fan_motor_kW": sd["fg_fan_motor_kW"],
        }

# ═══════════════════════════════════════════════
# SCENARIO METHODOLOGY TEXT
# ═══════════════════════════════════════════════
SCENARIO_METHODOLOGY = {
    "S1": """**What is happening:** S1 is the measured SCADA baseline — the plant as it operates today with no modifications.
The boiler produces 9.7 t/h of live steam at 50 bar, 444°C. All steam passes through the HP turbine expanding to
2.5 bar extraction pressure. At the extraction point, 3.5 t/h is diverted to the factory heat system, and the
remaining 6.2 t/h continues through the LP turbine to the condenser at 0.20 bar.

**Why these numbers:** The 9.7 t/h is constrained by the current flue gas system — the bag filter operates at 91%
and the FG fan at 84%. The extraction rate of 3.5 t/h matches the factory's current heat demand.

**Part-load effects:** At 9.7/13.0 = 74.6% of design steam flow, the HP turbine efficiency degrades from the design
value. The LP section at 6.2/10.0 = 62% of design also sees a penalty.""",

    "S2": """**What is happening:** S2 optimises combustion and extraction without major capital investment.
FGR is retuned and excess O₂ reduced from 5.1% to 4.0%, improving boiler efficiency and allowing ~0.8 t/h more steam.
Extraction is reduced from 3.5 to 3.0 t/h, sending more steam through the LP turbine.

**Why it works:** Lower O₂ means less excess air heated uselessly. Extra steam → more LP power. Reducing extraction
loses some heat revenue but gains more electrical revenue. CAPEX only €20k.""",

    "S3": """**What is happening:** S3 maximises electrical output by sending ALL steam through the LP turbine — zero
extraction. Factory heat demand met by propane backup instead.

**Trade-off:** Zero heat revenue from steam, but maximum possible electrical output at current boiler capacity.""",

    "S4": """**What is happening:** S4 represents summer operation when ambient temperature exceeds 25°C. Condenser
backpressure rises from 0.20 to 0.28 bar. All 5 condenser fans run at full speed (90 kW). Steam limited to 10.4 t/h.

**Key difference:** This is the only scenario with non-standard condenser pressure and summer fan mode. TESPy
re-solves the cycle at 0.28 bar, giving different LP/expander enthalpy drops.""",

    "S5": """**What is happening:** S5 implements price-responsive dispatch — DCS connected to EPEX NL spot market.
When prices are high, extraction is minimised. Uses annual weighted averages (10.5 t/h steam, 2.0 extraction, 8.5
condensing). €80k for DCS upgrade + EPEX API.""",

    "S6": """**What is happening:** Bag filter expanded from ~31,000 to 40,000 Nm³/h, removing the first bottleneck
and allowing 12 t/h. The FG fan (110 kW) becomes the next limit at ~103% utilisation.""",

    "S7": """**What is happening:** Full debottleneck — BOTH bag filter AND FG fan upgraded to 40,000 Nm³/h. FG fan
motor goes from 110→150 kW. Unlocks full 13 t/h boiler design capacity.""",

    "S8": """**What is happening:** Screw expander on extraction steam. Instead of throttling 3.5 t/h from 2.5→atm
through a valve (wasting pressure energy), the expander converts this to electricity.
**No extra fuel** — same boiler load as S1, pure energy recovery.""",

    "C1": """**What is happening:** Filter debottleneck (S6) + LP expander (S8) stacked. 12 t/h enabled by filter,
1 t/h extraction through expander, 11 t/h condensing. Extraction minimised to maximise LP power.""",

    "C2": """**What is happening:** Full debottleneck (S7) + LP expander (S8). Maximum power configuration at 13 t/h.
Pushes close to transformer limit of 2,520 kW. €700k total CAPEX.""",

    "C3": """**What is happening:** Combustion optimisation (S2) + smart dispatch (S5) combined. Lowest-cost investable
option at €100k. Purely operational improvements, no hardware changes to flue gas system.""",
}

# ═══════════════════════════════════════════════
# CREATE TABS
# ═══════════════════════════════════════════════
tab_names = (["📊 Overview & Comparison"]
    + [f"{'⚡' if 'S' in sd['sid'] else '🔗'} {sd['sid']}: {sd['name']}" for sd in SCENARIO_DEFS]
    + ["📐 Methodology", "💰 Economics", "🖼️ Figures Gallery"])
tabs = st.tabs(tab_names)

# ═══════════════════════════════════════════════
# COLLECT INPUTS FROM EACH SCENARIO TAB
# (We render input widgets in each tab first,
#  read from session_state, then compute below)
# ═══════════════════════════════════════════════
scenario_overrides = {}

for i, sd in enumerate(SCENARIO_DEFS):
    sid = sd["sid"]
    defaults = st.session_state.scenario_params[sid]

    with tabs[i + 1]:
        st.header(f"{sid}: {sd['name']}")
        st.markdown(f"*{sd['description']}*")

        # --- EDITABLE INPUTS ---
        st.markdown("### ⚙️ Scenario Parameters")
        st.caption("Adjust these to explore what-if variations. Changes update all results instantly.")

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            steam = st.number_input("Total steam (t/h)", value=defaults["steam_tph"],
                min_value=4.0, max_value=16.0, step=0.1, key=f"{sid}_steam", format="%.1f")
            extraction = st.number_input("Extraction (t/h)", value=defaults["extraction_tph"],
                min_value=0.0, max_value=steam, step=0.1, key=f"{sid}_ext", format="%.1f")
            condensing = round(steam - extraction, 2)
            st.metric("Condensing (t/h)", f"{condensing:.1f}", help="Auto-calculated: steam − extraction")

        with ic2:
            avail = st.number_input("Availability (%)", value=defaults["availability_pct"],
                min_value=50.0, max_value=100.0, step=0.5, key=f"{sid}_avail", format="%.1f")
            capex = st.number_input("CAPEX (kEUR)", value=defaults["capex_kEUR"],
                min_value=0.0, max_value=2000.0, step=10.0, key=f"{sid}_capex", format="%.0f")
            has_exp = st.checkbox("LP Expander installed", value=defaults["has_lp_expander"],
                key=f"{sid}_exp")

        with ic3:
            p_cond_scen = st.number_input("Condenser pressure (bar)", value=defaults["condenser_pressure_bar"],
                min_value=0.03, max_value=1.0, step=0.01, key=f"{sid}_pcond", format="%.2f")
            summer_fans = st.checkbox("Summer full condenser fans", value=defaults["summer_full_condenser_fans"],
                key=f"{sid}_summer")
            fg_limit = st.number_input("FG fan capacity (t/h)", value=defaults["fg_fan_limit_tph"],
                min_value=8.0, max_value=20.0, step=0.5, key=f"{sid}_fglim", format="%.1f")
            fg_motor = st.number_input("FG fan motor (kW)", value=defaults["fg_fan_motor_kW"],
                min_value=50.0, max_value=300.0, step=10.0, key=f"{sid}_fgmot", format="%.0f")

        scenario_overrides[sid] = dict(
            sid=sid, name=sd["name"], description=sd["description"],
            steam_tph=steam, extraction_tph=extraction, condensing_tph=condensing,
            availability_pct=avail, capex_kEUR=capex, has_lp_expander=has_exp,
            condenser_pressure_bar=p_cond_scen, summer_full_condenser_fans=summer_fans,
            fg_fan_limit_tph=fg_limit, fg_fan_motor_kW=fg_motor,
        )

# ═══════════════════════════════════════════════
# COMPUTE ALL SCENARIOS with current inputs
# ═══════════════════════════════════════════════
results = []
for sd in SCENARIO_DEFS:
    sid = sd["sid"]
    ov = scenario_overrides[sid]
    r = compute_scenario(
        cycle=base_cycle, transformer_limit_kW=trafo,
        LHV_MJkg=LHV, eta_boiler=eta_boiler,
        biomass_ratio_override=biomass_ratio_override, **ov)
    results.append(r)

baseline = results[0]

rows = []
for r in results:
    rows.append({
        "ID": r.sid, "Scenario": r.name,
        "Steam (t/h)": r.steam_tph, "Extraction (t/h)": r.extraction_tph,
        "Condensing (t/h)": r.condensing_tph,
        "LP Expander": "✓" if r.has_lp_expander else "—",
        "P_cond (bar)": r.condenser_pressure_bar,
        "P_HP (kW)": r.P_HP, "P_LP (kW)": r.P_LP, "P_Exp (kW)": r.P_exp,
        "P_Gross (kW)": r.P_gross, "P_Aux (kW)": r.P_aux,
        "P_Net (kW)": r.P_net, "P_Export (kW)": r.P_export,
        "Curtailed (kW)": r.P_curtailed,
        "ΔP vs S1 (kW)": round(r.P_export - baseline.P_export, 1),
        "Annual MWh Export": r.annual_MWh_export,
        "Avail (%)": r.availability_pct, "CAPEX (kEUR)": r.capex_kEUR,
        "η_HP eff (%)": r.eta_HP_eff*100, "η_LP eff (%)": r.eta_LP_eff*100,
        "η_Exp eff (%)": r.eta_exp_eff*100,
        "η_elec net (%)": r.elec_eff_pct_net, "η_CHP (%)": r.chp_eff_pct,
        "Heat (kW)": r.heat_output_kW,
        "Biomass (t/h)": r.biomass_tph, "Biomass (t/yr)": r.biomass_tpy,
    })
df = pd.DataFrame(rows)
curtailed_ids = df.loc[df["Curtailed (kW)"] > 0, "ID"].tolist()
curtailed_label = ", ".join(curtailed_ids) if curtailed_ids else "none"

# ═══════════════════════════════════════════════
# TAB 0: OVERVIEW (uses computed results)
# ═══════════════════════════════════════════════
with tabs[0]:
    st.title("CHP Steenwijk — TESPy Plant Performance (v2: Editable)")
    st.caption("Edit scenario parameters in each tab. Overview updates automatically.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Gross", f"{baseline.P_gross:.0f} kW", f"HP {baseline.P_HP:.0f} + LP {baseline.P_LP:.0f}")
    col2.metric("Baseline Aux", f"{baseline.P_aux:.0f} kW", f"Net {baseline.P_net:.0f} kW")
    col3.metric("Baseline Export", f"{baseline.P_export:.0f} kW", f"S1 @ {baseline.steam_tph} t/h")
    col4.metric("Curtailed", curtailed_label)
    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Bar(name='HP Turbine', x=df['ID'], y=df['P_HP (kW)'], marker_color='#0B6FA4'))
    fig.add_trace(go.Bar(name='LP Turbine', x=df['ID'], y=df['P_LP (kW)'], marker_color='#5F9BD1'))
    fig.add_trace(go.Bar(name='LP Expander', x=df['ID'], y=df['P_Exp (kW)'], marker_color='#D98C0B'))
    fig.add_trace(go.Bar(name='Auxiliaries', x=df['ID'], y=-df['P_Aux (kW)'], marker_color='rgba(80,80,80,0.55)'))
    fig.add_trace(go.Bar(name='Curtailed', x=df['ID'], y=-df['Curtailed (kW)'], marker_color='rgba(192,57,43,0.5)'))
    fig.add_hline(y=trafo, line_dash="dash", line_color="red", annotation_text=f"Transformer {trafo:.0f} kW")
    fig.update_layout(barmode='relative', title='Power Breakdown by Section',
                      yaxis_title='Power (kW)', height=470, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    deltas = df['ΔP vs S1 (kW)'].tolist()
    colors = ['#3C8D5A' if d >= 0 else '#C0392B' for d in deltas]
    fig2.add_trace(go.Bar(x=df['ID'], y=deltas, marker_color=colors,
                          text=[f"{d:+.0f}" for d in deltas], textposition='outside'))
    fig2.add_hline(y=0, line_color="black", line_width=0.5)
    fig2.update_layout(title='Export Power Δ vs S1 (kW)', yaxis_title='Δ Export (kW)', height=400, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Full Comparison Table")
    st.dataframe(df.style.format({
        'P_cond (bar)': '{:.2f}',
        'P_HP (kW)': '{:.0f}', 'P_LP (kW)': '{:.0f}', 'P_Exp (kW)': '{:.0f}',
        'P_Gross (kW)': '{:.0f}', 'P_Aux (kW)': '{:.0f}', 'P_Net (kW)': '{:.0f}',
        'P_Export (kW)': '{:.0f}', 'Curtailed (kW)': '{:.0f}',
        'ΔP vs S1 (kW)': '{:+.0f}', 'Annual MWh Export': '{:,.0f}',
        'η_HP eff (%)': '{:.1f}', 'η_LP eff (%)': '{:.1f}', 'η_Exp eff (%)': '{:.1f}',
        'η_elec net (%)': '{:.1f}', 'η_CHP (%)': '{:.1f}',
        'Heat (kW)': '{:.0f}', 'Biomass (t/h)': '{:.2f}', 'Biomass (t/yr)': '{:,.0f}',
    }).background_gradient(subset=['ΔP vs S1 (kW)'], cmap='RdYlGn'), use_container_width=True, height=470)

    st.markdown("### Efficiency Comparison")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='η_elec net', x=df['ID'], y=df['η_elec net (%)'], marker_color='#0B6FA4'))
    fig3.add_trace(go.Bar(name='η_CHP', x=df['ID'], y=df['η_CHP (%)'], marker_color='#D98C0B'))
    fig3.update_layout(barmode='group', yaxis_title='Efficiency (%)', height=400, template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════
# TABS 1-11: Scenario results (below the inputs)
# ═══════════════════════════════════════════════
for i, r in enumerate(results):
    with tabs[i + 1]:
        st.markdown("---")

        # KPI row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        delta = r.P_export - baseline.P_export
        col1.metric("P_export", f"{r.P_export:.0f} kW", f"{delta:+.0f} vs S1")
        col2.metric("P_gross", f"{r.P_gross:.0f} kW")
        col3.metric("Aux load", f"{r.P_aux:.0f} kW")
        col4.metric("Annual MWh", f"{r.annual_MWh_export:,}")
        col5.metric("η_elec, net", f"{r.elec_eff_pct_net:.1f}%")
        col6.metric("CAPEX", f"€{r.capex_kEUR:.0f}k")

        # Methodology
        st.markdown("### What Is Happening in This Scenario")
        meth = SCENARIO_METHODOLOGY.get(r.sid, "")
        if meth:
            st.markdown(meth)

        # Power calc
        c = r.cycle
        st.markdown("### Power Calculation (Step-by-Step)")
        st.markdown(f"""
**HP Section** — all {r.steam_tph} t/h through HP turbine (η_HP,eff = {r.eta_HP_eff:.1%}):
```
w_HP = h₁ − h₂ = {c.state1.h_kJkg:.1f} − {c.state2.h_kJkg:.1f} = {c.w_HP:.1f} kJ/kg
P_HP = {r.steam_tph} × {c.w_HP:.1f} × {eta_gen:.2f} / 3.6 = {r.P_HP:.1f} kW
```

**LP Section** — {r.condensing_tph} t/h through LP turbine (η_LP,eff = {r.eta_LP_eff:.1%}):
```
w_LP = h₂ − h₃ = {c.state2.h_kJkg:.1f} − {c.state3.h_kJkg:.1f} = {c.w_LP:.1f} kJ/kg
P_LP = {r.condensing_tph} × {c.w_LP:.1f} × {eta_gen:.2f} / 3.6 = {r.P_LP:.1f} kW
```
        """)

        if r.has_lp_expander:
            st.markdown(f"""
**LP Expander** — {r.extraction_tph} t/h through screw expander (η_exp,eff = {r.eta_exp_eff:.1%}):
```
w_exp = h₂ − h₃e = {c.state2.h_kJkg:.1f} − {c.state3e.h_kJkg:.1f} = {c.w_exp:.1f} kJ/kg
P_exp = {r.extraction_tph} × {c.w_exp:.1f} × {eta_gen:.2f} / 3.6 = {r.P_exp:.1f} kW
```
Useful heat from **post-expander** state:
```
q_heat = h₃e − h₄ = {c.state3e.h_kJkg:.1f} − {c.state4.h_kJkg:.1f} = {c.q_heat_post_expander:.1f} kJ/kg
Heat = {r.extraction_tph} × {c.q_heat_post_expander:.1f} / 3.6 = {r.heat_output_kW:.1f} kW
```
            """)
        else:
            if r.extraction_tph > 0:
                st.markdown(f"""
**Useful heat** — direct extraction:
```
q_heat = h₂ − h₄ = {c.state2.h_kJkg:.1f} − {c.state4.h_kJkg:.1f} = {c.q_heat_direct:.1f} kJ/kg
Heat = {r.extraction_tph} × {c.q_heat_direct:.1f} / 3.6 = {r.heat_output_kW:.1f} kW
```
                """)

        st.markdown(f"""
**Totals:**
```
P_gross  = {r.P_HP:.1f} + {r.P_LP:.1f} + {r.P_exp:.1f} = {r.P_gross:.1f} kW
P_aux    = {r.P_aux:.1f} kW
P_net    = {r.P_gross:.1f} − {r.P_aux:.1f} = {r.P_net:.1f} kW
P_export = min({r.P_net:.1f}, {trafo:.0f}) = {r.P_export:.1f} kW
```
        """)

        # Aux breakdown
        st.markdown("### Auxiliary Load Breakdown")
        aux_df = pd.DataFrame({
            "Component": ["Base misc", "Steam island", "Biomass handling", "FG fan",
                          "Condenser fans", "Heat system", "Expander skid"],
            "Load (kW)": [
                r.aux_breakdown.base_misc_kW, r.aux_breakdown.steam_island_kW,
                r.aux_breakdown.biomass_handling_kW, r.aux_breakdown.fg_fan_kW,
                r.aux_breakdown.condenser_fans_kW, r.aux_breakdown.heat_system_kW,
                r.aux_breakdown.expander_skid_kW],
            "How calculated": [
                "Fixed 115 kW",
                f"4 × {r.steam_tph} = {r.aux_breakdown.steam_island_kW:.1f}",
                f"6 × {r.biomass_tph:.2f} = {r.aux_breakdown.biomass_handling_kW:.1f}",
                f"{r.aux_breakdown.fg_fan_kW:.1f} (motor × load³)",
                f"{r.aux_breakdown.condenser_fans_kW:.1f}" + (" (summer: 90 kW)" if r.summer_full_condenser_fans else ""),
                f"15 + 3×{r.extraction_tph} = {r.aux_breakdown.heat_system_kW:.1f}",
                f"{'5 kW (installed)' if r.has_lp_expander else '0 (none)'}",
            ]
        })
        st.dataframe(aux_df, use_container_width=True, height=290)

        # Biomass
        h_gain = c.state1.h_kJkg - c.state4.h_kJkg
        st.markdown("### Biomass & Fuel")
        st.markdown(f"""
```
ratio    = {h_gain:.1f} / ({eta_boiler} × {LHV*1000:.0f}) = {r.biomass_ratio:.4f} t biomass/t steam
biomass  = {r.steam_tph} × {r.biomass_ratio:.4f} = {r.biomass_tph:.2f} t/h  ({r.biomass_tpy:,} t/yr)
Q_fuel   = {r.biomass_tph:.2f} × {LHV*1000:.0f} / 3.6 = {r.Q_fuel_kW:,.0f} kW
η_elec   = {r.P_export:.1f} / {r.Q_fuel_kW:,.0f} = {r.elec_eff_pct_net:.1f}%
η_CHP    = ({r.P_export:.1f} + {r.heat_output_kW:.0f}) / {r.Q_fuel_kW:,.0f} = {r.chp_eff_pct:.1f}%
```
        """)

        # Results summary
        st.markdown("### Results")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
| Metric | Value |
|---|---|
| Gross power | **{r.P_gross:.1f} kW** |
| Auxiliary load | **{r.P_aux:.1f} kW** |
| Net power | **{r.P_net:.1f} kW** |
| Export power | **{r.P_export:.1f} kW** |
| Curtailed | **{r.P_curtailed:.1f} kW** |
| Annual export | **{r.annual_MWh_export:,} MWh/yr** |
            """)
        with r2:
            st.markdown(f"""
| Metric | Value |
|---|---|
| η_elec gross | **{r.elec_eff_pct_gross:.1f}%** |
| η_elec net | **{r.elec_eff_pct_net:.1f}%** |
| η_CHP | **{r.chp_eff_pct:.1f}%** |
| Heat to factory | **{r.heat_output_kW:.0f} kW** |
| Biomass | **{r.biomass_tph:.2f} t/h** |
| Biomass annual | **{r.biomass_tpy:,} t/yr** |
            """)

        # Pie
        fig = go.Figure(data=[go.Pie(
            labels=['HP Turbine', 'LP Turbine', 'LP Expander', 'Auxiliaries', 'Curtailed'],
            values=[max(r.P_HP,0), max(r.P_LP,0), max(r.P_exp,0), max(r.P_aux,0), max(r.P_curtailed,0)],
            marker_colors=['#0B6FA4','#5F9BD1','#D98C0B','#7F8C8D','#C0392B'],
            hole=0.4, textinfo='label+value+percent')])
        fig.update_layout(title=f'{r.sid} Power Breakdown', height=360)
        st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════
        # S8 SPECIAL: Detailed expander analysis
        # ═══════════════════════════════════════
        if r.sid == "S8" and r.has_lp_expander:
            st.markdown("---")
            st.markdown("### LP Expander: Detailed Calculation at Each Operating Point")

            dh_isen = c.state2.h_kJkg - c.state3s.h_kJkg
            dh_screw = c.w_exp
            dh_micro = 0.80 * dh_isen

            st.markdown(f"""
```
Isentropic drop: h₂ − h₃s = {c.state2.h_kJkg:.1f} − {c.state3s.h_kJkg:.1f} = {dh_isen:.1f} kJ/kg
Screw (η={r.eta_exp_eff:.0%}):  Δh = {r.eta_exp_eff:.2f} × {dh_isen:.1f} = {dh_screw:.1f} kJ/kg
Micro  (η=80%):  Δh = 0.80 × {dh_isen:.1f} = {dh_micro:.1f} kJ/kg
```
            """)

            marker_flows = [2.0, 3.5, 5.0]
            marker_labels = ["Reduced", "Current", "High"]
            calc_rows = []
            for flow, label in zip(marker_flows, marker_labels):
                p_screw = flow * dh_screw * eta_gen / 3.6
                p_micro = flow * dh_micro * eta_gen / 3.6
                calc_rows.append({
                    "Point": f"{label}: {flow:.1f} t/h",
                    "Δh screw (kJ/kg)": f"{dh_screw:.1f}",
                    "P screw (kW)": f"{p_screw:.0f}",
                    "Calc screw": f"{flow} × {dh_screw:.1f} × {eta_gen:.2f} / 3.6",
                    "Δh micro (kJ/kg)": f"{dh_micro:.1f}",
                    "P micro (kW)": f"{p_micro:.0f}",
                    "Calc micro": f"{flow} × {dh_micro:.1f} × {eta_gen:.2f} / 3.6",
                })
            st.dataframe(pd.DataFrame(calc_rows), use_container_width=True)

            st.markdown(f"""
**Why screw expander over micro turbine?**
Micro turbine produces ~{(0.80/max(eta_exp,0.01) - 1)*100:.0f}% more power, but screw expanders:
- Tolerate wet steam (LP exhaust x ≈ {(c.state3e.x if c.state3e.x else 0.93):.3f})
- Lower maintenance, longer service intervals
- Better part-load at variable extraction rates
- Lower cost per kW at this scale (~300 kW class)
            """)

            flow_arr = np.linspace(0.3, 8.0, 100)
            screw_kw = flow_arr * dh_screw * eta_gen / 3.6
            micro_kw = flow_arr * dh_micro * eta_gen / 3.6
            fig_exp = go.Figure()
            fig_exp.add_trace(go.Scatter(x=flow_arr, y=screw_kw, mode='lines',
                name=f'Screw (η={eta_exp:.0%})', line=dict(color='#0B6FA4', width=2.5)))
            fig_exp.add_trace(go.Scatter(x=flow_arr, y=micro_kw, mode='lines',
                name='Micro turbine (η=80%)', line=dict(color='#3C8D5A', width=2, dash='dash')))
            for flow, label in zip(marker_flows, marker_labels):
                y = flow * dh_screw * eta_gen / 3.6
                fig_exp.add_trace(go.Scatter(x=[flow], y=[y], mode='markers+text',
                    marker=dict(color='#D98C0B', size=10),
                    text=[f"{label}: {flow} t/h<br>{y:.0f} kW"],
                    textposition='top center', textfont=dict(size=10), showlegend=False))
            fig_exp.add_hline(y=500, line_dash="dot", line_color="grey",
                              annotation_text="500-kW target", annotation_position="top right")
            fig_exp.update_layout(xaxis_title="Flow (t/h)", yaxis_title="Output (kWe)",
                xaxis=dict(range=[0, 8.1]), yaxis=dict(range=[0, max(700, max(micro_kw)+30)]),
                height=420, template='plotly_white', legend=dict(x=0.02, y=0.98))
            st.plotly_chart(fig_exp, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB: METHODOLOGY
# ═══════════════════════════════════════════════
with tabs[-3]:
    st.header("📐 Methodology")
    st.markdown("""
### TESPy Rankine Cycle

```
Source → HP Turbine → Splitter → LP Turbine     → Sink (condenser)
                               → Screw Expander → Sink (factory heat)
```

Each component is a validated TESPy object with IAPWS-IF97 steam properties.
    """)

    st.markdown("### Steam State Table (Reference Cycle)")
    states_data = {
        "Point": ["1", "2s", "2", "3s", "3", "3e", "4"],
        "Description": ["Live steam", "Isentropic HP exhaust", "Actual HP exhaust (TESPy)",
            "Isentropic LP exhaust", "Actual LP exhaust (TESPy)", "Expander exhaust (TESPy)", "Feedwater"],
        "P (bar)": [base_cycle.state1.P_bar, base_cycle.state2s.P_bar, base_cycle.state2.P_bar,
                    base_cycle.state3s.P_bar, base_cycle.state3.P_bar, base_cycle.state3e.P_bar, base_cycle.state4.P_bar],
        "T (°C)": [base_cycle.state1.T_C, base_cycle.state2s.T_C, base_cycle.state2.T_C,
                    base_cycle.state3s.T_C, base_cycle.state3.T_C, base_cycle.state3e.T_C, base_cycle.state4.T_C],
        "h (kJ/kg)": [base_cycle.state1.h_kJkg, base_cycle.state2s.h_kJkg, base_cycle.state2.h_kJkg,
                      base_cycle.state3s.h_kJkg, base_cycle.state3.h_kJkg, base_cycle.state3e.h_kJkg, base_cycle.state4.h_kJkg],
    }
    st.dataframe(pd.DataFrame(states_data).style.format({'P (bar)':'{:.2f}','T (°C)':'{:.1f}','h (kJ/kg)':'{:.1f}'}),
                 use_container_width=True)

    st.markdown(f"""
### Power Equation
$$P [kW] = \\dot{{m}} [t/h] \\times \\Delta h [kJ/kg] \\times \\eta_{{gen}} / 3.6$$

### Biomass Energy Estimation
$$\\text{{ratio}} = \\frac{{h_1 - h_4}}{{\\eta_{{boiler}} \\times LHV}} = \\frac{{{base_cycle.state1.h_kJkg:.1f} - {base_cycle.state4.h_kJkg:.1f}}}{{{eta_boiler:.2f} \\times {LHV*1000:.0f}}} = {(base_cycle.state1.h_kJkg - base_cycle.state4.h_kJkg)/(eta_boiler*LHV*1000):.4f}$$

### Auxiliary Load Model
| Component | Formula |
|---|---|
| Base misc | Fixed 115 kW |
| Steam island | 4 × steam_tph |
| Biomass handling | 6 × biomass_tph |
| FG fan | Motor × (steam / fan_cap)³ |
| Condenser fans | 28 + 6×(cond−6) + 140×(P−0.2); summer: 90 kW |
| Heat system | 15 + 3 × extraction_tph |
| Expander skid | 5 kW if installed |

### Part-Load Corrections
| Section | Formula | Correction range |
|---|---|---|
| HP turbine | η × (1 − 0.16 × (1 − load_ratio)) | [0.90, 1.00] |
| LP turbine | η × (1 − 0.13 × (1 − load_ratio)) | [0.88, 1.00] |
| Expander | η × (1 − 0.10 × (1 − load_ratio)) | [0.90, 1.00] |

### Transformer
Export capped at {trafo:.0f} kW. Curtailed: **{curtailed_label}**.
    """)

# ═══════════════════════════════════════════════
# TAB: ECONOMICS
# ═══════════════════════════════════════════════
with tabs[-2]:
    st.title("💰 Economic Analysis")
    st.caption("Adjust financial and O&M assumptions below. All economics recompute instantly. "
               "Generate all 27 report figures as downloadable ZIP.")

    # --- Financial inputs ---
    st.markdown("### Financial Assumptions")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        econ_elec_price = st.number_input("Electricity price (€/MWh)", value=100.0,
            min_value=30.0, max_value=400.0, step=5.0, key="econ_elec")
        econ_heat_price = st.number_input("Heat price (€/MWhth)", value=35.0,
            min_value=0.0, max_value=100.0, step=5.0, key="econ_heat")
        econ_biomass_price = st.number_input("Biomass price (€/t)", value=71.0,
            min_value=20.0, max_value=200.0, step=1.0, key="econ_bio")
    with fc2:
        econ_discount = st.number_input("Discount rate (WACC)", value=0.05,
            min_value=0.01, max_value=0.25, step=0.01, format="%.2f", key="econ_wacc")
        econ_npv_yrs = st.number_input("NPV horizon (years)", value=20,
            min_value=3, max_value=30, step=1, key="econ_npv_yr")
        econ_cf_yrs = st.number_input("Cashflow years", value=20,
            min_value=5, max_value=30, step=1, key="econ_cf_yr")
    with fc3:
        econ_elec_esc = st.number_input("Elec escalation (%/yr)", value=2.0,
            min_value=-5.0, max_value=10.0, step=0.5, format="%.1f", key="econ_elec_esc") / 100
        econ_bio_esc = st.number_input("Biomass escalation (%/yr)", value=2.0,
            min_value=-5.0, max_value=10.0, step=0.5, format="%.1f", key="econ_bio_esc") / 100
        econ_om_esc = st.number_input("O&M escalation (%/yr)", value=0.0,
            min_value=0.0, max_value=10.0, step=0.5, format="%.1f", key="econ_om_esc") / 100

    st.markdown("### O&M Assumptions")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        econ_fixed_base = st.number_input("Fixed O&M base (kEUR/yr)", value=528.0,
            min_value=50.0, max_value=1000.0, step=10.0, key="econ_fixed")
        econ_runtime = st.number_input("Runtime rate (€/hr)", value=0.0,
            min_value=0.0, max_value=50.0, step=0.5, key="econ_runtime")
    with oc2:
        econ_bio_tp = st.number_input("Biomass throughput O&M (€/t)", value=0.0,
            min_value=0.0, max_value=10.0, step=0.1, key="econ_biotp")
        econ_pwr_tp = st.number_input("Power throughput O&M (€/MWh)", value=0.0,
            min_value=0.0, max_value=5.0, step=0.05, format="%.2f", key="econ_pwrtp")
    with oc3:
        econ_start_cost = st.number_input("Start cost (kEUR/start)", value=0.0,
            min_value=0.0, max_value=10.0, step=0.1, format="%.3f", key="econ_start")
        econ_filter_om = st.number_input("Filter incremental O&M (kEUR)", value=0.0,
            min_value=0.0, max_value=50.0, step=1.0, key="econ_filtom")
        econ_lp_om = st.number_input("LP expander incremental O&M (kEUR)", value=0.0,
            min_value=0.0, max_value=50.0, step=1.0, key="econ_lpom")

    # Build config
    econ_fin = {
        "electricity_price_eur_per_mwh": econ_elec_price,
        "heat_price_eur_per_mwhth": econ_heat_price,
        "biomass_price_eur_per_t": econ_biomass_price,
        "npv_horizon_years": econ_npv_yrs,
        "cashflow_years": econ_cf_yrs,
        "discount_rate": econ_discount,
        "electricity_escalation": econ_elec_esc,
        "heat_escalation": 0.0,
        "biomass_escalation": econ_bio_esc,
        "om_escalation": econ_om_esc,
    }
    econ_om = {**DEFAULT_OM,
        "true_fixed_base_kEUR": econ_fixed_base,
        "runtime_rate_eur_per_hr": econ_runtime,
        "biomass_throughput_eur_per_t": econ_bio_tp,
        "power_throughput_eur_per_mwh": econ_pwr_tp,
        "start_cost_kEUR": econ_start_cost,
        "filter_incremental_om_kEUR": econ_filter_om,
        "lp_incremental_om_kEUR": econ_lp_om,
    }

    # Build economics dataframe from current scenario results
    econ_records = []
    for r in results:
        sid = r.sid
        econ_records.append({
            "ID": sid, "Scenario": r.name,
            "Category": "Scenario" if sid.startswith("S") else "Case",
            "Steam_tph": float(r.steam_tph),
            "Extraction_tph": float(r.extraction_tph),
            "Condensing_tph": float(r.condensing_tph),
            "Availability": float(r.availability_pct) / 100.0,
            "CAPEX_kEUR": float(r.capex_kEUR),
            "LPExpander": "Y" if r.has_lp_expander else "N",
            "FilterUpgrade": "Y" if sid in FILTER_UPGRADE_IDS else "N",
            "FanUpgrade": "Y" if sid in FAN_UPGRADE_IDS else "N",
            "FG_Nm3ph": FG_NM3PH_DEFAULT.get(sid, np.nan),
            "BagUtil": BAG_UTIL_DEFAULT.get(sid, np.nan),
            "FanUtil": FAN_UTIL_DEFAULT.get(sid, np.nan),
            "HPPower_kW": float(r.P_HP), "LPPower_kW": float(r.P_LP),
            "Expander_kW": float(r.P_exp), "Gross_kW": float(r.P_gross),
            "Aux_kW": float(r.P_aux), "Net_kW": float(r.P_net),
            "Export_kW": float(r.P_export), "Curtailed_kW": float(r.P_curtailed),
            "AnnualMWh": float(r.annual_MWh_export),
            "AnnualMWhGross": float(r.annual_MWh_gross),
            "HeatMWhth": float(r.heat_output_kW * r.hours_yr / 1000.0),
            "Biomass_tpy": float(r.biomass_tpy),
            "ElecEff": float(r.elec_eff_pct_net) / 100.0,
            "CHPEff": float(r.chp_eff_pct) / 100.0,
            "Status": "OK",
        })
    econ_df_raw = pd.DataFrame(econ_records).sort_values("ID", key=lambda s: s.map(SORT_MAP)).reset_index(drop=True)
    econ_df = enrich_dataframe(econ_df_raw, econ_fin, econ_om)

    # --- KPI cards ---
    st.markdown("---")
    st.markdown("### Key Economic Results")
    investable = econ_df[econ_df["CAPEX_kEUR"] > 0].copy()
    kc1, kc2, kc3, kc4 = st.columns(4)
    bl_margin = float(econ_df[econ_df["ID"]=="S1"]["Margin_kEUR"].iloc[0])
    kc1.metric("S1 Baseline Margin", f"€{bl_margin:.0f}k/yr")
    if len(investable) > 0:
        best = investable.sort_values("NPV_kEUR", ascending=False).iloc[0]
        kc2.metric("Best NPV", f"€{best['NPV_kEUR']:.0f}k", f"{best['ID']}: {best['Scenario']}")
        pb_str = f"{best['PBP_yr']:.1f} yr" if np.isfinite(best['PBP_yr']) else "n/a"
        kc3.metric("Best NPV Payback", pb_str, f"CAPEX €{best['CAPEX_kEUR']:.0f}k")
        best_m = investable.sort_values("DeltaMargin_kEUR", ascending=False).iloc[0]
        kc4.metric("Best Δ Margin", f"€{best_m['DeltaMargin_kEUR']:+.0f}k/yr", best_m["ID"])

    # --- Full economics table ---
    st.markdown("### Full Economic Comparison")
    econ_show = econ_df[["ID","Scenario","CAPEX_kEUR","Export_kW","AnnualMWh","HeatMWhth",
        "Biomass_tpy","TotalOMSharpened_kEUR","TotalRevenue_kEUR","Margin_kEUR",
        "DeltaMargin_kEUR","NPV_kEUR","PBP_yr"]].copy()
    econ_show.columns = ["ID","Scenario","CAPEX","Export kW","MWh/yr","Heat MWhth",
        "Biomass t/yr","O&M kEUR","Revenue kEUR","Margin kEUR",
        "Δ Margin","NPV kEUR","PBP yr"]
    st.dataframe(econ_show.style.format({
        'CAPEX':'{:.0f}','Export kW':'{:.0f}','MWh/yr':'{:,.0f}',
        'Heat MWhth':'{:,.0f}','Biomass t/yr':'{:,.0f}','O&M kEUR':'{:.0f}',
        'Revenue kEUR':'{:.0f}','Margin kEUR':'{:.0f}','Δ Margin':'{:+.0f}',
        'NPV kEUR':'{:.0f}','PBP yr':'{:.1f}',
    }).background_gradient(subset=['NPV kEUR'], cmap='RdYlGn'), use_container_width=True, height=460)
    

    st.markdown("### Delta Margin Comparison")
    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
    x=econ_df["ID"],
    y=econ_df["DeltaMargin_kEUR"],
    marker_color=[SCENARIO_COLORS.get(s, "#888") for s in econ_df["ID"]],
    text=[f"€{v:+.0f}k" for v in econ_df["DeltaMargin_kEUR"]],
    textposition='outside'
    ))
    fig_m.add_hline(y=0, line_dash="dash", line_color="#6FAFC8", annotation_text="S1 baseline")
    fig_m.update_layout(
    title="Delta Margin vs S1 (kEUR/yr)",
    yaxis_title="Δ Margin (kEUR/yr)",
    height=420,
    template='plotly_white'
    )
    st.plotly_chart(fig_m, use_container_width=True)
    # --- Interactive Plotly charts ---
    

    st.markdown("### NPV vs CAPEX")
    fig_npv = go.Figure()
    for _, row in econ_df.iterrows():
        fig_npv.add_trace(go.Scatter(
            x=[row["CAPEX_kEUR"]], y=[row["NPV_kEUR"]],
            mode='markers+text',
            marker=dict(size=14, color=SCENARIO_COLORS.get(row["ID"],"#888")),
            text=[row["ID"]], textposition='top center', showlegend=False))
    fig_npv.add_hline(y=0, line_color="grey")
    fig_npv.update_layout(title="NPV vs CAPEX (kEUR)",
        xaxis_title="CAPEX (kEUR)", yaxis_title="NPV (kEUR)",
        height=420, template='plotly_white')
    st.plotly_chart(fig_npv, use_container_width=True)

    st.markdown("### Cumulative Discounted Cash Flow")
    fig_cf = go.Figure()
    for r in results:
        cft = incremental_cashflow_table(econ_df, r.sid, econ_fin, econ_om, years=econ_cf_yrs)
        fig_cf.add_trace(go.Scatter(
            x=cft["Year"], y=cft["CumDiscountedIncrementalFCF_kEUR"],
            mode='lines', name=f"{r.sid}: {r.name}",
            line=dict(color=SCENARIO_COLORS.get(r.sid,"#888"), width=2)))
    fig_cf.add_hline(y=0, line_color="grey")
    fig_cf.update_layout(
        title=f"{econ_cf_yrs}-Year Cumulative Discounted Incremental Cash Flow vs S1",
        xaxis_title="Year", yaxis_title="kEUR", height=450, template='plotly_white')
    st.plotly_chart(fig_cf, use_container_width=True)

    # --- Year-by-year margin evolution with escalation ---
    st.markdown("### Margin Evolution Over Time (with escalation)")
    fig_evo = go.Figure()
    show_sids = ["S1", "S6", "S7", "S8", "C1", "C2"]
    for r in results:
        if r.sid not in show_sids:
            continue
        row_data = econ_df.set_index("ID").loc[r.sid]
        yearly_margins = []
        for y in range(1, econ_cf_yrs + 1):
            rc = annual_revenue_costs_kEUR(row_data, econ_fin, econ_om, year=y)
            m = rc["TotalRevenue_kEUR"] - rc["BiomassCost_kEUR"] - rc["TotalOMSharpened_kEUR"]
            yearly_margins.append(m)
        fig_evo.add_trace(go.Scatter(
            x=list(range(1, econ_cf_yrs + 1)), y=yearly_margins,
            mode='lines', name=f"{r.sid}: {r.name}",
            line=dict(color=SCENARIO_COLORS.get(r.sid, "#888"), width=2.5)))
    fig_evo.add_hline(y=0, line_color="grey", line_width=0.5)
    fig_evo.update_layout(
        title=f"Annual Margin by Year (€{econ_elec_price:.0f}/MWh elec +{econ_elec_esc*100:.1f}%/yr, €{econ_biomass_price:.0f}/t bio +{econ_bio_esc*100:.1f}%/yr)",
        xaxis_title="Year", yaxis_title="Margin (kEUR/yr)",
        height=480, template='plotly_white', legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig_evo, use_container_width=True)

    # --- Generate all 28 report figures and download ---
    st.markdown("---")
    st.markdown("### 📥 Download All 28 Report Figures (PNG + CSV)")
    st.caption("Generates the complete set of techno-economic figures with the exact report styling "
               "(DejaVu Sans, 360 DPI, matching colors and layout). Packaged as ZIP with CSVs.")

    if st.button("🖨️ Generate All 28 Figures & Download", type="primary"):
        with st.spinner("Generating 28 high-resolution figures... this takes ~30 seconds"):
            fig_out_dir = Path("/tmp/chp_econ_figures")
            # Build the config dict that generate_figures expects
            fig_cfg = {
                "financial_assumptions": econ_fin,
                "om_assumptions": econ_om,
                "technical_assumptions": {
                    "live_steam_pressure_bar": P_live,
                    "live_steam_temperature_C": T_live,
                    "extraction_pressure_bar": P_ext,
                    "condenser_pressure_bar": P_cond_ref,
                    "feedwater_temperature_C": T_fw,
                    "eta_hp": eta_HP, "eta_lp": eta_LP,
                    "eta_exp": eta_exp, "eta_gen": eta_gen,
                    "transformer_limit_kW": trafo,
                    "biomass_lhv_mj_per_kg": LHV,
                    "boiler_efficiency": eta_boiler,
                },
                "scenario_overrides": {},
            }
            fig_files = generate_figures(econ_df_raw, fig_cfg, fig_out_dir)

        st.success(f"Generated {len(fig_files)} figures")

        zip_path = fig_out_dir.parent / f"{fig_out_dir.name}.zip"
        with open(zip_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download ZIP ({zip_path.stat().st_size//1024} KB, {len(fig_files)} PNGs + CSVs)",
                data=f, file_name="chp_steenwijk_economics.zip",
                mime="application/zip")

        # Store generated files in session state for Gallery tab
        st.session_state["generated_fig_files"] = sorted(fig_files)

# ═══════════════════════════════════════════════
# TAB: FIGURES GALLERY
# ═══════════════════════════════════════════════
with tabs[-1]:
    st.title("🖼️ Figures Gallery")
    st.caption("After generating figures in the Economics tab, all 28 report-quality PNGs are displayed here.")

    if "generated_fig_files" in st.session_state and st.session_state["generated_fig_files"]:
        fig_files_gallery = st.session_state["generated_fig_files"]
        st.success(f"{len(fig_files_gallery)} figures available")

        # View mode selector
        view_mode = st.radio("Layout", ["2 columns", "Full width"], horizontal=True)

        if view_mode == "Full width":
            for fp in fig_files_gallery:
                st.image(str(fp), caption=fp.stem, use_container_width=True)
                st.markdown("---")
        else:
            cols = st.columns(2)
            for idx, fp in enumerate(fig_files_gallery):
                with cols[idx % 2]:
                    st.image(str(fp), caption=fp.stem, use_container_width=True)
    else:
        st.info("No figures generated yet. Go to the **💰 Economics** tab and click **Generate All 28 Figures** first.")

