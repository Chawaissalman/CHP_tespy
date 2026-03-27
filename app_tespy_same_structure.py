"""
CHP Steenwijk — TESPy-Native Plant Performance App (same tab structure as original)
Run with:
    pip install streamlit CoolProp plotly pandas numpy
    streamlit run app_improved_same_structure.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tespy_engine_chp import analyse_cycle, compute_scenario, SCENARIO_DEFS

st.set_page_config(page_title="CHP Steenwijk — TESPy-Native Plant Performance", layout="wide", page_icon="🔥")

# ═══════════════════════════════════════════════
# SIDEBAR: All Inputs
# ═══════════════════════════════════════════════
st.sidebar.title("🔧 Input Parameters")

st.sidebar.header("Steam Cycle")
P_live = st.sidebar.number_input("Live steam pressure (bar)", value=50.0, min_value=20.0, max_value=100.0, step=1.0)
T_live = st.sidebar.number_input("Live steam temperature (°C)", value=444.0, min_value=350.0, max_value=550.0, step=1.0)
P_ext = st.sidebar.number_input("Extraction pressure (bar)", value=2.5, min_value=0.5, max_value=10.0, step=0.1)
P_cond = st.sidebar.number_input("Reference condenser pressure (bar)", value=0.20, min_value=0.03, max_value=1.0, step=0.01, format="%.2f")
T_fw = st.sidebar.number_input("Feedwater temperature (°C)", value=130.0, min_value=50.0, max_value=200.0, step=5.0)

st.sidebar.header("Isentropic Efficiencies")
eta_HP = st.sidebar.slider("η HP turbine", 0.60, 0.95, 0.80, 0.01)
eta_LP = st.sidebar.slider("η LP turbine", 0.60, 0.95, 0.78, 0.01)
eta_exp = st.sidebar.slider("η LP expander (screw)", 0.50, 0.90, 0.75, 0.01)
eta_gen = st.sidebar.slider("η generator + mechanical", 0.85, 0.98, 0.94, 0.01)

st.sidebar.header("Plant Limits")
trafo = st.sidebar.number_input("Transformer limit (kW)", value=2520.0, step=10.0)

#st.sidebar.header("Boiler & Fuel")
#LHV = st.sidebar.number_input("Biomass LHV (MJ/kg)", value=10.5, min_value=5.0, max_value=20.0, step=0.5)
#eta_boiler = st.sidebar.slider("Boiler efficiency", 0.70, 0.95, 0.85, 0.01)
LHV = 10.5
eta_boiler = 0.85
# ═══════════════════════════════════════════════
# RUN ANALYSIS
# ═══════════════════════════════════════════════
base_cycle = analyse_cycle(
    P_live_bar=P_live,
    T_live_C=T_live,
    P_extraction_bar=P_ext,
    P_condenser_bar=P_cond,
    T_feedwater_C=T_fw,
    eta_HP=eta_HP,
    eta_LP=eta_LP,
    eta_exp=eta_exp,
    eta_gen=eta_gen,
)

results = [
    compute_scenario(
        cycle=base_cycle,
        transformer_limit_kW=trafo,
        LHV_MJkg=LHV,
        eta_boiler=eta_boiler,
        **sd,
    )
    for sd in SCENARIO_DEFS
]
baseline = results[0]

rows = []
for r in results:
    rows.append({
        "ID": r.sid,
        "Scenario": r.name,
        "Steam (t/h)": r.steam_tph,
        "Extraction (t/h)": r.extraction_tph,
        "Condensing (t/h)": r.condensing_tph,
        "LP Expander": "✓" if r.has_lp_expander else "—",
        "P_cond used (bar)": r.condenser_pressure_bar,
        "P_HP (kW)": r.P_HP,
        "P_LP (kW)": r.P_LP,
        "P_Exp (kW)": r.P_exp,
        "P_Gross (kW)": r.P_gross,
        "P_Aux (kW)": r.P_aux,
        "P_Net (kW)": r.P_net,
        "P_Export (kW)": r.P_export,
        "Curtailed (kW)": r.P_curtailed,
        "ΔP vs S1 (kW)": round(r.P_export - baseline.P_export, 1),
        "Annual MWh Gross": r.annual_MWh_gross,
        "Annual MWh Export": r.annual_MWh_export,
        "Avail (%)": r.availability_pct,
        "CAPEX (kEUR)": r.capex_kEUR,
        "η_HP eff (%)": r.eta_HP_eff * 100,
        "η_LP eff (%)": r.eta_LP_eff * 100,
        "η_Exp eff (%)": r.eta_exp_eff * 100,
        "η_elec gross (%)": r.elec_eff_pct_gross,
        "η_elec net (%)": r.elec_eff_pct_net,
        "η_CHP (%)": r.chp_eff_pct,
        "Heat to factory (kW)": r.heat_output_kW,
        "Biomass (t/h)": r.biomass_tph,
        "Biomass (t/yr)": r.biomass_tpy,
    })

df = pd.DataFrame(rows)
curtailed_ids = df.loc[df["Curtailed (kW)"] > 0, "ID"].tolist()
curtailed_label = ", ".join(curtailed_ids) if curtailed_ids else "none"

# ═══════════════════════════════════════════════
# TAB STRUCTURE
# ═══════════════════════════════════════════════
tab_names = ["📊 Overview & Comparison"] + [f"{'⚡' if 'S' in r.sid else '🔗'} {r.sid}: {r.name}" for r in results] + ["📐 Methodology"]
tabs = st.tabs(tab_names)

# ═══════════════════════════════════════════════
# TAB 0: OVERVIEW
# ═══════════════════════════════════════════════
with tabs[0]:
    st.title("CHP Steenwijk — TESPy-Native Plant Performance")
    #st.markdown("**Same app structure as the original, with all scenarios solved from a TESPy network.**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Gross Power", f"{baseline.P_gross:.0f} kW", f"HP {baseline.P_HP:.0f} + LP {baseline.P_LP:.0f}")
    col2.metric("Baseline Auxiliaries", f"{baseline.P_aux:.0f} kW", f"Net {baseline.P_net:.0f} kW")
    col3.metric("Baseline Export", f"{baseline.P_export:.0f} kW", f"S1 @ {baseline.steam_tph} t/h")
    col4.metric("Curtailed Scenarios", curtailed_label)

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Bar(name='HP Turbine', x=df['ID'], y=df['P_HP (kW)'], marker_color='#0B6FA4'))
    fig.add_trace(go.Bar(name='LP Turbine', x=df['ID'], y=df['P_LP (kW)'], marker_color='#5F9BD1'))
    fig.add_trace(go.Bar(name='LP Expander', x=df['ID'], y=df['P_Exp (kW)'], marker_color='#D98C0B'))
    fig.add_trace(go.Bar(name='Auxiliaries', x=df['ID'], y=-df['P_Aux (kW)'], marker_color='rgba(80,80,80,0.55)'))
    fig.add_trace(go.Bar(name='Curtailed', x=df['ID'], y=-df['Curtailed (kW)'], marker_color='rgba(192,57,43,0.5)'))
    fig.add_hline(y=trafo, line_dash="dash", line_color="red", annotation_text=f"Transformer {trafo:.0f} kW")
    fig.update_layout(barmode='relative', title='Power Breakdown by Section (gross → aux → export)',
                      yaxis_title='Power (kW)', height=470, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    deltas = df['ΔP vs S1 (kW)'].tolist()
    colors = ['#3C8D5A' if d >= 0 else '#C0392B' for d in deltas]
    fig2.add_trace(go.Bar(x=df['ID'], y=deltas, marker_color=colors,
                          text=[f"{d:+.0f}" for d in deltas], textposition='outside'))
    fig2.add_hline(y=0, line_color="black", line_width=0.5)
    fig2.update_layout(title='Export Power Increase vs S1 Baseline (kW)',
                       yaxis_title='Δ Export (kW)', height=400, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏆 Scenario Ranking")
    ranked = df[df['CAPEX (kEUR)'] > 0].sort_values('ΔP vs S1 (kW)', ascending=False)
    if len(ranked) > 0:
        best = ranked.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.success(
            f"**Best export increase:** {best['ID']} — {best['Scenario']}\n\n"
            f"ΔP = +{best['ΔP vs S1 (kW)']:.0f} kW | CAPEX = €{best['CAPEX (kEUR)']:.0f}k"
        )
        best_eff = df[df['CAPEX (kEUR)'] > 0].sort_values('η_elec net (%)', ascending=False).iloc[0]
        c2.info(
            f"**Best net electrical efficiency:** {best_eff['ID']} — {best_eff['Scenario']}\n\n"
            f"η_elec,net = {best_eff['η_elec net (%)']:.1f}% | η_CHP = {best_eff['η_CHP (%)']:.1f}%"
        )
        s8 = df[df['ID'] == 'S8'].iloc[0]
        c3.warning(
            f"**S8 — LP Expander (no extra fuel)**\n\n"
            f"ΔP = +{s8['ΔP vs S1 (kW)']:.0f} kW from pressure-energy recovery\n\n"
            f"Heat is now calculated from post-expander enthalpy"
        )

    st.markdown("### Full Comparison Table")
    st.dataframe(df.style.format({
        'P_cond used (bar)': '{:.2f}',
        'P_HP (kW)': '{:.0f}', 'P_LP (kW)': '{:.0f}', 'P_Exp (kW)': '{:.0f}',
        'P_Gross (kW)': '{:.0f}', 'P_Aux (kW)': '{:.0f}', 'P_Net (kW)': '{:.0f}',
        'P_Export (kW)': '{:.0f}', 'Curtailed (kW)': '{:.0f}',
        'ΔP vs S1 (kW)': '{:+.0f}', 'Annual MWh Gross': '{:,.0f}', 'Annual MWh Export': '{:,.0f}',
        'η_HP eff (%)': '{:.1f}', 'η_LP eff (%)': '{:.1f}', 'η_Exp eff (%)': '{:.1f}',
        'η_elec gross (%)': '{:.1f}', 'η_elec net (%)': '{:.1f}', 'η_CHP (%)': '{:.1f}',
        'Heat to factory (kW)': '{:.0f}', 'Biomass (t/h)': '{:.2f}', 'Biomass (t/yr)': '{:,.0f}',
    }).background_gradient(subset=['ΔP vs S1 (kW)'], cmap='RdYlGn'),
    use_container_width=True, height=470)

    st.markdown("### Efficiency Comparison")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Electrical Efficiency (gross)', x=df['ID'], y=df['η_elec gross (%)'], marker_color='#5F9BD1'))
    fig3.add_trace(go.Bar(name='Electrical Efficiency (net)', x=df['ID'], y=df['η_elec net (%)'], marker_color='#0B6FA4'))
    fig3.add_trace(go.Bar(name='CHP Efficiency', x=df['ID'], y=df['η_CHP (%)'], marker_color='#D98C0B'))
    fig3.update_layout(barmode='group', yaxis_title='Efficiency (%)', height=410, template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════
# TABS 1-11: Individual Scenarios
# ═══════════════════════════════════════════════
for i, r in enumerate(results):
    with tabs[i + 1]:
        st.header(f"{r.sid}: {r.name}")
        st.markdown(f"*{r.description}*")

        st.markdown("---")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        delta = r.P_export - baseline.P_export
        col1.metric("P_export", f"{r.P_export:.0f} kW", f"{delta:+.0f} vs S1")
        col2.metric("P_gross", f"{r.P_gross:.0f} kW")
        col3.metric("Aux load", f"{r.P_aux:.0f} kW")
        col4.metric("Annual MWh", f"{r.annual_MWh_export:,}")
        col5.metric("η_elec, net", f"{r.elec_eff_pct_net:.1f}%")
        col6.metric("CAPEX", f"€{r.capex_kEUR:.0f}k")

        st.markdown("### Assumptions")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | Total steam | **{r.steam_tph} t/h** |
            | Extraction | **{r.extraction_tph} t/h** |
            | Condensing | **{r.condensing_tph} t/h** |
            | Availability | **{r.availability_pct}%** |
            | LP Expander | **{'Yes' if r.has_lp_expander else 'No'}** |
            | Condenser pressure used | **{r.condenser_pressure_bar:.2f} bar** |
            | Summer full condenser fans | **{'Yes' if r.summer_full_condenser_fans else 'No'}** |
            """)
        with a2:
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | Live steam | **{P_live} bar, {T_live}°C** |
            | Extraction P | **{P_ext} bar** |
            | Reference condenser P | **{P_cond:.2f} bar** |
            | η_HP / η_LP / η_exp (design) | **{eta_HP:.0%} / {eta_LP:.0%} / {eta_exp:.0%}** |
            | η_HP / η_LP / η_exp (effective) | **{r.eta_HP_eff:.1%} / {r.eta_LP_eff:.1%} / {r.eta_exp_eff:.1%}** |
            | η_gen | **{eta_gen:.0%}** |
            """)

        c = r.cycle
        st.markdown("### Power Calculation (Step-by-Step)")
        st.markdown(f"""
        **HP Section** — all {r.steam_tph} t/h passes through HP turbine:
        ```
        w_HP = h₁ − h₂ = {c.state1.h_kJkg:.1f} − {c.state2.h_kJkg:.1f} = {c.w_HP:.1f} kJ/kg

        P_HP = ṁ_total × w_HP × η_gen / 3.6
             = {r.steam_tph} × {c.w_HP:.1f} × {eta_gen:.2f} / 3.6
             = {r.P_HP:.1f} kW
        ```

        **LP Section** — {r.condensing_tph} t/h continues through LP:
        ```
        w_LP = h₂ − h₃ = {c.state2.h_kJkg:.1f} − {c.state3.h_kJkg:.1f} = {c.w_LP:.1f} kJ/kg

        P_LP = ṁ_cond × w_LP × η_gen / 3.6
             = {r.condensing_tph} × {c.w_LP:.1f} × {eta_gen:.2f} / 3.6
             = {r.P_LP:.1f} kW
        ```
        """)

        if r.has_lp_expander:
            st.markdown(f"""
            **LP Expander** — {r.extraction_tph} t/h expands from extraction pressure to condenser pressure:
            ```
            w_exp = h₂ − h₃e = {c.state2.h_kJkg:.1f} − {c.state3e.h_kJkg:.1f} = {c.w_exp:.1f} kJ/kg

            P_exp = ṁ_ext × w_exp × η_gen / 3.6
                  = {r.extraction_tph} × {c.w_exp:.1f} × {eta_gen:.2f} / 3.6
                  = {r.P_exp:.1f} kW
            ```

            **Useful heat** is now taken from the **post-expander** state:
            ```
            q_heat,useful = h₃e − h₄ = {c.state3e.h_kJkg:.1f} − {c.state4.h_kJkg:.1f} = {c.q_heat_post_expander:.1f} kJ/kg
            Heat = ṁ_ext × q_heat,useful / 3.6 = {r.heat_output_kW:.1f} kW
            ```
            """)
        else:
            st.markdown(f"""
            **Useful heat to factory** — direct extraction case:
            ```
            q_heat,useful = h₂ − h₄ = {c.state2.h_kJkg:.1f} − {c.state4.h_kJkg:.1f} = {c.q_heat_direct:.1f} kJ/kg
            Heat = ṁ_ext × q_heat,useful / 3.6 = {r.heat_output_kW:.1f} kW
            ```
            """)

        st.markdown(f"""
        **Gross, net, and export power**
        ```
        P_gross = P_HP + P_LP + P_exp = {r.P_gross:.1f} kW
        P_aux   = {r.P_aux:.1f} kW
        P_net   = P_gross − P_aux = {r.P_net:.1f} kW
        P_export = min(P_net, transformer limit) = {r.P_export:.1f} kW
        ```
        """)

        st.markdown("### Auxiliary Load Breakdown")
        aux_df = pd.DataFrame({
            "Component": [
                "Base misc", "Steam island", "Biomass handling", "FG fan",
                "Condenser fans", "Heat system", "Expander skid"
            ],
            "Load (kW)": [
                r.aux_breakdown.base_misc_kW,
                r.aux_breakdown.steam_island_kW,
                r.aux_breakdown.biomass_handling_kW,
                r.aux_breakdown.fg_fan_kW,
                r.aux_breakdown.condenser_fans_kW,
                r.aux_breakdown.heat_system_kW,
                r.aux_breakdown.expander_skid_kW,
            ]
        })
        st.dataframe(aux_df.style.format({"Load (kW)": "{:.1f}"}), use_container_width=True, height=290)

        st.markdown("### Scenario Results")
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
            | Gross electrical efficiency | **{r.elec_eff_pct_gross:.1f}%** |
            | Net electrical efficiency | **{r.elec_eff_pct_net:.1f}%** |
            | CHP efficiency | **{r.chp_eff_pct:.1f}%** |
            | Heat to factory | **{r.heat_output_kW:.0f} kW** |
            | Biomass consumption | **{r.biomass_tph:.2f} t/h** |
            | Biomass annual | **{r.biomass_tpy:,} t/yr** |
            """)

        fig = go.Figure(data=[go.Pie(
            labels=['HP Turbine', 'LP Turbine', 'LP Expander', 'Auxiliaries', 'Curtailed'],
            values=[max(r.P_HP, 0.0), max(r.P_LP, 0.0), max(r.P_exp, 0.0), max(r.P_aux, 0.0), max(r.P_curtailed, 0.0)],
            marker_colors=['#0B6FA4', '#5F9BD1', '#D98C0B', '#7F8C8D', '#C0392B'],
            hole=0.4, textinfo='label+value+percent',
        )])
        fig.update_layout(title=f'{r.sid} Power Breakdown', height=360)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB: METHODOLOGY
# ═══════════════════════════════════════════════
with tabs[-1]:
    st.header("📐 Methodology")

    st.markdown("""
    ### TESPy-native thermodynamic framework

    This version solves each scenario with a **TESPy network**, not with a manual
    enthalpy-drop calculator. The network includes:

    1. **Steam generator**
    2. **HP turbine**
    3. **One extraction split**
    4. **LP condensing branch**
    5. **Optional LP expander on extraction steam**
    6. **Main condenser and process heat exchanger**
    7. **Condensate pumps, condensate merge, and feedwater conditioning**

    The app layout is unchanged, but the scenario results now come from TESPy's
    component and network balances.
    """)

    states_data = {
        "Point": ["1", "2s", "2", "3s", "3", "3e", "4"],
        "Description": [
            "Live steam (steam generator outlet)",
            "Isentropic HP exhaust reference",
            "Actual HP exhaust / extraction header",
            "Isentropic LP exhaust reference",
            "Actual LP exhaust to condenser",
            "Actual LP expander exhaust reference",
            "Process-return condensate reference",
        ],
        "P (bar)": [
            base_cycle.state1.P_bar, base_cycle.state2s.P_bar, base_cycle.state2.P_bar,
            base_cycle.state3s.P_bar, base_cycle.state3.P_bar, base_cycle.state3e.P_bar, base_cycle.state4.P_bar
        ],
        "T (°C)": [
            base_cycle.state1.T_C, base_cycle.state2s.T_C, base_cycle.state2.T_C,
            base_cycle.state3s.T_C, base_cycle.state3.T_C, base_cycle.state3e.T_C, base_cycle.state4.T_C
        ],
        "h (kJ/kg)": [
            base_cycle.state1.h_kJkg, base_cycle.state2s.h_kJkg, base_cycle.state2.h_kJkg,
            base_cycle.state3s.h_kJkg, base_cycle.state3.h_kJkg, base_cycle.state3e.h_kJkg, base_cycle.state4.h_kJkg
        ],
        "How computed": [
            f"TESPy live-steam boundary: P={P_live} bar, T={T_live}°C",
            "TESPy single-stage reference solve with η=1.0 to extraction pressure",
            f"TESPy HP turbine solve with η_HP={eta_HP:.0%}",
            "TESPy single-stage reference solve with η=1.0 to condenser pressure",
            f"TESPy LP turbine solve with η_LP={eta_LP:.0%}",
            f"TESPy expander reference solve with η_exp={eta_exp:.0%}",
            "TESPy process heat exchanger outlet / condensate return state",
        ],
    }
    st.dataframe(pd.DataFrame(states_data).style.format({
        'P (bar)': '{:.2f}', 'T (°C)': '{:.1f}', 'h (kJ/kg)': '{:.1f}'
    }), use_container_width=True)

    boiler_h = base_cycle.boiler_inlet.h_kJkg if base_cycle.boiler_inlet else float('nan')
    boiler_t = base_cycle.boiler_inlet.T_C if base_cycle.boiler_inlet else float('nan')
    boiler_p = base_cycle.boiler_inlet.P_bar if base_cycle.boiler_inlet else float('nan')

    st.markdown(f"""
    ### Power equation shown in the scenario tabs

    For each power-producing section:

    $$P [kW] = \dot{{m}} [t/h] \times \Delta h [kJ/kg] \times \eta_{{gen}} / 3.6$$

    In the rebuilt app, the **displayed section powers** come from **TESPy component
    powers**, while the equation above is retained in the UI as an engineering
    interpretation of the solved result.

    ### Steam generator duty

    The boiler-side steam duty is taken from the **TESPy steam generator heat duty**.
    Fuel input is then calculated as:

    $$Q_{{fuel}} = Q_{{steam\,generator}} / \eta_{{boiler}}$$

    Current reference boiler-inlet state from TESPy:

    - Pressure: **{boiler_p:.2f} bar**
    - Temperature: **{boiler_t:.1f} °C**
    - Enthalpy: **{boiler_h:.1f} kJ/kg**

    ### Specific work values (reference cycle)

    | Section | Δh (kJ/kg) | Basis |
    |---|---|---|
    | HP turbine | **{base_cycle.w_HP:.1f}** | TESPy live steam to HP outlet |
    | LP turbine | **{base_cycle.w_LP:.1f}** | TESPy HP outlet to LP outlet |
    | LP expander | **{base_cycle.w_exp:.1f}** | TESPy HP outlet to expander outlet |
    | Process heat (direct extraction) | **{base_cycle.q_heat_direct:.1f}** | TESPy process HX duty / extraction mass flow |
    | Process heat (post-expander) | **{base_cycle.q_heat_post_expander:.1f}** | TESPy post-expander process HX duty / extraction mass flow |

    ### Net export equation

    $$P_{{net}} = P_{{gross}} - P_{{aux}}$$
    $$P_{{export}} = \min(P_{{net}}, P_{{transformer}})$$

    ### Transformer limit

    Total export is capped at the transformer rating ({trafo:.0f} kW).
    Curtailment is calculated dynamically from model results. For the current settings,
    curtailed scenarios are: **{curtailed_label}**.

    ### Summer backpressure correction for S4

    S4 uses a higher condenser pressure than the reference case and assumes all
    condenser fans at full duty.

    ### Biomass consumption

    Biomass use is linked to the TESPy steam-generator duty using the boiler efficiency
    and biomass LHV entered in the app.
    """)
