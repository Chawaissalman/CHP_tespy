"""
CHP Steenwijk — TESPy-Based Thermodynamic + Plant Performance Engine
=====================================================================
Replaces CoolProp manual Rankine with TESPy network:
  Source -> HP Turbine -> Splitter -> LP Turbine   -> Sink (condenser)
                                   -> Screw Expander -> Sink (factory heat)

Auxiliary model is IDENTICAL to the original thermo_engine_improved.py.
"""

from __future__ import annotations
import json, warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import CoolProp.CoolProp as CP

warnings.filterwarnings("ignore")
from tespy.networks import Network
from tespy.components import Turbine, Splitter, Sink, Source
from tespy.connections import Connection

# ==========================================================================
# STEAM STATE
# ==========================================================================
@dataclass
class SteamState:
    name: str; P_bar: float; T_C: Optional[float] = None
    h_kJkg: Optional[float] = None; s_kJkgK: Optional[float] = None
    x: Optional[float] = None; phase: str = ""
    def __repr__(self):
        parts = [f"{self.name}: P={self.P_bar:.3f} bar"]
        if self.T_C is not None: parts.append(f"T={self.T_C:.1f}°C")
        if self.h_kJkg is not None: parts.append(f"h={self.h_kJkg:.1f} kJ/kg")
        if self.s_kJkgK is not None: parts.append(f"s={self.s_kJkgK:.4f} kJ/kg.K")
        if self.x is not None: parts.append(f"x={self.x:.3f}")
        if self.phase: parts.append(f"[{self.phase}]")
        return ", ".join(parts)

def _state_from_conn(name, conn):
    P_bar = conn.p.val; T_C = conn.T.val; h = conn.h.val; s = conn.s.val
    x = conn.x.val if hasattr(conn.x, 'val') else None
    if x is not None and 0 < x < 1:
        phase = f"two-phase (x={x:.3f})"
    else:
        try:
            T_sat = CP.PropsSI("T","P",P_bar*1e5,"Q",0,"Water")-273.15
            phase = "superheated" if T_C > T_sat+0.5 else "saturated"
        except: phase = "superheated"
    return SteamState(name=name,P_bar=P_bar,T_C=T_C,h_kJkg=h,s_kJkgK=s,
                      x=x if (x is not None and 0<x<1) else None, phase=phase)

def _state_from_PT(name, P_bar, T_C):
    P_Pa=P_bar*1e5; T_K=T_C+273.15
    h=CP.PropsSI("H","P",P_Pa,"T",T_K,"Water")/1000
    s=CP.PropsSI("S","P",P_Pa,"T",T_K,"Water")/1000
    phase=CP.PhaseSI("P",P_Pa,"T",T_K,"Water")
    return SteamState(name=name,P_bar=P_bar,T_C=T_C,h_kJkg=h,s_kJkgK=s,phase=phase)

def _isentropic_state(name, P_bar, s_kJkgK):
    P_Pa=P_bar*1e5; s_SI=s_kJkgK*1000
    h=CP.PropsSI("H","P",P_Pa,"S",s_SI,"Water")/1000
    T=CP.PropsSI("T","P",P_Pa,"S",s_SI,"Water")-273.15
    return SteamState(name=name,P_bar=P_bar,T_C=T,h_kJkg=h,s_kJkgK=s_kJkgK,phase="isentropic ref")

# ==========================================================================
# CYCLE RESULT
# ==========================================================================
@dataclass
class CycleResult:
    state1: SteamState; state2s: SteamState; state2: SteamState
    state3s: SteamState; state3: SteamState; state3e: SteamState; state4: SteamState
    w_HP: float; w_LP: float; w_exp: float
    q_heat_direct: float; q_heat_post_expander: float
    eta_HP: float; eta_LP: float; eta_exp: float; eta_gen: float
    P_live_bar: float; T_live_C: float; P_extraction_bar: float
    P_condenser_bar: float; T_feedwater_C: float

# ==========================================================================
# TESPy CYCLE SOLVER
# ==========================================================================
def analyse_cycle(
    P_live_bar=50.0, T_live_C=444.0, P_extraction_bar=2.5,
    P_condenser_bar=0.2, T_feedwater_C=130.0,
    eta_HP=0.80, eta_LP=0.78, eta_exp=0.75, eta_gen=0.94,
):
    nw = Network(iterinfo=False)
    nw.units.set_defaults(pressure="bar", temperature="degC",
        enthalpy="kJ / kg", entropy="kJ / kgK", mass_flow="kg / s")

    src = Source("src"); hp = Turbine("HP"); spl = Splitter("split")
    lp = Turbine("LP"); exp = Turbine("EXP"); snk_lp = Sink("snk_lp"); snk_exp = Sink("snk_exp")

    c1 = Connection(src,"out1",hp,"in1",label="c1")
    c2 = Connection(hp,"out1",spl,"in1",label="c2")
    c_lp = Connection(spl,"out1",lp,"in1",label="c_lp")
    c_exp = Connection(spl,"out2",exp,"in1",label="c_exp")
    c3 = Connection(lp,"out1",snk_lp,"in1",label="c3")
    c3e = Connection(exp,"out1",snk_exp,"in1",label="c3e")
    nw.add_conns(c1,c2,c_lp,c_exp,c3,c3e)

    c1.set_attr(p=P_live_bar, T=T_live_C, m=1.0, fluid={"water":1})
    hp.set_attr(eta_s=eta_HP); c2.set_attr(p=P_extraction_bar)
    c_lp.set_attr(m=0.5)
    lp.set_attr(eta_s=eta_LP); c3.set_attr(p=P_condenser_bar)
    exp.set_attr(eta_s=eta_exp); c3e.set_attr(p=P_condenser_bar)
    nw.solve("design")

    s1=_state_from_conn("1_LiveSteam",c1); s2=_state_from_conn("2_HP_Actual",c2)
    s3=_state_from_conn("3_LP_Actual",c3); s3e=_state_from_conn("3e_Expander",c3e)
    s4=_state_from_PT("4_Feedwater",P_live_bar,T_feedwater_C)
    s2s=_isentropic_state("2s_HP_Isentropic",P_extraction_bar,s1.s_kJkgK)
    s3s=_isentropic_state("3s_LP_Isentropic",P_condenser_bar,s2.s_kJkgK)

    w_HP=s1.h_kJkg-s2.h_kJkg; w_LP=s2.h_kJkg-s3.h_kJkg; w_exp=s2.h_kJkg-s3e.h_kJkg
    q_heat_direct=max(s2.h_kJkg-s4.h_kJkg,0.0)
    q_heat_post_expander=max(s3e.h_kJkg-s4.h_kJkg,0.0)

    return CycleResult(
        state1=s1,state2s=s2s,state2=s2,state3s=s3s,state3=s3,state3e=s3e,state4=s4,
        w_HP=w_HP,w_LP=w_LP,w_exp=w_exp,
        q_heat_direct=q_heat_direct,q_heat_post_expander=q_heat_post_expander,
        eta_HP=eta_HP,eta_LP=eta_LP,eta_exp=eta_exp,eta_gen=eta_gen,
        P_live_bar=P_live_bar,T_live_C=T_live_C,
        P_extraction_bar=P_extraction_bar,P_condenser_bar=P_condenser_bar,
        T_feedwater_C=T_feedwater_C)

# ==========================================================================
# PART-LOAD + AUXILIARY (IDENTICAL to original)
# ==========================================================================
def clamp(v, lo, hi): return max(lo, min(hi, v))

def part_load_corrected_eta(base_eta, load_ratio, section):
    load_ratio = clamp(load_ratio, 0.35, 1.15)
    if section.upper()=="HP":
        correction = clamp(1.0-0.16*(1.0-load_ratio), 0.90, 1.00)
    elif section.upper()=="LP":
        correction = clamp(1.0-0.13*(1.0-load_ratio), 0.88, 1.00)
    else:
        correction = clamp(1.0-0.10*(1.0-load_ratio), 0.90, 1.00)
    return base_eta * correction

@dataclass
class AuxiliaryBreakdown:
    base_misc_kW: float; steam_island_kW: float; biomass_handling_kW: float
    fg_fan_kW: float; condenser_fans_kW: float; heat_system_kW: float
    expander_skid_kW: float
    @property
    def total_kW(self):
        return (self.base_misc_kW + self.steam_island_kW + self.biomass_handling_kW
                + self.fg_fan_kW + self.condenser_fans_kW + self.heat_system_kW
                + self.expander_skid_kW)

def estimate_auxiliary_load_kW(
    steam_tph, extraction_tph, condensing_tph, biomass_tph,
    has_lp_expander, condenser_pressure_bar, summer_full_condenser_fans,
    fg_fan_limit_tph, fg_fan_motor_kW,
):
    """Identical to thermo_engine_improved.py — 7 components, calibrated to ~1.32 MW S1 net."""
    base_misc_kW = 115.0
    steam_island_kW = 4.0 * steam_tph
    biomass_handling_kW = 6.0 * biomass_tph
    fg_load_ratio = clamp(steam_tph / max(fg_fan_limit_tph, 1e-6), 0.0, 1.0)
    fg_fan_kW = fg_fan_motor_kW * (fg_load_ratio ** 3)
    if summer_full_condenser_fans:
        condenser_fans_kW = 90.0
    else:
        condenser_fans_kW = clamp(28.0 + 6.0*max(condensing_tph-6.0,0.0) + 140.0*max(condenser_pressure_bar-0.20,0.0), 25.0, 70.0)
    heat_system_kW = 15.0 + 3.0 * extraction_tph
    expander_skid_kW = 5.0 if has_lp_expander else 0.0
    return AuxiliaryBreakdown(base_misc_kW=base_misc_kW, steam_island_kW=steam_island_kW,
        biomass_handling_kW=biomass_handling_kW, fg_fan_kW=fg_fan_kW,
        condenser_fans_kW=condenser_fans_kW, heat_system_kW=heat_system_kW,
        expander_skid_kW=expander_skid_kW)

# ==========================================================================
# SCENARIO RESULT
# ==========================================================================
@dataclass
class ScenarioResult:
    sid: str; name: str; description: str
    steam_tph: float; extraction_tph: float; condensing_tph: float
    availability_pct: float; capex_kEUR: float; has_lp_expander: bool
    condenser_pressure_bar: float; summer_full_condenser_fans: bool
    eta_HP_eff: float; eta_LP_eff: float; eta_exp_eff: float
    P_HP: float; P_LP: float; P_exp: float
    P_gross: float; P_aux: float; P_net: float; P_export: float; P_curtailed: float
    hours_yr: int; annual_MWh_gross: int; annual_MWh_export: int
    w_HP_kJkg: float; w_LP_kJkg: float; w_exp_kJkg: float
    biomass_ratio: float; biomass_tph: float; biomass_tpy: float; Q_fuel_kW: float
    heat_output_kW: float; elec_eff_pct_gross: float; elec_eff_pct_net: float; chp_eff_pct: float
    aux_breakdown: AuxiliaryBreakdown
    cycle: CycleResult = field(repr=False)
    def to_dict(self):
        d = asdict(self); d["aux_breakdown_kW"] = d.pop("aux_breakdown"); d.pop("cycle",None); return d

# ==========================================================================
# SCENARIO ENGINE
# ==========================================================================
DEFAULT_DESIGN_STEAM_TPH = 13.0
DEFAULT_DESIGN_COND_TPH = 10.0
DEFAULT_DESIGN_EXP_TPH = 3.5

def compute_scenario(
    sid, name, description, steam_tph, extraction_tph, condensing_tph,
    availability_pct, capex_kEUR, has_lp_expander, cycle,
    transformer_limit_kW=2520.0, LHV_MJkg=10.5, eta_boiler=0.85,
    condenser_pressure_bar=None, summer_full_condenser_fans=False,
    fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    biomass_ratio_override=0.0,
):
    if abs((extraction_tph+condensing_tph)-steam_tph)>1e-6:
        raise ValueError(f"Scenario {sid}: extraction + condensing must equal total steam")
    P_cond = condenser_pressure_bar if condenser_pressure_bar is not None else cycle.P_condenser_bar
    eta_HP_eff = part_load_corrected_eta(cycle.eta_HP, steam_tph/DEFAULT_DESIGN_STEAM_TPH, "HP")
    eta_LP_eff = part_load_corrected_eta(cycle.eta_LP, max(condensing_tph,0.01)/DEFAULT_DESIGN_COND_TPH, "LP")
    exp_load = extraction_tph/DEFAULT_DESIGN_EXP_TPH if has_lp_expander and extraction_tph>0 else 0.0
    eta_exp_eff = part_load_corrected_eta(cycle.eta_exp, max(exp_load,0.35), "EXP") if has_lp_expander else cycle.eta_exp

    local_cycle = analyse_cycle(P_live_bar=cycle.P_live_bar, T_live_C=cycle.T_live_C,
        P_extraction_bar=cycle.P_extraction_bar, P_condenser_bar=P_cond,
        T_feedwater_C=cycle.T_feedwater_C,
        eta_HP=eta_HP_eff, eta_LP=eta_LP_eff, eta_exp=eta_exp_eff, eta_gen=cycle.eta_gen)

    eta_gen = local_cycle.eta_gen
    P_HP = steam_tph * local_cycle.w_HP * eta_gen / 3.6
    P_LP = condensing_tph * local_cycle.w_LP * eta_gen / 3.6
    P_exp = extraction_tph * local_cycle.w_exp * eta_gen / 3.6 if has_lp_expander and extraction_tph>0 else 0.0
    P_gross = P_HP + P_LP + P_exp

    h_gain = local_cycle.state1.h_kJkg - local_cycle.state4.h_kJkg
    biomass_ratio_thermo = h_gain / (eta_boiler * LHV_MJkg * 1000.0)
    biomass_ratio = biomass_ratio_override if biomass_ratio_override > 0.001 else biomass_ratio_thermo
    biomass_tph_val = steam_tph * biomass_ratio
    Q_fuel_kW = biomass_tph_val * LHV_MJkg * 1000.0 / 3.6

    aux = estimate_auxiliary_load_kW(
        steam_tph=steam_tph, extraction_tph=extraction_tph,
        condensing_tph=condensing_tph, biomass_tph=biomass_tph_val,
        has_lp_expander=has_lp_expander, condenser_pressure_bar=P_cond,
        summer_full_condenser_fans=summer_full_condenser_fans,
        fg_fan_limit_tph=fg_fan_limit_tph, fg_fan_motor_kW=fg_fan_motor_kW)

    P_net = max(P_gross - aux.total_kW, 0.0)
    P_export = min(P_net, transformer_limit_kW)
    P_curtailed = max(P_net - transformer_limit_kW, 0.0)
    hours_yr = round(8760 * availability_pct / 100.0)
    annual_MWh_gross = round(P_gross * hours_yr / 1000.0)
    annual_MWh_export = round(P_export * hours_yr / 1000.0)
    biomass_tpy = round(biomass_tph_val * hours_yr)
    q_heat = local_cycle.q_heat_post_expander if has_lp_expander else local_cycle.q_heat_direct
    heat_output_kW = extraction_tph * q_heat / 3.6
    elec_eff_pct_gross = (P_gross/Q_fuel_kW*100.0) if Q_fuel_kW>0 else 0.0
    elec_eff_pct_net = (P_export/Q_fuel_kW*100.0) if Q_fuel_kW>0 else 0.0
    chp_eff_pct = ((P_export+heat_output_kW)/Q_fuel_kW*100.0) if Q_fuel_kW>0 else 0.0

    return ScenarioResult(
        sid=sid, name=name, description=description,
        steam_tph=steam_tph, extraction_tph=extraction_tph,
        condensing_tph=condensing_tph, availability_pct=availability_pct,
        capex_kEUR=capex_kEUR, has_lp_expander=has_lp_expander,
        condenser_pressure_bar=P_cond, summer_full_condenser_fans=summer_full_condenser_fans,
        eta_HP_eff=round(eta_HP_eff,4), eta_LP_eff=round(eta_LP_eff,4), eta_exp_eff=round(eta_exp_eff,4),
        P_HP=round(P_HP,1), P_LP=round(P_LP,1), P_exp=round(P_exp,1),
        P_gross=round(P_gross,1), P_aux=round(aux.total_kW,1),
        P_net=round(P_net,1), P_export=round(P_export,1), P_curtailed=round(P_curtailed,1),
        hours_yr=hours_yr, annual_MWh_gross=annual_MWh_gross, annual_MWh_export=annual_MWh_export,
        w_HP_kJkg=round(local_cycle.w_HP,1), w_LP_kJkg=round(local_cycle.w_LP,1),
        w_exp_kJkg=round(local_cycle.w_exp,1),
        biomass_ratio=round(biomass_ratio,4), biomass_tph=round(biomass_tph_val,2),
        biomass_tpy=biomass_tpy, Q_fuel_kW=round(Q_fuel_kW),
        heat_output_kW=round(heat_output_kW),
        elec_eff_pct_gross=round(elec_eff_pct_gross,1),
        elec_eff_pct_net=round(elec_eff_pct_net,1), chp_eff_pct=round(chp_eff_pct,1),
        aux_breakdown=aux, cycle=local_cycle)

# ==========================================================================
# SCENARIO DEFINITIONS (identical to original)
# ==========================================================================
SCENARIO_DEFS = [
    dict(sid="S1",name="Current Baseline",steam_tph=9.7,extraction_tph=3.5,condensing_tph=6.2,
         availability_pct=91.8,capex_kEUR=0,has_lp_expander=False,
         description="Measured SCADA baseline. No modifications. FG fan at 84%, filter at 91%.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="S2",name="Optimised CHP",steam_tph=10.5,extraction_tph=3.0,condensing_tph=7.5,
         availability_pct=92.0,capex_kEUR=20,has_lp_expander=False,
         description="FGR retuning + O2 optimisation (5.1->4.0%) + extraction reduced 3.5->3.0 t/h.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="S3",name="Maximum Condensing",steam_tph=10.5,extraction_tph=0.0,condensing_tph=10.5,
         availability_pct=92.0,capex_kEUR=0,has_lp_expander=False,
         description="All steam to condenser. Zero extraction. Propane backup for factory heat.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="S4",name="Summer Mode",steam_tph=10.4,extraction_tph=2.0,condensing_tph=8.4,
         availability_pct=90.0,capex_kEUR=0,has_lp_expander=False,
         description="Condenser-limited at >25C ambient. Max 10.4 t/h. All 5 condenser fans at full speed.",
         condenser_pressure_bar=0.28,summer_full_condenser_fans=True,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="S5",name="Hybrid Dispatch",steam_tph=10.5,extraction_tph=2.0,condensing_tph=8.5,
         availability_pct=93.0,capex_kEUR=80,has_lp_expander=False,
         description="Price-responsive EPEX NL switching via DCS upgrade. Annual weighted averages.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="S6",name="Filter Debottleneck",steam_tph=12.0,extraction_tph=3.0,condensing_tph=9.0,
         availability_pct=93.0,capex_kEUR=200,has_lp_expander=False,
         description="Bag filter expanded to 40,000 Nm3/h. Removes first FG bottleneck -> 12 t/h.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=12.0,fg_fan_motor_kW=110.0),
    dict(sid="S7",name="Full Debottleneck",steam_tph=13.0,extraction_tph=3.0,condensing_tph=10.0,
         availability_pct=93.0,capex_kEUR=400,has_lp_expander=False,
         description="Both filter AND fan to 40,000 Nm3/h. Full 13 t/h boiler design capacity.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=13.0,fg_fan_motor_kW=150.0),
    dict(sid="S8",name="LP Steam Expander",steam_tph=10.5,extraction_tph=3.5,condensing_tph=7.0,
         availability_pct=92.0,capex_kEUR=300,has_lp_expander=True,
         description="Heliex Power screw expander on extraction steam (2.5->0.2 bar). No extra fuel.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
    dict(sid="C1",name="Filter + LP Expander",steam_tph=12.0,extraction_tph=1.0,condensing_tph=11.0,
         availability_pct=93.0,capex_kEUR=500,has_lp_expander=True,
         description="S6 + S8 stacked. Filter enables 12 t/h. Expander on reduced extraction.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=12.0,fg_fan_motor_kW=110.0),
    dict(sid="C2",name="Full + LP Expander",steam_tph=13.0,extraction_tph=1.0,condensing_tph=12.0,
         availability_pct=93.0,capex_kEUR=700,has_lp_expander=True,
         description="S7 + S8 stacked. Maximum power within transformer limit.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=13.0,fg_fan_motor_kW=150.0),
    dict(sid="C3",name="Low CAPEX Combo",steam_tph=10.5,extraction_tph=2.0,condensing_tph=8.5,
         availability_pct=93.0,capex_kEUR=100,has_lp_expander=False,
         description="S2 + S5 combined. Operational optimisation + automated dispatch.",
         condenser_pressure_bar=0.20,summer_full_condenser_fans=False,fg_fan_limit_tph=11.6,fg_fan_motor_kW=110.0),
]

def run_all_scenarios(cycle=None, **kw):
    if cycle is None: cycle = analyse_cycle(**kw)
    return [compute_scenario(cycle=cycle, **sd) for sd in SCENARIO_DEFS]

if __name__ == "__main__":
    cycle = analyse_cycle()
    results = run_all_scenarios(cycle)
    bl = results[0]
    print("="*90)
    for r in results:
        print(f"{r.sid:>2} | Gross {r.P_gross:7.1f} kW | Aux {r.P_aux:6.1f} kW | "
              f"Net {r.P_net:7.1f} kW | Export {r.P_export:7.1f} kW | "
              f"dP vs S1 {r.P_export-bl.P_export:+7.1f} kW")
