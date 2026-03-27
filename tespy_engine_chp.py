from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from tespy.components import Condenser, SimpleHeatExchanger, Sink, Source, Splitter, Turbine
from tespy.connections import Connection
from tespy.networks import Network


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SteamState:
    name: str
    P_bar: float
    T_C: Optional[float] = None
    h_kJkg: Optional[float] = None
    s_kJkgK: Optional[float] = None
    x: Optional[float] = None
    phase: str = ""


@dataclass
class CycleResult:
    state1: SteamState
    state2s: SteamState
    state2: SteamState
    state3s: SteamState
    state3: SteamState
    state3e: SteamState
    state4: SteamState
    w_HP: float
    w_LP: float
    w_exp: float
    q_heat_direct: float
    q_heat_post_expander: float
    eta_HP: float
    eta_LP: float
    eta_exp: float
    eta_gen: float
    P_live_bar: float
    T_live_C: float
    P_extraction_bar: float
    P_condenser_bar: float
    T_feedwater_C: float


@dataclass
class AuxiliaryBreakdown:
    base_misc_kW: float
    steam_island_kW: float
    biomass_handling_kW: float
    fg_fan_kW: float
    condenser_fans_kW: float
    heat_system_kW: float
    expander_skid_kW: float

    @property
    def total_kW(self) -> float:
        return (
            self.base_misc_kW
            + self.steam_island_kW
            + self.biomass_handling_kW
            + self.fg_fan_kW
            + self.condenser_fans_kW
            + self.heat_system_kW
            + self.expander_skid_kW
        )


@dataclass
class ScenarioResult:
    sid: str
    name: str
    description: str
    steam_tph: float
    extraction_tph: float
    condensing_tph: float
    availability_pct: float
    capex_kEUR: float
    has_lp_expander: bool
    condenser_pressure_bar: float
    summer_full_condenser_fans: bool
    eta_HP_eff: float
    eta_LP_eff: float
    eta_exp_eff: float
    P_HP: float
    P_LP: float
    P_exp: float
    P_gross: float
    P_aux: float
    P_net: float
    P_export: float
    P_curtailed: float
    hours_yr: int
    annual_MWh_gross: int
    annual_MWh_export: int
    w_HP_kJkg: float
    w_LP_kJkg: float
    w_exp_kJkg: float
    biomass_ratio: float
    biomass_tph: float
    biomass_tpy: float
    Q_fuel_kW: float
    heat_output_kW: float
    elec_eff_pct_gross: float
    elec_eff_pct_net: float
    chp_eff_pct: float
    aux_breakdown: AuxiliaryBreakdown
    cycle: CycleResult = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["aux_breakdown_kW"] = d.pop("aux_breakdown")
        d.pop("cycle", None)
        return d


@dataclass
class NativeSolveResult:
    state_live: SteamState
    state_hp_out: SteamState
    state_lp_out: SteamState
    state_process_out: Optional[SteamState]
    state_exp_out: Optional[SteamState]
    state_cond_out: Optional[SteamState]
    P_HP_kW: float
    P_LP_kW: float
    P_exp_kW: float
    Q_main_cond_kW: float
    Q_process_kW: float
    m_total_kg_s: float
    m_extraction_kg_s: float
    m_condensing_kg_s: float


# =============================================================================
# CONSTANTS
# =============================================================================


DEFAULT_DESIGN_STEAM_TPH = 13.0
DEFAULT_DESIGN_COND_TPH = 10.0
DEFAULT_DESIGN_EXP_TPH = 3.5

CONDENSER_PR = 0.99
PROCESS_HEX_PR = 0.99


# =============================================================================
# TESPY HELPERS
# =============================================================================


def _make_network() -> Network:
    nw = Network(iterinfo=False)
    nw.units.set_defaults(
        pressure="bar",
        temperature="degC",
        enthalpy="kJ / kg",
        mass_flow="kg / s",
    )
    return nw


def _phase_from_quality(x: float) -> str:
    if x != x:
        return "single-phase / superheated"
    if 0.0 <= x <= 1.0:
        if abs(x) < 1e-6:
            return "saturated liquid"
        if abs(x - 1.0) < 1e-6:
            return "saturated vapor"
        return f"two-phase (x={x:.3f})"
    return "single-phase / superheated"



def _state_from_conn(name: str, conn: Connection) -> SteamState:
    x_val = float(conn.x.val)
    x: Optional[float] = None
    if x_val == x_val and 0.0 <= x_val <= 1.0:
        x = x_val

    return SteamState(
        name=name,
        P_bar=float(conn.p.val),
        T_C=float(conn.T.val),
        h_kJkg=float(conn.h.val),
        s_kJkgK=float(conn.s.val) / 1000.0,
        x=x,
        phase=_phase_from_quality(x_val),
    )


@lru_cache(maxsize=256)
def _solve_reference_state(P_bar: float, T_C: float) -> SteamState:
    nw = _make_network()
    src = Source("reference source")
    snk = Sink("reference sink")
    c = Connection(src, "out1", snk, "in1", label="ref")
    nw.add_conns(c)
    c.set_attr(m=1.0, p=P_bar, T=T_C, fluid={"water": 1})
    nw.solve("design")
    if nw.status != 0:
        raise RuntimeError("TESPy could not solve the feedwater reference state.")
    return _state_from_conn("4_Feedwater", c)


@lru_cache(maxsize=512)
def _solve_turbine_stage(
    name: str,
    P_in_bar: float,
    P_out_bar: float,
    eta_s: float,
    T_in_C: Optional[float] = None,
    h_in_kJkg: Optional[float] = None,
) -> Tuple[SteamState, SteamState, float]:
    nw = _make_network()
    src = Source(f"{name} source")
    tur = Turbine(name)
    snk = Sink(f"{name} sink")

    c1 = Connection(src, "out1", tur, "in1", label=f"{name}_in")
    c2 = Connection(tur, "out1", snk, "in1", label=f"{name}_out")
    nw.add_conns(c1, c2)

    attrs: Dict[str, Any] = {"m": 1.0, "p": P_in_bar, "fluid": {"water": 1}}
    if T_in_C is not None:
        attrs["T"] = T_in_C
    if h_in_kJkg is not None:
        attrs["h"] = h_in_kJkg

    tur.set_attr(eta_s=eta_s)
    c1.set_attr(**attrs)
    c2.set_attr(p=P_out_bar)

    nw.solve("design")
    if nw.status != 0:
        raise RuntimeError(f"TESPy could not solve turbine stage '{name}'.")

    inlet = _state_from_conn(f"{name}_inlet", c1)
    outlet = _state_from_conn(f"{name}_outlet", c2)
    shaft_power_kW = abs(float(tur.P.val)) / 1000.0
    return inlet, outlet, shaft_power_kW


@lru_cache(maxsize=512)
def _solve_condensing_only_network(
    steam_tph: float,
    P_live_bar: float,
    T_live_C: float,
    P_extraction_bar: float,
    P_condenser_bar: float,
    eta_HP: float,
    eta_LP: float,
    cooling_inlet_T_C: float = 20.0,
    cooling_outlet_T_C: float = 28.0,
    cooling_inlet_p_bar: float = 3.0,
) -> NativeSolveResult:
    m_total = steam_tph * 1000.0 / 3600.0

    nw = _make_network()

    live = Source("live steam")
    hp = Turbine("hp turbine")
    lp = Turbine("lp turbine")
    cond = Condenser("main condenser")
    cond_sink = Sink("condensate sink")
    cw_src = Source("cooling water source")
    cw_sink = Sink("cooling water sink")

    c1 = Connection(live, "out1", hp, "in1", label="1")
    c2 = Connection(hp, "out1", lp, "in1", label="2")
    c3 = Connection(lp, "out1", cond, "in1", label="3")
    c4 = Connection(cond, "out1", cond_sink, "in1", label="4")
    cw1 = Connection(cw_src, "out1", cond, "in2", label="cw1")
    cw2 = Connection(cond, "out2", cw_sink, "in1", label="cw2")

    nw.add_conns(c1, c2, c3, c4, cw1, cw2)

    hp.set_attr(eta_s=eta_HP)
    lp.set_attr(eta_s=eta_LP)
    cond.set_attr(pr1=CONDENSER_PR, pr2=0.99)

    c1.set_attr(m=m_total, p=P_live_bar, T=T_live_C, fluid={"water": 1})
    c2.set_attr(p=P_extraction_bar)
    c3.set_attr(p=P_condenser_bar)
    cw1.set_attr(T=cooling_inlet_T_C, p=cooling_inlet_p_bar, fluid={"water": 1})
    cw2.set_attr(T=cooling_outlet_T_C)

    nw.solve("design")
    if nw.status != 0:
        raise RuntimeError("TESPy could not solve the condensing-only CHP network.")

    return NativeSolveResult(
        state_live=_state_from_conn("1_LiveSteam", c1),
        state_hp_out=_state_from_conn("2_HP_Actual", c2),
        state_lp_out=_state_from_conn("3_LP_Actual", c3),
        state_process_out=None,
        state_exp_out=None,
        state_cond_out=_state_from_conn("4_Condensate", c4),
        P_HP_kW=abs(float(hp.P.val)) / 1000.0,
        P_LP_kW=abs(float(lp.P.val)) / 1000.0,
        P_exp_kW=0.0,
        Q_main_cond_kW=abs(float(cond.Q.val)) / 1000.0,
        Q_process_kW=0.0,
        m_total_kg_s=float(c1.m.val),
        m_extraction_kg_s=0.0,
        m_condensing_kg_s=float(c3.m.val),
    )


@lru_cache(maxsize=1024)
def _solve_split_network(
    steam_tph: float,
    extraction_tph: float,
    condensing_tph: float,
    P_live_bar: float,
    T_live_C: float,
    P_extraction_bar: float,
    P_condenser_bar: float,
    eta_HP: float,
    eta_LP: float,
    eta_exp: float,
    has_lp_expander: bool,
    cooling_inlet_T_C: float = 20.0,
    cooling_outlet_T_C: float = 28.0,
    cooling_inlet_p_bar: float = 3.0,
) -> NativeSolveResult:
    if abs((extraction_tph + condensing_tph) - steam_tph) > 1e-9:
        raise ValueError("Extraction plus condensing flow must equal total steam flow.")

    if extraction_tph <= 1e-9:
        return _solve_condensing_only_network(
            steam_tph=steam_tph,
            P_live_bar=P_live_bar,
            T_live_C=T_live_C,
            P_extraction_bar=P_extraction_bar,
            P_condenser_bar=P_condenser_bar,
            eta_HP=eta_HP,
            eta_LP=eta_LP,
            cooling_inlet_T_C=cooling_inlet_T_C,
            cooling_outlet_T_C=cooling_outlet_T_C,
            cooling_inlet_p_bar=cooling_inlet_p_bar,
        )

    m_total = steam_tph * 1000.0 / 3600.0
    m_ext = extraction_tph * 1000.0 / 3600.0

    nw = _make_network()

    live = Source("live steam")
    hp = Turbine("hp turbine")
    split = Splitter("extraction split", num_out=2)
    lp = Turbine("lp turbine")
    main_cond = Condenser("main condenser")
    cond_sink = Sink("condensate sink")
    cw_src = Source("cooling water source")
    cw_sink = Sink("cooling water sink")
    proc_hex = SimpleHeatExchanger("process heat exchanger")
    proc_sink = Sink("process condensate sink")

    c1 = Connection(live, "out1", hp, "in1", label="1")
    c2 = Connection(hp, "out1", split, "in1", label="2")
    c4 = Connection(split, "out2", lp, "in1", label="4")
    c5 = Connection(lp, "out1", main_cond, "in1", label="5")
    c6 = Connection(main_cond, "out1", cond_sink, "in1", label="6")
    cw1 = Connection(cw_src, "out1", main_cond, "in2", label="cw1")
    cw2 = Connection(main_cond, "out2", cw_sink, "in1", label="cw2")

    nw.add_conns(c1, c2, c4, c5, c6, cw1, cw2)

    hp.set_attr(eta_s=eta_HP)
    lp.set_attr(eta_s=eta_LP)
    main_cond.set_attr(pr1=CONDENSER_PR, pr2=0.99)
    proc_hex.set_attr(pr=PROCESS_HEX_PR)

    if has_lp_expander:
        exp = Turbine("lp expander")
        c3 = Connection(split, "out1", exp, "in1", label="3")
        c3e = Connection(exp, "out1", proc_hex, "in1", label="3e")
        c7 = Connection(proc_hex, "out1", proc_sink, "in1", label="7")
        nw.add_conns(c3, c3e, c7)
        exp.set_attr(eta_s=eta_exp)
        c3.set_attr(m=m_ext)
        c3e.set_attr(p=P_condenser_bar)
    else:
        exp = None
        c3 = Connection(split, "out1", proc_hex, "in1", label="3")
        c7 = Connection(proc_hex, "out1", proc_sink, "in1", label="7")
        nw.add_conns(c3, c7)
        c3.set_attr(m=m_ext)

    c1.set_attr(m=m_total, p=P_live_bar, T=T_live_C, fluid={"water": 1})
    c2.set_attr(p=P_extraction_bar)
    c5.set_attr(p=P_condenser_bar)
    c7.set_attr(x=0)
    cw1.set_attr(T=cooling_inlet_T_C, p=cooling_inlet_p_bar, fluid={"water": 1})
    cw2.set_attr(T=cooling_outlet_T_C)

    nw.solve("design")
    if nw.status != 0:
        raise RuntimeError("TESPy could not solve the split CHP network.")

    return NativeSolveResult(
        state_live=_state_from_conn("1_LiveSteam", c1),
        state_hp_out=_state_from_conn("2_HP_Actual", c2),
        state_lp_out=_state_from_conn("5_LP_Actual", c5),
        state_process_out=_state_from_conn("7_ProcessCondensate", c7),
        state_exp_out=_state_from_conn("3e_Expander", c3e) if has_lp_expander else None,
        state_cond_out=_state_from_conn("6_Condensate", c6),
        P_HP_kW=abs(float(hp.P.val)) / 1000.0,
        P_LP_kW=abs(float(lp.P.val)) / 1000.0,
        P_exp_kW=abs(float(exp.P.val)) / 1000.0 if exp is not None else 0.0,
        Q_main_cond_kW=abs(float(main_cond.Q.val)) / 1000.0,
        Q_process_kW=abs(float(proc_hex.Q.val)) / 1000.0,
        m_total_kg_s=float(c1.m.val),
        m_extraction_kg_s=float(c3.m.val),
        m_condensing_kg_s=float(c5.m.val),
    )


# =============================================================================
# CYCLE ANALYSIS
# =============================================================================


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))



def analyse_cycle(
    P_live_bar: float = 50.0,
    T_live_C: float = 444.0,
    P_extraction_bar: float = 2.5,
    P_condenser_bar: float = 0.2,
    T_feedwater_C: float = 130.0,
    eta_HP: float = 0.80,
    eta_LP: float = 0.78,
    eta_exp: float = 0.75,
    eta_gen: float = 0.94,
) -> CycleResult:
    _, s2 = _solve_turbine_stage(
        name="hp_actual",
        P_in_bar=P_live_bar,
        P_out_bar=P_extraction_bar,
        eta_s=eta_HP,
        T_in_C=T_live_C,
    )[:2]
    s1, s2s = _solve_turbine_stage(
        name="hp_isentropic",
        P_in_bar=P_live_bar,
        P_out_bar=P_extraction_bar,
        eta_s=1.0,
        T_in_C=T_live_C,
    )[:2]
    _, s3 = _solve_turbine_stage(
        name="lp_actual",
        P_in_bar=s2.P_bar,
        P_out_bar=P_condenser_bar,
        eta_s=eta_LP,
        h_in_kJkg=s2.h_kJkg,
    )[:2]
    _, s3s = _solve_turbine_stage(
        name="lp_isentropic",
        P_in_bar=s2.P_bar,
        P_out_bar=P_condenser_bar,
        eta_s=1.0,
        h_in_kJkg=s2.h_kJkg,
    )[:2]
    _, s3e = _solve_turbine_stage(
        name="exp_actual",
        P_in_bar=s2.P_bar,
        P_out_bar=P_condenser_bar,
        eta_s=eta_exp,
        h_in_kJkg=s2.h_kJkg,
    )[:2]
    s4 = _solve_reference_state(P_live_bar, T_feedwater_C)

    w_HP = s1.h_kJkg - s2.h_kJkg
    w_LP = s2.h_kJkg - s3.h_kJkg
    w_exp = s2.h_kJkg - s3e.h_kJkg
    q_heat_direct = max(s2.h_kJkg - s4.h_kJkg, 0.0)
    q_heat_post_expander = max(s3e.h_kJkg - s4.h_kJkg, 0.0)

    return CycleResult(
        state1=s1,
        state2s=s2s,
        state2=s2,
        state3s=s3s,
        state3=s3,
        state3e=s3e,
        state4=s4,
        w_HP=w_HP,
        w_LP=w_LP,
        w_exp=w_exp,
        q_heat_direct=q_heat_direct,
        q_heat_post_expander=q_heat_post_expander,
        eta_HP=eta_HP,
        eta_LP=eta_LP,
        eta_exp=eta_exp,
        eta_gen=eta_gen,
        P_live_bar=P_live_bar,
        T_live_C=T_live_C,
        P_extraction_bar=P_extraction_bar,
        P_condenser_bar=P_condenser_bar,
        T_feedwater_C=T_feedwater_C,
    )


# =============================================================================
# PART-LOAD / AUXILIARY MODEL
# =============================================================================


def part_load_corrected_eta(base_eta: float, load_ratio: float, section: str) -> float:
    load_ratio = clamp(load_ratio, 0.35, 1.15)
    if section.upper() == "HP":
        correction = clamp(1.0 - 0.16 * (1.0 - load_ratio), 0.90, 1.00)
    elif section.upper() == "LP":
        correction = clamp(1.0 - 0.13 * (1.0 - load_ratio), 0.88, 1.00)
    else:
        correction = clamp(1.0 - 0.10 * (1.0 - load_ratio), 0.90, 1.00)
    return base_eta * correction



def estimate_auxiliary_load_kW(
    steam_tph: float,
    extraction_tph: float,
    condensing_tph: float,
    biomass_tph: float,
    has_lp_expander: bool,
    condenser_pressure_bar: float,
    summer_full_condenser_fans: bool,
    fg_fan_limit_tph: float,
    fg_fan_motor_kW: float,
) -> AuxiliaryBreakdown:
    base_misc_kW = 115.0
    steam_island_kW = 4.0 * steam_tph
    biomass_handling_kW = 6.0 * biomass_tph

    fg_load_ratio = clamp(steam_tph / max(fg_fan_limit_tph, 1e-6), 0.0, 1.0)
    fg_fan_kW = fg_fan_motor_kW * (fg_load_ratio ** 3)

    if summer_full_condenser_fans:
        condenser_fans_kW = 90.0
    else:
        condenser_fans_kW = clamp(
            28.0 + 6.0 * max(condensing_tph - 6.0, 0.0) + 140.0 * max(condenser_pressure_bar - 0.20, 0.0),
            25.0,
            70.0,
        )

    heat_system_kW = 15.0 + 3.0 * extraction_tph
    expander_skid_kW = 5.0 if has_lp_expander else 0.0

    return AuxiliaryBreakdown(
        base_misc_kW=base_misc_kW,
        steam_island_kW=steam_island_kW,
        biomass_handling_kW=biomass_handling_kW,
        fg_fan_kW=fg_fan_kW,
        condenser_fans_kW=condenser_fans_kW,
        heat_system_kW=heat_system_kW,
        expander_skid_kW=expander_skid_kW,
    )


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================


SCENARIO_DEFS: List[Dict[str, Any]] = [
    dict(
        sid="S1", name="Current Baseline",
        steam_tph=9.7, extraction_tph=3.5, condensing_tph=6.2,
        availability_pct=91.8, capex_kEUR=0, has_lp_expander=False,
        description="Measured SCADA baseline. No modifications. FG fan at 84%, filter at 91%.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S2", name="Optimised CHP",
        steam_tph=10.5, extraction_tph=3.0, condensing_tph=7.5,
        availability_pct=92.0, capex_kEUR=20, has_lp_expander=False,
        description="FGR retuning + O₂ optimisation (5.1→4.0%) + extraction reduced 3.5→3.0 t/h.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S3", name="Maximum Condensing",
        steam_tph=10.5, extraction_tph=0.0, condensing_tph=10.5,
        availability_pct=92.0, capex_kEUR=0, has_lp_expander=False,
        description="All steam to condenser. Zero extraction. Propane backup for factory heat.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S4", name="Summer Mode",
        steam_tph=10.4, extraction_tph=2.0, condensing_tph=8.4,
        availability_pct=90.0, capex_kEUR=0, has_lp_expander=False,
        description="Condenser-limited at >25°C ambient. Max 10.4 t/h. All 5 condenser fans at full speed.",
        condenser_pressure_bar=0.28,
        summer_full_condenser_fans=True,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S5", name="Hybrid Dispatch",
        steam_tph=10.5, extraction_tph=2.0, condensing_tph=8.5,
        availability_pct=93.0, capex_kEUR=80, has_lp_expander=False,
        description="Price-responsive EPEX NL switching via DCS upgrade. Annual weighted averages.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S6", name="Filter Debottleneck",
        steam_tph=12.0, extraction_tph=3.0, condensing_tph=9.0,
        availability_pct=93.0, capex_kEUR=200, has_lp_expander=False,
        description="Bag filter expanded to 40,000 Nm³/h. Removes first FG bottleneck → 12 t/h.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=12.0, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="S7", name="Full Debottleneck",
        steam_tph=13.0, extraction_tph=3.0, condensing_tph=10.0,
        availability_pct=93.0, capex_kEUR=400, has_lp_expander=False,
        description="Both filter AND fan to 40,000 Nm³/h. Full 13 t/h boiler design capacity.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=13.0, fg_fan_motor_kW=150.0,
    ),
    dict(
        sid="S8", name="LP Steam Expander",
        steam_tph=10.5, extraction_tph=3.5, condensing_tph=7.0,
        availability_pct=92.0, capex_kEUR=300, has_lp_expander=True,
        description="Heliex Power screw expander on extraction steam (2.5→0.2 bar). No extra fuel.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="C1", name="Filter + LP Expander",
        steam_tph=12.0, extraction_tph=1.0, condensing_tph=11.0,
        availability_pct=93.0, capex_kEUR=500, has_lp_expander=True,
        description="S6 + S8 stacked. Filter enables 12 t/h. Expander on reduced extraction.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=12.0, fg_fan_motor_kW=110.0,
    ),
    dict(
        sid="C2", name="Full + LP Expander",
        steam_tph=13.0, extraction_tph=1.0, condensing_tph=12.0,
        availability_pct=93.0, capex_kEUR=700, has_lp_expander=True,
        description="S7 + S8 stacked. Maximum power within transformer limit.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=13.0, fg_fan_motor_kW=150.0,
    ),
    dict(
        sid="C3", name="Low CAPEX Combo",
        steam_tph=10.5, extraction_tph=2.0, condensing_tph=8.5,
        availability_pct=93.0, capex_kEUR=100, has_lp_expander=False,
        description="S2 + S5 combined. Operational optimisation + automated dispatch.",
        condenser_pressure_bar=0.20, summer_full_condenser_fans=False,
        fg_fan_limit_tph=11.6, fg_fan_motor_kW=110.0,
    ),
]


# =============================================================================
# SCENARIO ENGINE
# =============================================================================


def compute_scenario(
    sid: str,
    name: str,
    description: str,
    steam_tph: float,
    extraction_tph: float,
    condensing_tph: float,
    availability_pct: float,
    capex_kEUR: float,
    has_lp_expander: bool,
    cycle: CycleResult,
    transformer_limit_kW: float = 2520.0,
    LHV_MJkg: float = 10.5,
    eta_boiler: float = 0.85,
    condenser_pressure_bar: Optional[float] = None,
    summer_full_condenser_fans: bool = False,
    fg_fan_limit_tph: float = 11.6,
    fg_fan_motor_kW: float = 110.0,
) -> ScenarioResult:
    if abs((extraction_tph + condensing_tph) - steam_tph) > 1e-6:
        raise ValueError(f"Scenario {sid}: extraction + condensing must equal total steam")

    P_cond = condenser_pressure_bar if condenser_pressure_bar is not None else cycle.P_condenser_bar

    eta_HP_eff = part_load_corrected_eta(cycle.eta_HP, steam_tph / DEFAULT_DESIGN_STEAM_TPH, "HP")
    eta_LP_eff = part_load_corrected_eta(cycle.eta_LP, max(condensing_tph, 0.01) / DEFAULT_DESIGN_COND_TPH, "LP")
    exp_load = extraction_tph / DEFAULT_DESIGN_EXP_TPH if has_lp_expander and extraction_tph > 0 else 0.0
    eta_exp_eff = (
        part_load_corrected_eta(cycle.eta_exp, max(exp_load, 0.35), "EXP")
        if has_lp_expander
        else cycle.eta_exp
    )

    local_cycle = analyse_cycle(
        P_live_bar=cycle.P_live_bar,
        T_live_C=cycle.T_live_C,
        P_extraction_bar=cycle.P_extraction_bar,
        P_condenser_bar=P_cond,
        T_feedwater_C=cycle.T_feedwater_C,
        eta_HP=eta_HP_eff,
        eta_LP=eta_LP_eff,
        eta_exp=eta_exp_eff,
        eta_gen=cycle.eta_gen,
    )

    solved_network = _solve_split_network(
        steam_tph=steam_tph,
        extraction_tph=extraction_tph,
        condensing_tph=condensing_tph,
        P_live_bar=cycle.P_live_bar,
        T_live_C=cycle.T_live_C,
        P_extraction_bar=cycle.P_extraction_bar,
        P_condenser_bar=P_cond,
        eta_HP=eta_HP_eff,
        eta_LP=eta_LP_eff,
        eta_exp=eta_exp_eff,
        has_lp_expander=has_lp_expander,
        cooling_outlet_T_C=28.0,
    )

    eta_gen = local_cycle.eta_gen
    P_HP = solved_network.P_HP_kW * eta_gen
    P_LP = solved_network.P_LP_kW * eta_gen
    P_exp = solved_network.P_exp_kW * eta_gen if has_lp_expander else 0.0
    P_gross = P_HP + P_LP + P_exp

    h_gain = local_cycle.state1.h_kJkg - local_cycle.state4.h_kJkg
    biomass_ratio = h_gain / (eta_boiler * LHV_MJkg * 1000.0)
    biomass_tph = steam_tph * biomass_ratio
    Q_fuel_kW = biomass_tph * LHV_MJkg * 1000.0 / 3.6

    aux = estimate_auxiliary_load_kW(
        steam_tph=steam_tph,
        extraction_tph=extraction_tph,
        condensing_tph=condensing_tph,
        biomass_tph=biomass_tph,
        has_lp_expander=has_lp_expander,
        condenser_pressure_bar=P_cond,
        summer_full_condenser_fans=summer_full_condenser_fans,
        fg_fan_limit_tph=fg_fan_limit_tph,
        fg_fan_motor_kW=fg_fan_motor_kW,
    )

    P_net = max(P_gross - aux.total_kW, 0.0)
    P_export = min(P_net, transformer_limit_kW)
    P_curtailed = max(P_net - transformer_limit_kW, 0.0)

    hours_yr = round(8760 * availability_pct / 100.0)
    annual_MWh_gross = round(P_gross * hours_yr / 1000.0)
    annual_MWh_export = round(P_export * hours_yr / 1000.0)
    biomass_tpy = round(biomass_tph * hours_yr)

    heat_output_kW = solved_network.Q_process_kW

    elec_eff_pct_gross = (P_gross / Q_fuel_kW * 100.0) if Q_fuel_kW > 0 else 0.0
    elec_eff_pct_net = (P_export / Q_fuel_kW * 100.0) if Q_fuel_kW > 0 else 0.0
    chp_eff_pct = ((P_export + heat_output_kW) / Q_fuel_kW * 100.0) if Q_fuel_kW > 0 else 0.0

    return ScenarioResult(
        sid=sid,
        name=name,
        description=description,
        steam_tph=steam_tph,
        extraction_tph=extraction_tph,
        condensing_tph=condensing_tph,
        availability_pct=availability_pct,
        capex_kEUR=capex_kEUR,
        has_lp_expander=has_lp_expander,
        condenser_pressure_bar=P_cond,
        summer_full_condenser_fans=summer_full_condenser_fans,
        eta_HP_eff=round(eta_HP_eff, 4),
        eta_LP_eff=round(eta_LP_eff, 4),
        eta_exp_eff=round(eta_exp_eff, 4),
        P_HP=round(P_HP, 1),
        P_LP=round(P_LP, 1),
        P_exp=round(P_exp, 1),
        P_gross=round(P_gross, 1),
        P_aux=round(aux.total_kW, 1),
        P_net=round(P_net, 1),
        P_export=round(P_export, 1),
        P_curtailed=round(P_curtailed, 1),
        hours_yr=hours_yr,
        annual_MWh_gross=annual_MWh_gross,
        annual_MWh_export=annual_MWh_export,
        w_HP_kJkg=round(local_cycle.w_HP, 1),
        w_LP_kJkg=round(local_cycle.w_LP, 1),
        w_exp_kJkg=round(local_cycle.w_exp, 1),
        biomass_ratio=round(biomass_ratio, 4),
        biomass_tph=round(biomass_tph, 2),
        biomass_tpy=biomass_tpy,
        Q_fuel_kW=round(Q_fuel_kW),
        heat_output_kW=round(heat_output_kW),
        elec_eff_pct_gross=round(elec_eff_pct_gross, 1),
        elec_eff_pct_net=round(elec_eff_pct_net, 1),
        chp_eff_pct=round(chp_eff_pct, 1),
        aux_breakdown=aux,
        cycle=local_cycle,
    )



def run_all_scenarios(cycle: Optional[CycleResult] = None, **cycle_kwargs: Any) -> List[ScenarioResult]:
    if cycle is None:
        cycle = analyse_cycle(**cycle_kwargs)
    return [compute_scenario(cycle=cycle, **sd) for sd in SCENARIO_DEFS]
