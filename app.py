import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq

# Thermodynamic libraries
import thermo
from thermo import Mixture, Chemical

# -----------------------------------------------------------------------------
# 1. CORE THERMODYNAMIC & COMPRESSOR PERFORMANCE SOLVER
# -----------------------------------------------------------------------------
G_ACCEL = 9.80665  # m/s^2

class UniversalCompressorAnalyzer:
    """
    Generic compressor performance analyzer using Python's `thermo` library
    with ASME PTC 10 Schultz polytropic formulation.
    """
    def __init__(self, components: list, fractions: list, basis: str = 'mole'):
        self.components = components
        self.basis = basis
        
        if len(components) == 1:
            self.fluid_type = "Pure"
            self.chem_name = components[0]
        else:
            self.fluid_type = "Mixture"
            total = sum(fractions)
            if basis == 'mole':
                self.zs = [f / total for f in fractions]
                self.ws = None
            else:
                self.ws = [f / total for f in fractions]
                self.zs = None

    def _get_state(self, T: float, P: float):
        """Creates a thermo state instance at (T, P) with auto-selected EOS."""
        if self.fluid_type == "Pure":
            return Chemical(self.chem_name, T=T, P=P)
        else:
            return Mixture(self.components, zs=self.zs, ws=self.ws, T=T, P=P)

    def solve(self, P1: float, T1: float, P2: float, T2_act: float, mass_flow: float):
        """
        Executes isentropic flash and ASME PTC 10 analysis.
        All inputs in SI units (Pa, K, kg/s).
        """
        def extract_props(st):
            MW = st.MW * 1e-3  # kg/mol
            Z = getattr(st, 'Z_g', None) or getattr(st, 'Z', 1.0)
            H = getattr(st, 'H_g', None) or getattr(st, 'H', 0.0)
            S = getattr(st, 'S_g', None) or getattr(st, 'S', 0.0)
            v = (Z * thermo.constants.R * st.T) / (st.P * MW)
            return MW, Z, v, H / MW, S / MW

        # 1. Suction State
        st1 = self._get_state(T=T1, P=P1)
        MW, Z1, v1, h1, s1 = extract_props(st1)

        # 2. Actual Discharge State
        st2_act = self._get_state(T=T2_act, P=P2)
        _, Z2_act, v2_act, h2_act, s2_act = extract_props(st2_act)

        # 3. Isentropic Flash (s(T2_is, P2) = s1)
        def entropy_residual(T_guess):
            st_guess = self._get_state(T=T_guess, P=P2)
            _, _, _, _, s_guess = extract_props(st_guess)
            return s_guess - s1

        rp = P2 / P1
        T_lower = max(T1, 150.0)
        T_upper = T1 * (rp ** 0.45) + 200.0

        try:
            T2_is = brentq(entropy_residual, a=T_lower, b=T_upper, xtol=1e-4)
        except ValueError:
            T2_is = brentq(entropy_residual, a=T1 * 0.9, b=T_upper * 1.5, xtol=1e-3)

        st2_is = self._get_state(T=T2_is, P=P2)
        _, Z2_is, v2_is, h2_is, _ = extract_props(st2_is)

        # 4. Energy Steps (kJ/kg)
        dH_act = (h2_act - h1) / 1e3
        H_is = (h2_is - h1) / 1e3
        eta_is = (H_is / dH_act) * 100.0

        # 5. ASME PTC 10 Schultz Parameters
        vol_ratio_is = v1 / v2_is
        vol_ratio_act = v1 / v2_act
        
        k_v = np.log(rp) / np.log(vol_ratio_is)
        n = np.log(rp) / np.log(vol_ratio_act)
        m = (n - 1.0) / n

        w_is_ideal = (k_v / (k_v - 1.0)) * (P2 * v2_is - P1 * v1)
        f_schultz = (h2_is - h1) / w_is_ideal if abs(w_is_ideal) > 1e-4 else 1.0

        Z_avg = 0.5 * (Z1 + Z2_act)
        H_p = f_schultz * (Z_avg * thermo.constants.R * T1 / MW) * (1.0 / m) * (rp ** m - 1.0) / 1e3
        H_p_meters = (H_p * 1e3) / G_ACCEL
        eta_p = (H_p / dH_act) * 100.0

        return {
            "MW": MW * 1e3,
            "rp": rp,
            "Q1_m3h": mass_flow * v1 * 3600.0,
            "states": {
                "Suction (1)": {"T (°C)": T1 - 273.15, "P (bar)": P1 / 1e5, "Z": Z1, "v (m³/kg)": v1, "h (kJ/kg)": h1 / 1e3, "s (kJ/kg·K)": s1 / 1e3},
                "Isentropic (2is)": {"T (°C)": T2_is - 273.15, "P (bar)": P2 / 1e5, "Z": Z2_is, "v (m³/kg)": v2_is, "h (kJ/kg)": h2_is / 1e3, "s (kJ/kg·K)": s1 / 1e3},
                "Actual (2act)": {"T (°C)": T2_act - 273.15, "P (bar)": P2 / 1e5, "Z": Z2_act, "v (m³/kg)": v2_act, "h (kJ/kg)": h2_act / 1e3, "s (kJ/kg·K)": s2_act / 1e3}
            },
            "kpi": {
                "H_p": H_p, "H_p_m": H_p_meters, "eta_p": eta_p, "H_is": H_is, "eta_is": eta_is,
                "dH_act": dH_act, "gas_power": mass_flow * dH_act, "poly_power": mass_flow * H_p,
                "is_power": mass_flow * H_is, "f_schultz": f_schultz, "k_v": k_v, "n": n, "m": m, "T2_is": T2_is
            }
        }

# -----------------------------------------------------------------------------
# 2. STREAMLIT FRONT-END INTERFACE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Universal Compressor Analyzer", page_icon="⚙️", layout="wide")
st.title("Universal Centrifugal Compressor Thermodynamic Analyzer")
st.markdown("Automated Equation-of-State selection via **`thermo`** & **ASME PTC 10 Schultz Polytropic Head** computation.")

# Sidebar Configuration
with st.sidebar:
    st.header("1. Gas Mixture Definition")
    fluid_mode = st.radio("Stream Selection", ["Common Preset", "Pure Component", "Custom Multi-Component"])
    
    basis = "mole"
    comps, fracs = [], []

    if fluid_mode == "Common Preset":
        preset = st.selectbox("Preset Blend", ["Sales Gas (Pipeline)", "Wet Gas / Untreated", "Ethylene Cracker Gas", "Pure Carbon Dioxide", "Pure Hydrogen"])
        presets_data = {
            "Sales Gas (Pipeline)": (["Methane", "Ethane", "Propane", "n-Butane", "Nitrogen", "Carbon Dioxide"], [0.94, 0.035, 0.010, 0.005, 0.005, 0.005]),
            "Wet Gas / Untreated": (["Methane", "Ethane", "Propane", "Isobutane", "n-Butane", "Carbon Dioxide"], [0.82, 0.09, 0.045, 0.015, 0.015, 0.015]),
            "Ethylene Cracker Gas": (["Ethylene", "Ethane", "Methane", "Propylene", "Hydrogen"], [0.45, 0.20, 0.20, 0.10, 0.05]),
            "Pure Carbon Dioxide": (["Carbon Dioxide"], [1.0]),
            "Pure Hydrogen": (["Hydrogen"], [1.0])
        }
        comps, fracs = presets_data[preset]
    elif fluid_mode == "Pure Component":
        comps = [st.text_input("Chemical Name or Formula", value="Methane").strip()]
        fracs = [1.0]
    else:
        basis = st.selectbox("Composition Basis", ["mole", "mass"])
        comps = [c.strip() for c in st.text_area("Components (comma-separated)", "Methane, Ethane, Propane, Carbon Dioxide").split(",") if c.strip()]
        fracs = [float(f.strip()) for f in st.text_area("Fractions (comma-separated, same order)", "0.90, 0.06, 0.03, 0.01").split(",") if f.strip()]

    st.header("2. Operating Boundary Conditions")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        P1_bar = st.number_input("P₁ Suction (bar a)", value=20.0, min_value=0.2, step=1.0)
        T1_c = st.number_input("T₁ Suction (°C)", value=32.0, step=1.0)
    with col_u2:
        P2_bar = st.number_input("P₂ Discharge (bar a)", value=68.0, min_value=P1_bar + 0.1, step=1.0)
        T2_act_c = st.number_input("T₂ Actual Discharge (°C)", value=138.0, min_value=T1_c + 0.5, step=1.0)
    
    flow_unit = st.selectbox("Flow Rate Unit", ["tonne/hr", "kg/s", "MMSCFD (Std. Gas)"])
    flow_val = st.number_input("Flow Rate Value", value=40.0, min_value=0.01, step=1.0)

# Flow conversion to kg/s
mass_flow_kgs = flow_val * (1000.0 / 3600.0 if flow_unit == "tonne/hr" else (1.0 if flow_unit == "kg/s" else 0.327))

# Solver Execution
try:
    with st.spinner("Executing auto-selected EOS flash routines..."):
        analyzer = UniversalCompressorAnalyzer(components=comps, fractions=fracs, basis=basis)
        res = analyzer.solve(P1=P1_bar * 1e5, T1=T1_c + 273.15, P2=P2_bar * 1e5, T2_act=T2_act_c + 273.15, mass_flow=mass_flow_kgs)
        kpi, states = res["kpi"], res["states"]

    # 3. EXECUTIVE KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Polytropic Head (Hp)", f"{kpi['H_p']:.2f} kJ/kg", f"{kpi['H_p_m']:.1f} m head")
    k2.metric("Polytropic Efficiency (ηp)", f"{kpi['eta_p']:.2f} %")
    k3.metric("Isentropic Efficiency (ηis)", f"{kpi['eta_is']:.2f} %")
    k4.metric("Gas Shaft Power", f"{kpi['gas_power']:.1f} kW", f"{kpi['gas_power']/745.7:.1f} HP")

    if states["Actual (2act)"]["s (kJ/kg·K)"] - states["Suction (1)"]["s (kJ/kg·K)"] < -1e-4:
        st.warning("⚠️ Second Law Violation: Actual discharge entropy is lower than suction entropy. Verify discharge temperature measurement.")

    st.markdown("---")

    # 4. DETAILED TABS
    tab_summary, tab_asme, tab_diagrams = st.tabs(["State Point Matrix", "ASME PTC 10 Diagnostics", "Thermodynamic Paths (P-h & T-s)"])

    with tab_summary:
        st.subheader("State Point Thermodynamic Properties")
        st.dataframe(pd.DataFrame(states).T.style.format("{:.4f}"), use_container_width=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.table(pd.DataFrame({
                "Parameter": ["Compression Ratio (rp)", "Gas Molecular Weight (MW)", "Suction Actual Flow (Q₁)", "Specific Volume Ratio (v₁/v₂)"],
                "Value": [f"{res['rp']:.3f}", f"{res['MW']:.3f} kg/kmol", f"{res['Q1_m3h']:.2f} m³/h", f"{states['Suction (1)']['v (m³/kg)'] / states['Actual (2act)']['v (m³/kg)']:.3f}"]
            }))
        with c_right:
            st.table(pd.DataFrame({
                "Metric": ["Actual Enthalpy Rise (Δh_act)", "Isentropic Head (H_is)", "Polytropic Head (H_p)", "Isentropic Discharge Temp (T₂_is)"],
                "Value": [f"{kpi['dH_act']:.2f} kJ/kg", f"{kpi['H_is']:.2f} kJ/kg", f"{kpi['H_p']:.2f} kJ/kg", f"{kpi['T2_is'] - 273.15:.2f} °C"]
            }))

    with tab_asme:
        st.subheader("ASME PTC 10 Schultz Real Gas Evaluation")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            * **Isentropic Volume Exponent ($k_v$):** `{kpi['k_v']:.4f}`
            * **Polytropic Volume Exponent ($n$):** `{kpi['n']:.4f}`
            * **Polytropic Temperature Exponent ($m = \\frac{{n-1}}{{n}}$):** `{kpi['m']:.4f}`
            """)
        with col_m2:
            st.markdown(f"""
            * **Schultz Compressibility Correction Factor ($f$):** `{kpi['f_schultz']:.4f}`
            * **Suction Compressibility ($Z_1$):** `{states['Suction (1)']['Z']:.4f}`
            * **Discharge Compressibility ($Z_{{2,\\text{{act}}}}$):** `{states['Actual (2act)']['Z']:.4f}`
            """)

    with tab_diagrams:
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            fig_ph = go.Figure()
            fig_ph.add_trace(go.Scatter(x=[states["Suction (1)"]["h (kJ/kg)"], states["Isentropic (2is)"]["h (kJ/kg)"]], y=[P1_bar, P2_bar], mode='lines+markers', name='Isentropic Path', line=dict(color='green', dash='dash', width=2)))
            fig_ph.add_trace(go.Scatter(x=[states["Suction (1)"]["h (kJ/kg)"], states["Actual (2act)"]["h (kJ/kg)"]], y=[P1_bar, P2_bar], mode='lines+markers', name='Actual Path', line=dict(color='crimson', width=3)))
            fig_ph.update_layout(title="Pressure-Enthalpy (P-h)", xaxis_title="Enthalpy (kJ/kg)", yaxis_title="Pressure (bar)", yaxis_type="log", height=420)
            st.plotly_chart(fig_ph, use_container_width=True)

        with p_c2:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=[states["Suction (1)"]["s (kJ/kg·K)"], states["Isentropic (2is)"]["s (kJ/kg·K)"]], y=[states["Suction (1)"]["T (°C)"], states["Isentropic (2is)"]["T (°C)"]], mode='lines+markers', name='Isentropic Path', line=dict(color='green', dash='dash', width=2)))
            fig_ts.add_trace(go.Scatter(x=[states["Suction (1)"]["s (kJ/kg·K)"], states["Actual (2act)"]["s (kJ/kg·K)"]], y=[states["Suction (1)"]["T (°C)"], states["Actual (2act)"]["T (°C)"]], mode='lines+markers', name='Actual Path', line=dict(color='crimson', width=3)))
            fig_ts.update_layout(title="Temperature-Entropy (T-s)", xaxis_title="Entropy (kJ/kg·K)", yaxis_title="Temperature (°C)", height=420)
            st.plotly_chart(fig_ts, use_container_width=True)

except Exception as err:
    st.error(f"Thermodynamic solver encountered an issue: {str(err)}")
    st.info("Check component spellings, ensure fractions sum to 1.0, and verify process pressures remain in the single-phase vapor region.")
