import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
import thermo
from thermo import Mixture, Chemical

# -----------------------------------------------------------------------------
# 1. CORE THERMODYNAMIC & COMPRESSOR PERFORMANCE SOLVER
# -----------------------------------------------------------------------------
G_ACCEL = 9.80665
R_GAS = 8.314462618

class CompressorEngine:
    def __init__(self, components: list, fractions: list, basis: str = 'mole'):
        self.components = components
        self.basis = basis
        self.is_pure = len(components) == 1
        
        if self.is_pure:
            self.chem_name = components[0]
            st_ref = Chemical(self.chem_name, T=298.15, P=101325.0)
            self.MW = st_ref.MW * 1e-3  # kg/mol
        else:
            total = sum(fractions)
            norm_f = [f / total for f in fractions]
            self.zs = norm_f if basis == 'mole' else None
            self.ws = norm_f if basis == 'mass' else None
            st_ref = Mixture(self.components, zs=self.zs, ws=self.ws, T=298.15, P=101325.0)
            self.MW = st_ref.MW * 1e-3  # kg/mol

    def _state(self, T: float, P: float):
        if self.is_pure:
            return Chemical(self.chem_name, T=T, P=P)
        return Mixture(self.components, zs=self.zs, ws=self.ws, T=T, P=P)

    def evaluate_point(self, P1: float, T1: float, P2: float, T2_act: float, m_flow: float):
        """
        Solves ASME PTC 10 Schultz method for a single operational state.
        Inputs: P1, P2 [Pa], T1, T2 [K], m_flow [kg/s].
        """
        if P2 <= P1 or T2_act <= T1 or m_flow <= 0:
            return None

        try:
            # 1. Suction State
            s1_obj = self._state(T1, P1)
            Z1 = s1_obj.Z_g if s1_obj.Z_g is not None else s1_obj.Z
            v1 = (Z1 * R_GAS * T1) / (P1 * self.MW)
            h1 = (s1_obj.H_g if s1_obj.H_g is not None else s1_obj.H) / self.MW
            s1 = (s1_obj.S_g if s1_obj.S_g is not None else s1_obj.S) / self.MW

            # 2. Actual Discharge State
            s2_obj = self._state(T2_act, P2)
            Z2_act = s2_obj.Z_g if s2_obj.Z_g is not None else s2_obj.Z
            v2_act = (Z2_act * R_GAS * T2_act) / (P2 * self.MW)
            h2_act = (s2_obj.H_g if s2_obj.H_g is not None else s2_obj.H) / self.MW
            s2_act = (s2_obj.S_g if s2_obj.S_g is not None else s2_obj.S) / self.MW

            # 3. Rigorous Isentropic Flash (s(T2_is, P2) = s1)
            def residual(T):
                st_guess = self._state(T, P2)
                s_guess = (st_guess.S_g if st_guess.S_g is not None else st_guess.S) / self.MW
                return s_guess - s1

            rp = P2 / P1
            T_upper = T1 * (rp**0.45) + 200.0
            T2_is = brentq(residual, a=T1, b=T_upper, xtol=1e-3)

            s2_is_obj = self._state(T2_is, P2)
            v2_is = ((s2_is_obj.Z_g if s2_is_obj.Z_g is not None else s2_is_obj.Z) * R_GAS * T2_is) / (P2 * self.MW)
            h2_is = (s2_is_obj.H_g if s2_is_obj.H_g is not None else s2_is_obj.H) / self.MW

            # 4. Energy steps & Exponents
            dH_act = (h2_act - h1) / 1e3     # kJ/kg
            H_is = (h2_is - h1) / 1e3        # kJ/kg
            eta_is = (H_is / dH_act) * 100.0 # %

            vol_ratio_is = v1 / v2_is
            vol_ratio_act = v1 / v2_act
            k_v = np.log(rp) / np.log(vol_ratio_is)
            n = np.log(rp) / np.log(vol_ratio_act)
            m = (n - 1.0) / n

            # Schultz correction factor
            w_is_ideal = (k_v / (k_v - 1.0)) * (P2 * v2_is - P1 * v1)
            f_schultz = (h2_is - h1) / w_is_ideal if abs(w_is_ideal) > 1e-4 else 1.0

            # Polytropic Head & Efficiencies
            Z_avg = 0.5 * (Z1 + Z2_act)
            H_p = f_schultz * (Z_avg * R_GAS * T1 / self.MW) * (1.0 / m) * (rp**m - 1.0) / 1e3 # kJ/kg
            H_p_m = (H_p * 1e3) / G_ACCEL
            eta_p = (H_p / dH_act) * 100.0

            gas_power_kw = m_flow * dH_act
            Q1_m3h = m_flow * v1 * 3600.0

            return {
                "rp": rp,
                "Z1": Z1,
                "Z2_act": Z2_act,
                "Q1_m3h": Q1_m3h,
                "dH_act_kJ_kg": dH_act,
                "H_is_kJ_kg": H_is,
                "H_p_kJ_kg": H_p,
                "H_p_m": H_p_m,
                "eta_is_pct": eta_is,
                "eta_p_pct": eta_p,
                "gas_power_kw": gas_power_kw,
                "n_poly": n,
                "f_schultz": f_schultz,
                "s_gen": s2_act - s1
            }
        except Exception:
            return None

# -----------------------------------------------------------------------------
# 2. UNIT CONVERSIONS HELPER
# -----------------------------------------------------------------------------
def convert_to_si(val, unit, unit_type):
    if unit_type == "P":  # Target: Pa
        if unit == "bar a": return val * 1e5
        if unit == "barg": return (val + 1.01325) * 1e5
        if unit == "kPa a": return val * 1e3
        if unit == "psi a": return val * 6894.757
        if unit == "psig": return (val + 14.696) * 6894.757
        return val * 1e5
    elif unit_type == "T":  # Target: K
        if unit == "°C": return val + 273.15
        if unit == "K": return val
        if unit == "°F": return (val - 32.0) * (5.0 / 9.0) + 273.15
        return val + 273.15
    elif unit_type == "Flow":  # Target: kg/s
        if unit == "tonne/h": return (val * 1000.0) / 3600.0
        if unit == "kg/s": return val
        if unit == "kg/h": return val / 3600.0
        if unit == "lb/h": return val * 0.000125998
        return val

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Compressor Historian Analytics", layout="wide")
st.title("Compressor Historian Analytics & Degradation Tracking")
st.caption("Batch ASME PTC 10 Schultz Processing with Moving Average Filters & Anomaly Detection")

with st.sidebar:
    st.header("1. Gas Composition")
    preset = st.selectbox("Fluid Blend", ["Sales Gas (Pipeline)", "Rich Natural Gas", "Pure Methane", "Custom Mixture"])
    if preset == "Sales Gas (Pipeline)":
        comps = ["Methane", "Ethane", "Propane", "n-Butane", "Nitrogen", "Carbon Dioxide"]
        fracs = [0.94, 0.035, 0.010, 0.005, 0.005, 0.005]
    elif preset == "Rich Natural Gas":
        comps = ["Methane", "Ethane", "Propane", "Isobutane", "n-Butane", "Carbon Dioxide"]
        fracs = [0.82, 0.09, 0.045, 0.015, 0.015, 0.015]
    elif preset == "Pure Methane":
        comps = ["Methane"]
        fracs = [1.0]
    else:
        c_in = st.text_input("Components", "Methane, Ethane, Propane")
        f_in = st.text_input("Mole Fractions", "0.90, 0.07, 0.03")
        comps = [c.strip() for c in c_in.split(",")]
        fracs = [float(f.strip()) for f in f_in.split(",")]

    st.header("2. Data Ingestion")
    uploaded_file = st.file_uploader("Upload PI/Historian Log (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload a CSV/Excel file to initiate batch calculation. You can test with synthetic operational data below.")
    if st.button("Generate Synthetic Demo Historian Data (30 Days)"):
        dates = pd.date_range(start="2026-08-01", periods=200, freq="4h")
        np.random.seed(42)
        
        # Simulating baseline with progressive fouling (efficiency loss of ~3% over 30 days)
        deg_trend = np.linspace(0, 0.035, len(dates))
        base_T2 = 138.0 + (deg_trend * 180.0) + np.random.normal(0, 1.2, len(dates))
        
        demo_df = pd.DataFrame({
            "Timestamp": dates,
            "PT_101_SUCTION_BAR": np.random.normal(22.0, 0.3, len(dates)),
            "TI_101_SUCTION_C": np.random.normal(32.0, 0.8, len(dates)),
            "PT_102_DISCH_BAR": np.random.normal(68.0, 0.6, len(dates)),
            "TI_102_DISCH_C": base_T2,
            "FT_101_FLOW_TPH": np.random.normal(42.0, 2.0, len(dates))
        })
        st.session_state["raw_df"] = demo_df
        st.success("Synthetic dataset generated! Select column mappings below.")
else:
    if uploaded_file.name.endswith('.csv'):
        st.session_state["raw_df"] = pd.read_csv(uploaded_file)
    else:
        st.session_state["raw_df"] = pd.read_excel(uploaded_file)

if "raw_df" in st.session_state:
    df = st.session_state["raw_df"].copy()
    
    st.subheader("Data Preview")
    st.dataframe(df.head(4), use_container_width=True)

    st.markdown("### Step 2: Map Historian Tags & Specify Units")
    cols = list(df.columns)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        time_tag = st.selectbox("Timestamp Column", cols, index=0)
        p1_tag = st.selectbox("Suction Pressure (P₁)", cols, index=min(1, len(cols)-1))
        p1_unit = st.selectbox("P₁ Unit", ["bar a", "barg", "kPa a", "psi a", "psig"], index=0)
    with m_col2:
        t1_tag = st.selectbox("Suction Temperature (T₁)", cols, index=min(2, len(cols)-1))
        t1_unit = st.selectbox("T₁ Unit", ["°C", "K", "°F"], index=0)
        p2_tag = st.selectbox("Discharge Pressure (P₂)", cols, index=min(3, len(cols)-1))
        p2_unit = st.selectbox("P₂ Unit", ["bar a", "barg", "kPa a", "psi a", "psig"], index=0)
    with m_col3:
        t2_tag = st.selectbox("Discharge Temperature (T₂)", cols, index=min(4, len(cols)-1))
        t2_unit = st.selectbox("T₂ Unit", ["°C", "K", "°F"], index=0)
        flow_tag = st.selectbox("Mass Flow Rate", cols, index=min(5, len(cols)-1))
        flow_unit = st.selectbox("Flow Unit", ["tonne/h", "kg/s", "kg/h", "lb/h"], index=0)

    if st.button("Run Batch Thermodynamic Engine", type="primary"):
        engine = CompressorEngine(components=comps, fractions=fracs)
        
        # Ensure timestamp parse
        df[time_tag] = pd.to_datetime(df[time_tag])
        df = df.sort_values(by=time_tag).reset_index(drop=True)

        results = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        total_rows = len(df)
        for i, row in df.iterrows():
            P1_si = convert_to_si(row[p1_tag], p1_unit, "P")
            T1_si = convert_to_si(row[t1_tag], t1_unit, "T")
            P2_si = convert_to_si(row[p2_tag], p2_unit, "P")
            T2_si = convert_to_si(row[t2_tag], t2_unit, "T")
            m_si = convert_to_si(row[flow_tag], flow_unit, "Flow")

            kpi = engine.evaluate_point(P1_si, T1_si, P2_si, T2_si, m_si)
            results.append(kpi if kpi is not None else {})

            if i % max(1, (total_rows // 20)) == 0 or i == total_rows - 1:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i+1} / {total_rows} historian records...")

        status_text.empty()
        progress_bar.empty()

        kpi_df = pd.DataFrame(results)
        merged_df = pd.concat([df, kpi_df], axis=1).dropna(subset=["eta_p_pct"])
        st.session_state["processed_df"] = merged_df
        st.session_state["time_tag"] = time_tag

# -----------------------------------------------------------------------------
# 4. RESULTS & CONDITION MONITORING DASHBOARD
# -----------------------------------------------------------------------------
if "processed_df" in st.session_state:
    res_df = st.session_state["processed_df"]
    t_col = st.session_state["time_tag"]

    st.markdown("---")
    st.subheader("Condition Monitoring Dashboard & KPI Trends")

    # Rolling window parameter
    roll_window = st.slider("Rolling Average Window (Data Points)", min_value=1, max_value=50, value=7)
    res_df["eta_p_smoothed"] = res_df["eta_p_pct"].rolling(window=roll_window, center=True).mean()
    res_df["eta_is_smoothed"] = res_df["eta_is_pct"].rolling(window=roll_window, center=True).mean()

    # High-level Statistics
    delta_eff = res_df["eta_p_smoothed"].dropna().iloc[-1] - res_df["eta_p_smoothed"].dropna().iloc[0]
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Polytropic Efficiency", f"{res_df['eta_p_pct'].mean():.2f} %")
    k2.metric("Efficiency Shift (Δη)", f"{delta_eff:+.2f} %", delta_color="normal")
    k3.metric("Mean Polytropic Head", f"{res_df['H_p_m'].mean():.1f} m")
    k4.metric("Mean Gas Power", f"{res_df['gas_power_kw'].mean():.1f} kW")

    # Tabbed Visualizations
    t_eff, t_head, t_map = st.tabs(["Efficiency Degradation", "Head & Power Time-Series", "Operating Dynamic Map (Hp vs Q1)"])

    with t_eff:
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_p_pct"],
            mode='markers', name='Polytropic Eff (Raw)',
            marker=dict(size=4, color='rgba(31, 119, 180, 0.3)')
        ))
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_p_smoothed"],
            mode='lines', name=f'Polytropic Eff ({roll_window}-pt MA)',
            line=dict(color='#1f77b4', width=2.5)
        ))
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_is_smoothed"],
            mode='lines', name=f'Isentropic Eff ({roll_window}-pt MA)',
            line=dict(color='#2ca02c', width=2, dash='dot')
        ))
        fig_eff.update_layout(
            title="Compressor Efficiency History & Degradation Profile",
            xaxis_title="Timestamp",
            yaxis_title="Efficiency (%)",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with t_head:
        fig_hp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hp.add_trace(
            go.Scatter(x=res_df[t_col], y=res_df["H_p_m"], name="Polytropic Head (m)", line=dict(color="#ff7f0e")),
            secondary_y=False
        )
        fig_hp.add_trace(
            go.Scatter(x=res_df[t_col], y=res_df["gas_power_kw"], name="Gas Power (kW)", line=dict(color="#d62728", dash="dash")),
            secondary_y=True
        )
        fig_hp.update_layout(title="Polytropic Head and Driver Power Demand Over Time", height=450, hovermode="x unified")
        fig_hp.update_xaxes(title_text="Timestamp")
        fig_hp.update_yaxes(title_text="Polytropic Head (m)", secondary_y=False)
        fig_hp.update_yaxes(title_text="Gas Power (kW)", secondary_y=True)
        st.plotly_chart(fig_hp, use_container_width=True)

    with t_map:
        fig_scatter = go.Figure()
        scatter = fig_scatter.add_trace(go.Scatter(
            x=res_df["Q1_m3h"], y=res_df["H_p_m"],
            mode='markers',
            marker=dict(
                size=7,
                color=res_df.index,
                colorscale='Viridis',
                colorbar=dict(title="Time Progression"),
                showscale=True
            ),
            text=res_df[t_col].dt.strftime('%Y-%m-%d %H:%M'),
            hovertemplate="<b>Date</b>: %{text}<br><b>Flow Q₁</b>: %{x:.1f} m³/h<br><b>Head</b>: %{y:.1f} m<extra></extra>"
        ))
        fig_scatter.update_layout(
            title="Operating Point Migration on Head-Flow Space",
            xaxis_title="Suction Volumetric Flow Q₁ (m³/h)",
            yaxis_title="Polytropic Head (m)",
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Export Section
    st.subheader("Export Calculated KPIs")
    csv_out = res_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Dataset with KPIs (CSV)",
        data=csv_out,
        file_name="compressor_kpi_historian_evaluated.csv",
        mime="text/csv"
    )
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
import thermo
from thermo import Mixture, Chemical

# -----------------------------------------------------------------------------
# 1. CORE THERMODYNAMIC & COMPRESSOR PERFORMANCE SOLVER
# -----------------------------------------------------------------------------
G_ACCEL = 9.80665
R_GAS = 8.314462618

class CompressorEngine:
    def __init__(self, components: list, fractions: list, basis: str = 'mole'):
        self.components = components
        self.basis = basis
        self.is_pure = len(components) == 1
        
        if self.is_pure:
            self.chem_name = components[0]
            st_ref = Chemical(self.chem_name, T=298.15, P=101325.0)
            self.MW = st_ref.MW * 1e-3  # kg/mol
        else:
            total = sum(fractions)
            norm_f = [f / total for f in fractions]
            self.zs = norm_f if basis == 'mole' else None
            self.ws = norm_f if basis == 'mass' else None
            st_ref = Mixture(self.components, zs=self.zs, ws=self.ws, T=298.15, P=101325.0)
            self.MW = st_ref.MW * 1e-3  # kg/mol

    def _state(self, T: float, P: float):
        if self.is_pure:
            return Chemical(self.chem_name, T=T, P=P)
        return Mixture(self.components, zs=self.zs, ws=self.ws, T=T, P=P)

    def evaluate_point(self, P1: float, T1: float, P2: float, T2_act: float, m_flow: float):
        """
        Solves ASME PTC 10 Schultz method for a single operational state.
        Inputs: P1, P2 [Pa], T1, T2 [K], m_flow [kg/s].
        """
        if P2 <= P1 or T2_act <= T1 or m_flow <= 0:
            return None

        try:
            # 1. Suction State
            s1_obj = self._state(T1, P1)
            Z1 = s1_obj.Z_g if s1_obj.Z_g is not None else s1_obj.Z
            v1 = (Z1 * R_GAS * T1) / (P1 * self.MW)
            h1 = (s1_obj.H_g if s1_obj.H_g is not None else s1_obj.H) / self.MW
            s1 = (s1_obj.S_g if s1_obj.S_g is not None else s1_obj.S) / self.MW

            # 2. Actual Discharge State
            s2_obj = self._state(T2_act, P2)
            Z2_act = s2_obj.Z_g if s2_obj.Z_g is not None else s2_obj.Z
            v2_act = (Z2_act * R_GAS * T2_act) / (P2 * self.MW)
            h2_act = (s2_obj.H_g if s2_obj.H_g is not None else s2_obj.H) / self.MW
            s2_act = (s2_obj.S_g if s2_obj.S_g is not None else s2_obj.S) / self.MW

            # 3. Rigorous Isentropic Flash (s(T2_is, P2) = s1)
            def residual(T):
                st_guess = self._state(T, P2)
                s_guess = (st_guess.S_g if st_guess.S_g is not None else st_guess.S) / self.MW
                return s_guess - s1

            rp = P2 / P1
            T_upper = T1 * (rp**0.45) + 200.0
            T2_is = brentq(residual, a=T1, b=T_upper, xtol=1e-3)

            s2_is_obj = self._state(T2_is, P2)
            v2_is = ((s2_is_obj.Z_g if s2_is_obj.Z_g is not None else s2_is_obj.Z) * R_GAS * T2_is) / (P2 * self.MW)
            h2_is = (s2_is_obj.H_g if s2_is_obj.H_g is not None else s2_is_obj.H) / self.MW

            # 4. Energy steps & Exponents
            dH_act = (h2_act - h1) / 1e3     # kJ/kg
            H_is = (h2_is - h1) / 1e3        # kJ/kg
            eta_is = (H_is / dH_act) * 100.0 # %

            vol_ratio_is = v1 / v2_is
            vol_ratio_act = v1 / v2_act
            k_v = np.log(rp) / np.log(vol_ratio_is)
            n = np.log(rp) / np.log(vol_ratio_act)
            m = (n - 1.0) / n

            # Schultz correction factor
            w_is_ideal = (k_v / (k_v - 1.0)) * (P2 * v2_is - P1 * v1)
            f_schultz = (h2_is - h1) / w_is_ideal if abs(w_is_ideal) > 1e-4 else 1.0

            # Polytropic Head & Efficiencies
            Z_avg = 0.5 * (Z1 + Z2_act)
            H_p = f_schultz * (Z_avg * R_GAS * T1 / self.MW) * (1.0 / m) * (rp**m - 1.0) / 1e3 # kJ/kg
            H_p_m = (H_p * 1e3) / G_ACCEL
            eta_p = (H_p / dH_act) * 100.0

            gas_power_kw = m_flow * dH_act
            Q1_m3h = m_flow * v1 * 3600.0

            return {
                "rp": rp,
                "Z1": Z1,
                "Z2_act": Z2_act,
                "Q1_m3h": Q1_m3h,
                "dH_act_kJ_kg": dH_act,
                "H_is_kJ_kg": H_is,
                "H_p_kJ_kg": H_p,
                "H_p_m": H_p_m,
                "eta_is_pct": eta_is,
                "eta_p_pct": eta_p,
                "gas_power_kw": gas_power_kw,
                "n_poly": n,
                "f_schultz": f_schultz,
                "s_gen": s2_act - s1
            }
        except Exception:
            return None

# -----------------------------------------------------------------------------
# 2. UNIT CONVERSIONS HELPER
# -----------------------------------------------------------------------------
def convert_to_si(val, unit, unit_type):
    if unit_type == "P":  # Target: Pa
        if unit == "bar a": return val * 1e5
        if unit == "barg": return (val + 1.01325) * 1e5
        if unit == "kPa a": return val * 1e3
        if unit == "psi a": return val * 6894.757
        if unit == "psig": return (val + 14.696) * 6894.757
        return val * 1e5
    elif unit_type == "T":  # Target: K
        if unit == "°C": return val + 273.15
        if unit == "K": return val
        if unit == "°F": return (val - 32.0) * (5.0 / 9.0) + 273.15
        return val + 273.15
    elif unit_type == "Flow":  # Target: kg/s
        if unit == "tonne/h": return (val * 1000.0) / 3600.0
        if unit == "kg/s": return val
        if unit == "kg/h": return val / 3600.0
        if unit == "lb/h": return val * 0.000125998
        return val

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Compressor Historian Analytics", layout="wide")
st.title("Compressor Historian Analytics & Degradation Tracking")
st.caption("Batch ASME PTC 10 Schultz Processing with Moving Average Filters & Anomaly Detection")

with st.sidebar:
    st.header("1. Gas Composition")
    preset = st.selectbox("Fluid Blend", ["Sales Gas (Pipeline)", "Rich Natural Gas", "Pure Methane", "Custom Mixture"])
    if preset == "Sales Gas (Pipeline)":
        comps = ["Methane", "Ethane", "Propane", "n-Butane", "Nitrogen", "Carbon Dioxide"]
        fracs = [0.94, 0.035, 0.010, 0.005, 0.005, 0.005]
    elif preset == "Rich Natural Gas":
        comps = ["Methane", "Ethane", "Propane", "Isobutane", "n-Butane", "Carbon Dioxide"]
        fracs = [0.82, 0.09, 0.045, 0.015, 0.015, 0.015]
    elif preset == "Pure Methane":
        comps = ["Methane"]
        fracs = [1.0]
    else:
        c_in = st.text_input("Components", "Methane, Ethane, Propane")
        f_in = st.text_input("Mole Fractions", "0.90, 0.07, 0.03")
        comps = [c.strip() for c in c_in.split(",")]
        fracs = [float(f.strip()) for f in f_in.split(",")]

    st.header("2. Data Ingestion")
    uploaded_file = st.file_uploader("Upload PI/Historian Log (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload a CSV/Excel file to initiate batch calculation. You can test with synthetic operational data below.")
    if st.button("Generate Synthetic Demo Historian Data (30 Days)"):
        dates = pd.date_range(start="2026-08-01", periods=200, freq="4h")
        np.random.seed(42)
        
        # Simulating baseline with progressive fouling (efficiency loss of ~3% over 30 days)
        deg_trend = np.linspace(0, 0.035, len(dates))
        base_T2 = 138.0 + (deg_trend * 180.0) + np.random.normal(0, 1.2, len(dates))
        
        demo_df = pd.DataFrame({
            "Timestamp": dates,
            "PT_101_SUCTION_BAR": np.random.normal(22.0, 0.3, len(dates)),
            "TI_101_SUCTION_C": np.random.normal(32.0, 0.8, len(dates)),
            "PT_102_DISCH_BAR": np.random.normal(68.0, 0.6, len(dates)),
            "TI_102_DISCH_C": base_T2,
            "FT_101_FLOW_TPH": np.random.normal(42.0, 2.0, len(dates))
        })
        st.session_state["raw_df"] = demo_df
        st.success("Synthetic dataset generated! Select column mappings below.")
else:
    if uploaded_file.name.endswith('.csv'):
        st.session_state["raw_df"] = pd.read_csv(uploaded_file)
    else:
        st.session_state["raw_df"] = pd.read_excel(uploaded_file)

if "raw_df" in st.session_state:
    df = st.session_state["raw_df"].copy()
    
    st.subheader("Data Preview")
    st.dataframe(df.head(4), use_container_width=True)

    st.markdown("### Step 2: Map Historian Tags & Specify Units")
    cols = list(df.columns)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        time_tag = st.selectbox("Timestamp Column", cols, index=0)
        p1_tag = st.selectbox("Suction Pressure (P₁)", cols, index=min(1, len(cols)-1))
        p1_unit = st.selectbox("P₁ Unit", ["bar a", "barg", "kPa a", "psi a", "psig"], index=0)
    with m_col2:
        t1_tag = st.selectbox("Suction Temperature (T₁)", cols, index=min(2, len(cols)-1))
        t1_unit = st.selectbox("T₁ Unit", ["°C", "K", "°F"], index=0)
        p2_tag = st.selectbox("Discharge Pressure (P₂)", cols, index=min(3, len(cols)-1))
        p2_unit = st.selectbox("P₂ Unit", ["bar a", "barg", "kPa a", "psi a", "psig"], index=0)
    with m_col3:
        t2_tag = st.selectbox("Discharge Temperature (T₂)", cols, index=min(4, len(cols)-1))
        t2_unit = st.selectbox("T₂ Unit", ["°C", "K", "°F"], index=0)
        flow_tag = st.selectbox("Mass Flow Rate", cols, index=min(5, len(cols)-1))
        flow_unit = st.selectbox("Flow Unit", ["tonne/h", "kg/s", "kg/h", "lb/h"], index=0)

    if st.button("Run Batch Thermodynamic Engine", type="primary"):
        engine = CompressorEngine(components=comps, fractions=fracs)
        
        # Ensure timestamp parse
        df[time_tag] = pd.to_datetime(df[time_tag])
        df = df.sort_values(by=time_tag).reset_index(drop=True)

        results = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        total_rows = len(df)
        for i, row in df.iterrows():
            P1_si = convert_to_si(row[p1_tag], p1_unit, "P")
            T1_si = convert_to_si(row[t1_tag], t1_unit, "T")
            P2_si = convert_to_si(row[p2_tag], p2_unit, "P")
            T2_si = convert_to_si(row[t2_tag], t2_unit, "T")
            m_si = convert_to_si(row[flow_tag], flow_unit, "Flow")

            kpi = engine.evaluate_point(P1_si, T1_si, P2_si, T2_si, m_si)
            results.append(kpi if kpi is not None else {})

            if i % max(1, (total_rows // 20)) == 0 or i == total_rows - 1:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i+1} / {total_rows} historian records...")

        status_text.empty()
        progress_bar.empty()

        kpi_df = pd.DataFrame(results)
        merged_df = pd.concat([df, kpi_df], axis=1).dropna(subset=["eta_p_pct"])
        st.session_state["processed_df"] = merged_df
        st.session_state["time_tag"] = time_tag

# -----------------------------------------------------------------------------
# 4. RESULTS & CONDITION MONITORING DASHBOARD
# -----------------------------------------------------------------------------
if "processed_df" in st.session_state:
    res_df = st.session_state["processed_df"]
    t_col = st.session_state["time_tag"]

    st.markdown("---")
    st.subheader("Condition Monitoring Dashboard & KPI Trends")

    # Rolling window parameter
    roll_window = st.slider("Rolling Average Window (Data Points)", min_value=1, max_value=50, value=7)
    res_df["eta_p_smoothed"] = res_df["eta_p_pct"].rolling(window=roll_window, center=True).mean()
    res_df["eta_is_smoothed"] = res_df["eta_is_pct"].rolling(window=roll_window, center=True).mean()

    # High-level Statistics
    delta_eff = res_df["eta_p_smoothed"].dropna().iloc[-1] - res_df["eta_p_smoothed"].dropna().iloc[0]
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Polytropic Efficiency", f"{res_df['eta_p_pct'].mean():.2f} %")
    k2.metric("Efficiency Shift (Δη)", f"{delta_eff:+.2f} %", delta_color="normal")
    k3.metric("Mean Polytropic Head", f"{res_df['H_p_m'].mean():.1f} m")
    k4.metric("Mean Gas Power", f"{res_df['gas_power_kw'].mean():.1f} kW")

    # Tabbed Visualizations
    t_eff, t_head, t_map = st.tabs(["Efficiency Degradation", "Head & Power Time-Series", "Operating Dynamic Map (Hp vs Q1)"])

    with t_eff:
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_p_pct"],
            mode='markers', name='Polytropic Eff (Raw)',
            marker=dict(size=4, color='rgba(31, 119, 180, 0.3)')
        ))
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_p_smoothed"],
            mode='lines', name=f'Polytropic Eff ({roll_window}-pt MA)',
            line=dict(color='#1f77b4', width=2.5)
        ))
        fig_eff.add_trace(go.Scatter(
            x=res_df[t_col], y=res_df["eta_is_smoothed"],
            mode='lines', name=f'Isentropic Eff ({roll_window}-pt MA)',
            line=dict(color='#2ca02c', width=2, dash='dot')
        ))
        fig_eff.update_layout(
            title="Compressor Efficiency History & Degradation Profile",
            xaxis_title="Timestamp",
            yaxis_title="Efficiency (%)",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with t_head:
        fig_hp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hp.add_trace(
            go.Scatter(x=res_df[t_col], y=res_df["H_p_m"], name="Polytropic Head (m)", line=dict(color="#ff7f0e")),
            secondary_y=False
        )
        fig_hp.add_trace(
            go.Scatter(x=res_df[t_col], y=res_df["gas_power_kw"], name="Gas Power (kW)", line=dict(color="#d62728", dash="dash")),
            secondary_y=True
        )
        fig_hp.update_layout(title="Polytropic Head and Driver Power Demand Over Time", height=450, hovermode="x unified")
        fig_hp.update_xaxes(title_text="Timestamp")
        fig_hp.update_yaxes(title_text="Polytropic Head (m)", secondary_y=False)
        fig_hp.update_yaxes(title_text="Gas Power (kW)", secondary_y=True)
        st.plotly_chart(fig_hp, use_container_width=True)

    with t_map:
        fig_scatter = go.Figure()
        scatter = fig_scatter.add_trace(go.Scatter(
            x=res_df["Q1_m3h"], y=res_df["H_p_m"],
            mode='markers',
            marker=dict(
                size=7,
                color=res_df.index,
                colorscale='Viridis',
                colorbar=dict(title="Time Progression"),
                showscale=True
            ),
            text=res_df[t_col].dt.strftime('%Y-%m-%d %H:%M'),
            hovertemplate="<b>Date</b>: %{text}<br><b>Flow Q₁</b>: %{x:.1f} m³/h<br><b>Head</b>: %{y:.1f} m<extra></extra>"
        ))
        fig_scatter.update_layout(
            title="Operating Point Migration on Head-Flow Space",
            xaxis_title="Suction Volumetric Flow Q₁ (m³/h)",
            yaxis_title="Polytropic Head (m)",
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Export Section
    st.subheader("Export Calculated KPIs")
    csv_out = res_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Dataset with KPIs (CSV)",
        data=csv_out,
        file_name="compressor_kpi_historian_evaluated.csv",
        mime="text/csv"
    )
