import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Span, Div, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import linear_palette
import colorcet as cc
from streamlit_bokeh import streamlit_bokeh
from scipy.stats import linregress
from bokeh.layouts import column
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit, prange

# ==========================================
# 0. GLOBAL CONSTANTS
# ==========================================
PLOT_OPTIONS = {
    "Population Dynamics": "Tracks the normalized biomass (B/B₀) over time. Frozen baselines appear as dashed lines.",
    "Survival Probability": "Percentage of droplets remaining 'alive' (Biomass > 1) over time.",
    "MIC vs Volume (Inoculum Effect)": "The Minimum Inhibitory Concentration (MIC) required to kill bacteria at different volumes.",
    "Droplet Distribution": "Histogram comparing total droplet sizes vs. occupied droplet sizes.",
    "Initial Density & Vc": "Scatter plot of initial bacterial density vs volume, identifying the Critical Volume (Vc).",
    "Fold Change": "Log2 Fold Change of biomass (Final/Initial) vs Volume.",
    "N0 vs Volume": "Initial biomass (N0) plotted against Droplet Volume.",
    "Net Growth Rate (μ - λ)": "The net growth rate (Growth μ - Lysis λ) over time.",
    "Substrate Dynamics": "Depletion of substrate (S) over time.",
    "Antibiotic Dynamics": "Effective antibiotic concentration (free or bound) over time.",
    "Density Dynamics": "Bacterial density (Biomass/Volume) evolution over time.",
    "Bound Antibiotic": "Number of antibiotic molecules bound per droplet over time.",
    "Growth/Death Heatmap": "Landscape of survival (Fold Change) across Volume and Antibiotic Concentration."
}

PLOT_LIST = list(PLOT_OPTIONS.keys())

# ==========================================
# 1. PAGE CONFIG & HELPERS
# ==========================================

def configure_page():
    st.set_page_config(
        page_title="μ-SPLASH Simulation", 
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Custom CSS for better containers
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem;}
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .highlight-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #ff4b4b;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ODE MATH MODELS
# ==========================================

@njit(cache=True, fastmath=True, nogil=True)
def vec_effective_concentration(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n):
    y = y_flat.reshape((N, 5))
    A_free, A_bound, B_live, _, S = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
    density = B_live / V
    A_eff = A_bound / np.maximum(density, 1e-12)
    mu = mu_max * S / (Ks + S)
    A_eff_n = np.power(A_eff, n)
    hill_term = A_eff_n / (K_D ** n + A_eff_n + 1e-12)
    lambda_D = lambda_max * hill_term
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1.0 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    dY = np.stack((dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt), axis=1)
    return dY.flatten()

@njit(cache=True, fastmath=True, nogil=True)
def vec_linear_lysis(y_flat, t, N, V, A0_vec, mu_max, Ks, Y, a, b, K_A0, n):
    y = y_flat.reshape((N, 3))
    B_live, _, S = y[:, 0], y[:, 1], y[:, 2]
    density = B_live / V
    mu = mu_max * S / (Ks + S)
    A0_n = np.power(A0_vec, n)
    term_A0 = A0_n / (K_A0 ** n + A0_n + 1e-12)
    lambda_D = a * term_A0 * mu + b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1.0 / Y) * mu * density
    dY = np.stack((dB_live_dt, dB_dead_dt, dS_dt), axis=1)
    return dY.flatten()

@njit(cache=True, fastmath=True, nogil=True)
def vec_combined_model(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b):
    y = y_flat.reshape((N, 5))
    A_free, A_bound, B_live, _, S = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
    density = B_live / V
    A_eff = A_bound / np.maximum(density, 1e-12)
    mu = mu_max * S / (Ks + S)
    A_eff_n = np.power(A_eff, n)
    hill_term = A_eff_n / (K_D ** n + A_eff_n + 1e-12)
    lambda_D = a * hill_term * mu + b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1.0 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    dY = np.stack((dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt), axis=1)
    return dY.flatten()

# ==========================================
# 3. HELPER: LOAD HISTORY & CALLBACKS
# ==========================================

def load_params_from_history(row):
    mapping = {
        'model': 'model_select',
        't_start': 't_start', 't_end': 't_end', 'dt': 'dt',
        'mean_log10': 'mean_log10', 'std_log10': 'std_log10', 'n_samples': 'n_samples',
        'conc_exp': 'conc_exp',
        'mu_max': 'mu_max', 'Y': 'Y', 'S0': 'S0', 'Ks': 'Ks',
        'A0': 'A0',
        'K_on': 'K_on', 'K_off': 'K_off', 'K_D': 'K_D', 
        'n_hill': 'n_hill_1', 
        'lambda_max': 'lambda_max',
        'a': 'a', 'b': 'b', 'K_A0': 'K_A0',
        'mic_threshold': 'mic_threshold'
    }
    for col, state_key in mapping.items():
        if col in row and row[col] is not None:
            if col == 'n_hill':
                st.session_state['n_hill_1'] = row[col]
                st.session_state['n_hill_2'] = row[col]
                st.session_state['n_hill'] = row[col]
            else:
                try:
                    val = row[col]
                    if isinstance(val, (np.integer, int)): val = int(val)
                    elif isinstance(val, (np.floating, float)): val = float(val)
                    st.session_state[state_key] = val
                except Exception:
                    pass

def on_rerun_click():
    if "history_table" in st.session_state and st.session_state["history_table"]["selection"]["rows"]:
        idx = st.session_state["history_table"]["selection"]["rows"][0]
        if idx < len(st.session_state.run_history):
            selected_row = st.session_state.run_history[idx]
            load_params_from_history(selected_row)
            st.session_state.trigger_run = True

def on_freeze_baseline():
    if st.session_state.sim_results is not None:
        st.session_state.baseline_results = st.session_state.sim_results.copy()
        st.toast("✅ Baseline Frozen!", icon="❄️")

def on_clear_baseline():
    st.session_state.baseline_results = None
    st.toast("Baseline Cleared.", icon="🧹")

# ==========================================
# 4. EQUATION LAB (IMPROVED & DEPTH)
# ==========================================
def render_equation_lab_page():
    st.title("📐 Equation Lab")
    st.markdown("A deep dive into the mathematical framework driving the μ-SPLASH simulation.")
    
    st.info("💡 **Core Concept:** These models calculate the battle between **Growth ($\mu$)** and **Lysis ($\lambda$)**. The net change in biomass is $dB/dt = (\mu - \lambda)B$.")

    tab1, tab2, tab3 = st.tabs(["1️⃣ Effective Concentration", "2️⃣ Linear Lysis", "3️⃣ Combined Model"])

    with tab1:
        st.header("Effective Concentration Model")
        st.markdown("""
        **Philosophy:** Antibiotics are finite resources. In a small droplet, if you have many bacteria, there isn't enough antibiotic to go around. 
        This model tracks the specific number of antibiotic molecules bound **per cell**.
        """)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Differential Equations")
            st.latex(r"\frac{dA_{free}}{dt} = \underbrace{-k_{on} A_{free} \frac{B}{V}}_{\text{Binding}} + \underbrace{k_{off} A_{bound}}_{\text{Unbinding}} + \underbrace{\lambda A_{bound}}_{\text{Recycling from dead}}")
            st.latex(r"\frac{dB}{dt} = (\mu - \lambda) B")
            
        with c2:
            st.subheader("The 'Inoculum Effect' Term")
            st.markdown(r"""
            The key driver of the inoculum effect is **Density ($B/V$)**.
            
            $$A_{eff} = \frac{A_{bound}}{\text{Density}}$$

            As Density ($\uparrow$) increases, the Effective Antibiotic per cell ($A_{eff}$) decreases ($\downarrow$), leading to survival in small droplets.
            """)

        st.markdown("---")
        st.subheader("Parameter Definitions")
        p1, p2, p3 = st.columns(3)
        p1.markdown("**$K_{on}$ / $K_{off}$**\n\nBinding kinetics. Determines how fast antibiotic binds to the cell targets.")
        p2.markdown("**$K_D$ (Dissociation Constant)**\n\nThe antibiotic concentration at which 50% of the maximum kill rate is achieved. Lower $K_D$ = More potent drug.")
        p3.markdown("**$n$ (Hill Coefficient)**\n\nDetermines the steepness of the kill curve. High $n$ means the drug is 'all-or-nothing'.")

    with tab2:
        st.header("Linear Lysis Rate")
        st.markdown("""
        **Philosophy:** This is a phenomenological model. It assumes that antibiotic killing is **growth-dependent** (e.g., Beta-lactams like Penicillin only kill dividing cells).
        It does *not* track binding kinetics or depletion, making it computationally faster but less mechanistic regarding the inoculum effect.
        """)

        col_math, col_expl = st.columns([1, 1])
        with col_math:
            st.latex(r"\lambda = \underbrace{\left( a \cdot \frac{A_0^n}{K_{A0}^n + A_0^n} \right)}_{\text{Drug Term}} \times \underbrace{\mu}_{\text{Growth Term}} + b")
        
        with col_expl:
            st.markdown("""
            - **$A_0$**: External antibiotic concentration (assumed constant).
            - **$\mu$**: Current growth rate. If $\mu=0$ (no growth), lysis is minimal ($\approx b$).
            - **$a$**: Slope factor. How strongly does growth translate to death?
            """)

    with tab3:
        st.header("Combined Hybrid Model")
        st.markdown("""
        **Philosophy:** The "Best of Both Worlds". It uses the **Depletion/Binding** mechanics of Model 1 to calculate $A_{eff}$, but uses the **Growth-Dependence** of Model 2 for the lysis rate.
        """)
        st.latex(r"\lambda = \left( a \cdot \underbrace{\frac{A_{eff}^n}{K_D^n + A_{eff}^n}}_{\text{Binding Logic}} \cdot \mu \right) + b")
        st.warning("Use this if you suspect the drug is depleted by biomass (inoculum effect) BUT only works on actively dividing cells.")

# ==========================================
# 5. UI COMPONENTS (SIDEBAR)
# ==========================================

def render_sidebar():
    # --- NAVIGATION (Top Level) ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", ["🚀 Simulator", "📐 Equation Lab"], horizontal=True, label_visibility="collapsed")
    st.sidebar.markdown("---")

    if app_mode == "Equation Lab":
        return None, "Equation Lab"

    st.sidebar.header("⚙️ Configuration")
    params = {}

    # Model Selector
    params['model'] = st.sidebar.selectbox(
        "Mathematical Model", 
        ["Effective Concentration", "Linear Lysis Rate", "Combined Model"], 
        key='model_select'
    )

    # 1. Time Settings
    with st.sidebar.expander("⏱️ Time Settings", expanded=False):
        col1, col2 = st.columns(2)
        params['t_start'] = col1.number_input("Start (h)", value=st.session_state.get('t_start', 0.0), step=1.0, key='t_start')
        params['t_end'] = col2.number_input("End (h)", value=st.session_state.get('t_end', 24.0), step=1.0, key='t_end')
        params['dt'] = st.number_input("Step size (h)", value=st.session_state.get('dt', 1.0), min_value=0.01, step=0.5, key='dt')

    # 2. Inoculum
    with st.sidebar.expander("💧 Inoculum & Droplets", expanded=True):
        params['n_samples'] = st.number_input("Total Droplets (N)", 1000, 100000, st.session_state.get('n_samples', 17000), 1000, key='n_samples')
        c1, c2 = st.columns(2)
        params['mean_log10'] = c1.number_input("Mean Log10(Vol)", 1.0, 8.0, st.session_state.get('mean_log10', 3.0), 0.1, key='mean_log10')
        params['std_log10'] = c2.number_input("Std Dev", 0.1, 3.0, st.session_state.get('std_log10', 1.2), 0.1, key='std_log10')
        params['conc_exp'] = st.slider("Inoculum Conc (10^x)", -7.0, -1.0, st.session_state.get('conc_exp', -4.3), 0.1, key='conc_exp')
        params['concentration'] = 10 ** params['conc_exp']

    # 3. Biology
    with st.sidebar.expander("🧫 Bacterial Physiology", expanded=False):
        c1, c2 = st.columns(2)
        params['mu_max'] = c1.number_input("μ_max", value=st.session_state.get('mu_max', 0.7), key='mu_max')
        params['Ks'] = c2.number_input("Ks", value=st.session_state.get('Ks', 2.0), key='Ks')
        c3, c4 = st.columns(2)
        params['Y'] = c3.number_input("Yield", value=st.session_state.get('Y', 0.001), format="%.4f", key='Y')
        params['S0'] = c4.number_input("S0", value=st.session_state.get('S0', 1.0), key='S0')

    # 4. Pharmacodynamics
    with st.sidebar.expander("💊 Pharmacodynamics", expanded=True):
        params['A0'] = st.number_input("Initial Antibiotic (A0)", value=st.session_state.get('A0', 10.0), key='A0')
        params['mic_threshold'] = st.slider("Death Threshold (Log2 FC)", -6.0, 0.0, st.session_state.get('mic_threshold', -1.0), 0.1, key='mic_threshold')

        defaults = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']
        for key in defaults: params[key] = 0.0

        if params['model'] in ["Effective Concentration", "Combined Model"]:
            c1, c2 = st.columns(2)
            params['K_on'] = c1.number_input("K_on", value=st.session_state.get('K_on', 750.0), key='K_on')
            params['K_off'] = c2.number_input("K_off", value=st.session_state.get('K_off', 0.01), key='K_off')
            params['K_D'] = st.number_input("K_D", value=st.session_state.get('K_D', 12000.0), key='K_D')
            if 'n_hill' not in st.session_state: st.session_state.n_hill = 20.0
            params['n_hill'] = st.number_input("Hill (n)", value=st.session_state.get('n_hill_1', st.session_state.n_hill), key='n_hill_1')

        if params['model'] == "Effective Concentration":
            params['lambda_max'] = st.number_input("λ_max", value=st.session_state.get('lambda_max', 1.0), key='lambda_max')

        if params['model'] in ["Linear Lysis Rate", "Combined Model"]:
            c1, c2 = st.columns(2)
            params['a'] = c1.number_input("Slope (a)", value=st.session_state.get('a', 3.0), key='a')
            params['b'] = c2.number_input("Intercept (b)", value=st.session_state.get('b', 0.1), key='b')

        if params['model'] == "Linear Lysis Rate":
            params['K_A0'] = st.number_input("K_A0", value=st.session_state.get('K_A0', 10.0), key='K_A0')
            val_n_hill = st.session_state.get('n_hill_2', 20.0)
            params['n_hill'] = st.number_input("Hill (n)", value=val_n_hill, key='n_hill_2')

        if 'n_hill_1' in params: params['n_hill'] = params['n_hill_1']
        elif 'n_hill_2' in params: params['n_hill'] = params['n_hill_2']

    return params, "Simulator"

# ==========================================
# 6. CORE LOGIC
# ==========================================

@njit(cache=True, parallel=True, fastmath=True)
def _generate_population_fast(n, mean, std, conc, mean_pix, std_pix):
    log_data = np.random.normal(mean, std, int(n))
    volume_data = 10 ** log_data
    mask_vol = (volume_data >= 1000) & (volume_data <= 1e8)
    trimmed_vol = volume_data[mask_vol]
    lambdas = trimmed_vol * conc
    cell_counts = np.zeros(len(lambdas), dtype=np.int64)
    for i in prange(len(lambdas)):
        cell_counts[i] = np.random.poisson(lambdas[i])
    occupied_mask = cell_counts > 0
    final_vols = trimmed_vol[occupied_mask].copy()
    final_counts = cell_counts[occupied_mask].copy()
    base_biomass = final_counts * mean_pix
    noise_base = np.random.normal(0.0, 1.0, len(final_counts))
    noise_scale = np.sqrt(final_counts) * std_pix
    noise = noise_base * noise_scale
    raw_biomass = base_biomass + noise
    final_biomass = np.round(raw_biomass)
    final_biomass = np.maximum(final_biomass, 1.0)
    return final_vols, final_counts, final_biomass, trimmed_vol

@st.cache_data(show_spinner="Generating population...")
def generate_population(mean, std, n, conc, mean_pix, std_pix):
    return _generate_population_fast(n, mean, std, conc, mean_pix, std_pix)

def calculate_vc_and_density(vols, biomass, theoretical_conc, mean_pix):
    if len(vols) == 0: return pd.DataFrame(), 0.0
    effective_count = biomass / mean_pix
    df = pd.DataFrame({'Volume': vols, 'Biomass': biomass, 'Count': effective_count})
    df['InitialDensity'] = df['Count'] / df['Volume']
    df = df.sort_values(by='Volume').reset_index(drop=True)
    log_density = np.log10(df['InitialDensity'])
    df['RollingLogMean'] = log_density.rolling(window=100, min_periods=1).mean()
    df['RollingMeanDensity'] = 10 ** df['RollingLogMean']
    convergence_window = 2
    tolerance = 0.05
    differences = np.abs(1 - (df['RollingMeanDensity'] / theoretical_conc))
    rolling_diff = differences.rolling(window=convergence_window).mean()
    met_conditions = np.where(rolling_diff <= tolerance)[0]
    if len(met_conditions) > 0: closest_index = met_conditions[0]
    else: closest_index = differences.idxmin()
    vc_val = df.loc[closest_index, 'Volume']
    return df, vc_val

@njit(cache=True, fastmath=True)
def calculate_batch_lambda(sol_reshaped, t_eval, vols, model_type_int,
                           mu_max, Ks, K_D, n_hill, lambda_max, A0, K_A0, a, b):
    if model_type_int == 1:
        S = sol_reshaped[:, :, 2]
    else:
        A_bound = sol_reshaped[:, :, 1]
        B_live = sol_reshaped[:, :, 2]
        S = sol_reshaped[:, :, 4]
    density = B_live / vols
    mu_mat = mu_max * S / (Ks + S)
    if model_type_int == 0:
        A_eff = A_bound / np.maximum(density, 1e-12)
        A_eff_n = np.power(A_eff, n_hill)
        K_D_n = K_D ** n_hill
        hill = A_eff_n / (K_D_n + A_eff_n + 1e-12)
        lambda_matrix = lambda_max * hill
    elif model_type_int == 1:
        A0_n = np.power(A0, n_hill)
        K_A0_n = K_A0 ** n_hill
        term_A0 = A0_n / (K_A0_n + A0_n + 1e-12)
        lambda_matrix = a * term_A0 * mu_mat + b
    elif model_type_int == 2:
        A_eff = A_bound / np.maximum(density, 1e-12)
        A_eff_n = np.power(A_eff, n_hill)
        K_D_n = K_D ** n_hill
        hill = A_eff_n / (K_D_n + A_eff_n + 1e-12)
        lambda_matrix = a * hill * mu_mat + b
    else:
        lambda_matrix = np.zeros_like(mu_mat)
    return lambda_matrix

@njit(cache=True, fastmath=True)
def calculate_derived_metrics(sol_reshaped, vols, model_type_int, mu_max, Ks):
    if model_type_int == 1:
        B_live = sol_reshaped[:, :, 0]
        S = sol_reshaped[:, :, 2]
        A_bound = np.zeros_like(B_live)
    else:
        A_bound = sol_reshaped[:, :, 1]
        B_live = sol_reshaped[:, :, 2]
        S = sol_reshaped[:, :, 4]
    density = B_live / vols
    if model_type_int == 1: A_eff = np.zeros_like(density)
    else: A_eff = A_bound / np.maximum(density, 1e-12)
    mu_mat = mu_max * S / (Ks + S)
    return A_eff, mu_mat, density, A_bound, B_live, S

@njit(cache=True, fastmath=True)
def fast_accumulate_bins_serial(bin_sums, a_eff_bin_sums, density_bin_sums,
                                  a_bound_bin_sums, net_rate_bin_sums, s_bin_sums,
                                  alive_bin_sums,
                                  bin_counts, bin_edges, vols,
                                  batch_blive_T, batch_a_eff_T, batch_density_T,
                                  batch_abound_T, batch_net_rate, batch_S_T,
                                  batch_alive_mask_T):
    n_droplets = len(vols)
    n_bins = len(bin_edges) - 1
    for i in range(n_droplets):
        vol = vols[i]
        bin_idx = -1
        for b in range(n_bins):
            if vol >= bin_edges[b] and vol < bin_edges[b + 1]:
                bin_idx = b
                break
        if bin_idx != -1:
            bin_sums[bin_idx, :] += batch_blive_T[i, :]
            a_eff_bin_sums[bin_idx, :] += batch_a_eff_T[i, :]
            density_bin_sums[bin_idx, :] += batch_density_T[i, :]
            a_bound_bin_sums[bin_idx, :] += batch_abound_T[i, :]
            net_rate_bin_sums[bin_idx, :] += batch_net_rate[i, :]
            s_bin_sums[bin_idx, :] += batch_S_T[i, :]
            alive_bin_sums[bin_idx, :] += batch_alive_mask_T[i, :]
            bin_counts[bin_idx] += 1

def solve_individual_droplet(idx, single_vol, single_biomass, t_eval, params, bin_edges):
    batch_vols = np.array([single_vol])
    batch_biomass = np.array([single_biomass])
    current_batch_size = 1
    N_STEPS = len(t_eval)
    n_bins = len(bin_edges) - 1

    local_bin_sums = np.zeros((n_bins, N_STEPS))
    local_a_eff = np.zeros((n_bins, N_STEPS))
    local_density = np.zeros((n_bins, N_STEPS))
    local_a_bound = np.zeros((n_bins, N_STEPS))
    local_net_rate = np.zeros((n_bins, N_STEPS))
    local_s_sums = np.zeros((n_bins, N_STEPS))
    local_alive_sums = np.zeros((n_bins, N_STEPS))
    local_bin_counts = np.zeros(n_bins)
    
    model = params['model']
    if model == "Effective Concentration":
        model_int = 0
        num_vars = 5
        func = vec_effective_concentration
    elif model == "Linear Lysis Rate":
        model_int = 1
        num_vars = 3
        func = vec_linear_lysis
    else:
        model_int = 2
        num_vars = 5
        func = vec_combined_model

    args = ()
    y0_flat = None

    if model == "Effective Concentration":
        y0_mat = np.zeros((current_batch_size, 5))
        y0_mat[:, 0] = params['A0']
        y0_mat[:, 2] = batch_biomass
        y0_mat[:, 4] = params['S0']
        y0_flat = y0_mat.flatten()
        args = (current_batch_size, batch_vols, params['mu_max'], params['Ks'], params['Y'],
                params['K_on'], params['K_off'], params['lambda_max'], params['K_D'], params['n_hill'])
    elif model == "Linear Lysis Rate":
        y0_mat = np.zeros((current_batch_size, 3))
        y0_mat[:, 0] = batch_biomass
        y0_mat[:, 2] = params['S0']
        y0_flat = y0_mat.flatten()
        A0_vec = np.full(current_batch_size, params['A0'])
        args = (current_batch_size, batch_vols, A0_vec, params['mu_max'], params['Ks'], params['Y'],
                params['a'], params['b'], params['K_A0'], params['n_hill'])
    elif model == "Combined Model":
        y0_mat = np.zeros((current_batch_size, 5))
        y0_mat[:, 0] = params['A0']
        y0_mat[:, 2] = batch_biomass
        y0_mat[:, 4] = params['S0']
        y0_flat = y0_mat.flatten()
        args = (current_batch_size, batch_vols, params['mu_max'], params['Ks'], params['Y'],
                params['K_on'], params['K_off'], params['K_D'], params['n_hill'], params['a'], params['b'])

    sol = odeint(func, y0_flat, t_eval, args=args, rtol=1e-3, atol=1e-6)
    sol_reshaped = sol.reshape(N_STEPS, current_batch_size, num_vars)

    batch_lambda_vals = calculate_batch_lambda(
        sol_reshaped, t_eval, batch_vols, model_int,
        params['mu_max'], params['Ks'], params['K_D'], params['n_hill'],
        params.get('lambda_max', 0), params['A0'], params.get('K_A0', 0),
        params.get('a', 0), params.get('b', 0)
    )
    batch_lambda_T = batch_lambda_vals.T

    (batch_a_eff, batch_mu, batch_density,
        batch_abound, batch_blive_cont, batch_S) = calculate_derived_metrics(
        sol_reshaped, batch_vols, model_int, params['mu_max'], params['Ks']
    )

    batch_a_eff_T = batch_a_eff.T
    batch_density_T = batch_density.T
    batch_mu_T = batch_mu.T
    batch_abound_T = batch_abound.T
    batch_S_T = batch_S.T

    if model == "Linear Lysis Rate":
        batch_a_eff_T[:] = params['A0']

    batch_net_rate = batch_mu_T - batch_lambda_T
    batch_blive = np.round(batch_blive_cont)
    batch_blive_T = batch_blive.T
    
    batch_alive_mask = (batch_blive > 1.0).astype(np.float64)
    batch_alive_mask_T = batch_alive_mask.T

    final_c = np.mean(batch_blive[-2:, :], axis=0) if N_STEPS > 2 else batch_blive[-1, :]
    final_c = np.where(final_c < 2.0, 0.0, final_c)

    fast_accumulate_bins_serial(
        local_bin_sums, local_a_eff, local_density,
        local_a_bound, local_net_rate, local_s_sums,
        local_alive_sums,
        local_bin_counts, bin_edges, batch_vols,
        batch_blive_T, batch_a_eff_T, batch_density_T,
        batch_abound_T, batch_net_rate, batch_S_T,
        batch_alive_mask_T
    )

    return (idx, final_c[0], local_bin_sums, local_bin_counts, 
            local_a_eff, local_density, local_a_bound, local_net_rate, local_s_sums, local_alive_sums)

def _compute_simulation_core(vols, initial_biomass, total_vols_range, params):
    t_eval = np.arange(params['t_start'], params['t_end'] + params['dt'] / 100.0, params['dt'])
    if len(t_eval) < 2: t_eval = np.linspace(params['t_start'], params['t_end'], 2)
    N_STEPS = len(t_eval)
    N_occupied = len(vols)
    min_exp = int(np.floor(np.log10(total_vols_range[0])))
    max_exp = int(np.ceil(np.log10(total_vols_range[1])))
    bin_edges_log = np.arange(min_exp, max_exp + 1)
    bin_edges = 10 ** bin_edges_log
    n_bins = len(bin_edges) - 1
    total_bin_sums = np.zeros((n_bins, N_STEPS))
    total_a_eff = np.zeros((n_bins, N_STEPS))
    total_density = np.zeros((n_bins, N_STEPS))
    total_a_bound = np.zeros((n_bins, N_STEPS))
    total_net_rate = np.zeros((n_bins, N_STEPS))
    total_s_sums = np.zeros((n_bins, N_STEPS))
    total_alive_sums = np.zeros((n_bins, N_STEPS))
    total_bin_counts = np.zeros(n_bins)
    final_counts_all = np.zeros(N_occupied)

    show_progress = (N_occupied > 100)
    if show_progress:
        progress_bar = st.progress(0, text="Initializing parallel simulation...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(N_occupied):
            futures.append(executor.submit(
                solve_individual_droplet, i, vols[i], initial_biomass[i], t_eval, params, bin_edges
            ))
        completed_count = 0
        for future in as_completed(futures):
            try:
                (idx, final_c_val, l_sums, l_counts, l_a_eff, l_dens, l_abound, l_net, l_s, l_alive) = future.result()
                final_counts_all[idx] = final_c_val
                total_bin_sums += l_sums
                total_bin_counts += l_counts
                total_a_eff += l_a_eff
                total_density += l_dens
                total_a_bound += l_abound
                total_net_rate += l_net
                total_s_sums += l_s
                total_alive_sums += l_alive
                completed_count += 1
                if show_progress and (completed_count % 50 == 0 or completed_count == N_occupied):
                    elapsed_sec = time.time() - start_time
                    prog_val = min(completed_count / N_occupied, 1.0)
                    progress_bar.progress(prog_val, text=f"Droplet {completed_count}/{N_occupied} processed")
            except Exception as e:
                st.error(f"Simulation failed for droplet: {e}")
                if show_progress: progress_bar.empty()
                return None
    
    if show_progress: progress_bar.empty()
    return (total_bin_sums, total_bin_counts, final_counts_all, t_eval, bin_edges,
            total_a_eff, total_density, total_a_bound, total_net_rate, total_s_sums, total_alive_sums)

def run_simulation(vols, initial_biomass, total_vols_range, params):
    return _compute_simulation_core(vols, initial_biomass, total_vols_range, params)

# ==========================================
# 7. PLOTTING FUNCTIONS
# ==========================================

def int_to_superscript(n):
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))

def plot_heatmap(conc_grid, vol_centers, data_matrix):
    x_list = []
    y_list = []
    c_list = []
    w = (np.log10(vol_centers.max()) - np.log10(vol_centers.min())) / len(vol_centers)
    h = (conc_grid[1] - conc_grid[0]) if len(conc_grid) > 1 else 1.0

    for i, conc in enumerate(conc_grid):
        for j, vol in enumerate(vol_centers):
            val = data_matrix[i, j]
            if not np.isnan(val):
                x_list.append(np.log10(vol))
                y_list.append(conc)
                c_list.append(val)

    source = ColumnDataSource(data={'x': x_list, 'y': y_list, 'fc': c_list, 'vol': 10**np.array(x_list)})
    mapper = linear_cmap(field_name='fc', palette=cc.CET_D1[::-1], low=-5, high=5)

    p = figure(
        title="Survival Landscape: Volume vs Antibiotic Dose",
        x_axis_label="Volume (µm³)",
        y_axis_label="Antibiotic Concentration (µg/ml)",
        width=1200, height=800,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above"
    )
    p.rect(x='x', y='y', width=w*1.02, height=h*0.98, source=source, fill_color=mapper, line_color=None)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0), title="Log2 FC")
    p.add_layout(color_bar, 'right')
    
    overrides = {}
    for i in range(15): overrides[i] = f"10{int_to_superscript(i)}"
    p.xaxis.major_label_overrides = overrides
    return p

def plot_mic_vs_volume(heatmap_data, params):
    matrix = heatmap_data['matrix']
    concs = heatmap_data['conc_grid']
    vols = heatmap_data['vol_centers']
    death_thresh = params.get('mic_threshold', -1.0)
    
    mic_values = []
    vol_values = []
    
    for j in range(matrix.shape[1]):
        col_fc = matrix[:, j]
        death_indices = np.where(col_fc < death_thresh)[0]
        if len(death_indices) > 0:
            mic_idx = death_indices[0]
            mic_values.append(concs[mic_idx])
            vol_values.append(vols[j])
        else:
            mic_values.append(concs[-1]) 
            vol_values.append(vols[j])
            
    source = ColumnDataSource(data={'vol': vol_values, 'mic': mic_values})
    p = figure(
        title=f"Inoculum Effect: Effective MIC vs Droplet Volume (Death Threshold FC < {death_thresh})",
        x_axis_label="Droplet Volume (µm³)",
        y_axis_label="Effective MIC (µg/ml)",
        x_axis_type="log",
        width=1200, height=800,
        tools="pan,wheel_zoom,reset,save"
    )
    p.line('vol', 'mic', source=source, line_width=4, color="#ff4b4b", legend_label="MIC")
    p.scatter('vol', 'mic', source=source, size=6, color="white")
    
    hover = HoverTool(tooltips=[("Volume", "@vol{0,0}"), ("MIC", "@mic{0.00}")])
    p.add_tools(hover)
    p.legend.location = "top_right"
    return p

def plot_survival_probability(t_eval, alive_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Survival Probability (%)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Time-to-Extinction: Survival Probability over Time")
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, unique_bins) if unique_bins > 0 else []
    legend_items = []
    color_idx = 0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            alive_traj = (alive_bin_sums[i, :] / bin_counts[i]) * 100.0
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)} (n={int(bin_counts[i])})"
            r = p.line(t_eval, alive_traj, line_color=colors[color_idx], line_width=3, alpha=0.9)
            legend_items.append((label, [r]))
            color_idx += 1
    legend = Legend(items=legend_items, title="Volume Bins", click_policy="hide")
    p.add_layout(legend, 'right')
    return p

def plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges, baseline_data=None):
    p = figure(x_axis_label="Time (h)", y_axis_label="Normalized Biomass (B/B₀)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save")
    
    if baseline_data:
        b_sums, b_counts, _, _, b_edges, _, _, _, _, _, _ = baseline_data["sim_output"]
        unique_bins_b = sum(1 for c in b_counts if c > 0)
        colors_b = linear_palette([cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]], unique_bins_b) if unique_bins_b > 0 else []
        c_idx_b = 0
        for i in range(len(b_counts)):
            if b_counts[i] > 0:
                mean_traj = b_sums[i, :] / b_counts[i]
                initial_val = mean_traj[0]
                norm_traj = mean_traj / initial_val if initial_val > 1e-9 else mean_traj
                r_base = p.line(t_eval, norm_traj, line_color=colors_b[c_idx_b], line_width=2, line_dash="dashed", alpha=0.4)
                c_idx_b += 1
        dummy_line = p.line([], [], line_color="grey", line_dash="dashed", alpha=0.5, line_width=2)
        p.add_layout(Legend(items=[("Baseline (Previous Run)", [dummy_line])], location="top_left"))

    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, unique_bins) if unique_bins > 0 else []
    legend_items = []
    color_idx = 0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_traj = bin_sums[i, :] / bin_counts[i]
            initial_val = mean_traj[0]
            norm_traj = mean_traj / initial_val if initial_val > 1e-9 else mean_traj
            norm_traj = np.where(norm_traj <= 0, np.nan, norm_traj)
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)} (n={int(bin_counts[i])})"
            r = p.line(t_eval, norm_traj, line_color=colors[color_idx], line_width=3, alpha=0.9)
            legend_items.append((label, [r]))
            color_idx += 1
    
    total_biomass_traj = np.sum(bin_sums, axis=0)
    total_N0 = total_biomass_traj[0]
    meta_norm = total_biomass_traj / total_N0 if total_N0 > 1e-9 else total_biomass_traj
    r_meta = p.line(t_eval, meta_norm, line_color="white", line_width=4,
                    line_dash="solid", alpha=1.0)
    legend_items.insert(0, ("Metapopulation (Avg)", [r_meta]))
    
    legend = Legend(items=legend_items, title="Volume Bins", click_policy="hide")
    p.add_layout(legend, 'right')
    return p

def plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Net Growth Rate (μ - λ) [1/h]",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Net Growth Rate (μ - λ) Dynamics")
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []
    color_idx = 0
    legend_items = []
    zero_line = Span(location=0, dimension='width', line_color='white', line_dash='dotted', line_width=2)
    p.add_layout(zero_line)
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_net_rate = net_rate_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_net_rate, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1
    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_a_eff_dynamics(t_eval, a_eff_bin_sums, bin_counts, bin_edges, params):
    title_text = "Effective Antibiotic Conc. (Bound/Density)"
    y_label = "Effective Conc. (A_eff)"
    if params['model'] == "Linear Lysis Rate":
        title_text = "External Antibiotic Concentration (A0)"
        y_label = "Concentration (A0)"
    p = figure(x_axis_label="Time (h)", y_axis_label=y_label,
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title=title_text)
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []
    color_idx = 0
    legend_items = []
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_a_eff = a_eff_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_a_eff, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1
    threshold_val = 0.0
    label_text = "Threshold"
    if params['model'] == "Linear Lysis Rate":
        threshold_val = params['K_A0']
        label_text = "K_A0"
    else:
        threshold_val = params['K_D']
        label_text = "K_D"
    thresh_line = Span(location=threshold_val, dimension='width', line_color='red',
                       line_dash='dotted', line_width=3)
    p.add_layout(thresh_line)
    r_thresh_dummy = p.line([], [], line_color='red', line_dash='dotted', line_width=3)
    legend_items.insert(0, (f"{label_text} ({threshold_val:.0f})", [r_thresh_dummy]))
    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Cell Density (Biomass/Volume)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Average Cell Density over Time", y_axis_type='log')
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []
    color_idx = 0
    legend_items = []
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_vals = density_bin_sums[i, :] / bin_counts[i]
            mean_vals = np.where(mean_vals <= 0, np.nan, mean_vals)
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_vals, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1
    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_substrate_dynamics(t_eval, s_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Substrate Concentration (S)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Substrate Depletion over Time")
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []
    color_idx = 0
    legend_items = []
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_vals = s_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_vals, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1
    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_abound_dynamics(t_eval, abound_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Bound Antibiotic (Molecules/Droplet)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Average Bound Antibiotic (A_bound) over Time")
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []
    color_idx = 0
    legend_items = []
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_vals = abound_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i + 1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_vals, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1
    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_distribution(total_vols, occupied_vols):
    min_exp = int(np.floor(np.log10(total_vols.min())))
    max_exp = int(np.ceil(np.log10(total_vols.max())))
    log_bins = np.linspace(min_exp, max_exp, 30)
    hist_total, edges_total = np.histogram(np.log10(total_vols), bins=log_bins)
    hist_occ, _ = np.histogram(np.log10(occupied_vols), bins=log_bins)
    edges_linear = 10 ** edges_total
    p = figure(x_axis_label="Volume", y_axis_label="Frequency",
               x_axis_type="log", height=800, width=1200, tools="pan,wheel_zoom,reset,save")
    p.quad(top=hist_total, bottom=0, left=edges_linear[:-1], right=edges_linear[1:],
           fill_color="grey", line_color="white", alpha=0.5, legend_label="Total Droplets")
    p.quad(top=hist_occ, bottom=0, left=edges_linear[:-1], right=edges_linear[1:],
           fill_color="#718dbf", line_color="white", alpha=0.6, legend_label="Occupied Droplets")
    p.legend.location = "top_right"
    return p

def plot_initial_density_vc(df_density, vc_val, theoretical_density):
    source = ColumnDataSource(df_density)
    p = figure(x_axis_type='log', y_axis_type='log',
               x_axis_label='Volume (μm³)', y_axis_label='Initial Density (biomass/μm³)',
               width=1200, height=800, output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    r_dens = p.scatter('Volume', 'InitialDensity', source=source, color='silver', alpha=0.6, size=4)
    r_roll = p.line(df_density['Volume'], df_density['RollingMeanDensity'], color='red', line_width=3)
    min_v, max_v = df_density['Volume'].min(), df_density['Volume'].max()
    r_theo = p.line([min_v, max_v], [theoretical_density, theoretical_density], color='white', line_width=3)
    p.add_layout(Span(location=vc_val, dimension='height', line_color='white', line_dash='dashed', line_width=3))
    r_vc_dum = p.line([min_v, max_v], [theoretical_density, theoretical_density],
                      color='white', line_dash='dashed', line_width=3, visible=False)
    legend = Legend(items=[
        LegendItem(label='Initial Density', renderers=[r_dens]),
        LegendItem(label='Rolling Mean', renderers=[r_roll]),
        LegendItem(label='Inoculum Density', renderers=[r_theo]),
        LegendItem(label='Vc', renderers=[r_vc_dum])
    ], location='top_right')
    p.add_layout(legend)
    return p

def plot_fold_change(vols, initial_biomass, final_biomass, vc_val):
    min_fc = -6.0
    with np.errstate(divide='ignore', invalid='ignore'):
        fc_raw = final_biomass / initial_biomass
        fc_log2 = np.log2(fc_raw)
    fc_log2 = np.where(np.isneginf(fc_log2) | np.isnan(fc_log2) | (fc_log2 < min_fc), min_fc, fc_log2)
    df_fc = pd.DataFrame({'Volume': vols, 'FoldChange': fc_log2, 'DropletID': np.arange(len(vols))})
    df_fc = df_fc.sort_values(by='Volume').reset_index(drop=True)
    df_sub = df_fc[df_fc['FoldChange'] > min_fc].copy()
    if not df_sub.empty: df_sub['MovingAverage'] = df_sub['FoldChange'].rolling(window=100, min_periods=1).mean()
    else: df_sub['MovingAverage'] = np.nan
    total_initial = np.sum(initial_biomass)
    total_final = np.sum(final_biomass)
    meta_fc = np.log2(total_final / total_initial) if total_final > 0 else min_fc
    source = ColumnDataSource(df_fc)
    sub_source = ColumnDataSource(df_sub)
    p = figure(x_axis_type='log', y_axis_type='linear',
               x_axis_label='Volume (μm³)', y_axis_label='Log2 biomass Fold Change',
               width=1200, height=800, y_range=(-7, 9), output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    r_scat = p.scatter('Volume', 'FoldChange', source=source, color='silver', alpha=0.6, size=4)
    if not df_sub.empty: r_ma = p.line('Volume', 'MovingAverage', source=sub_source, color='red', line_width=3)
    else: r_ma = p.line([], [], color='red')
    min_v, max_v = df_fc['Volume'].min(), df_fc['Volume'].max()
    r_meta = p.line([min_v, max_v], [meta_fc, meta_fc], color='white', line_width=3)
    r_base = p.line([min_v, max_v], [0, 0], color='white', line_dash='dashdot', line_width=3)
    r_vc = p.line([vc_val, vc_val], [-10, 10], color='white', line_dash='dashed', line_width=3)
    p.add_tools(HoverTool(tooltips=[('Volume', '@Volume{0,0}'), ('Fold Change', '@FoldChange{0.00}'), ('ID', '@DropletID')], renderers=[r_scat]))
    legend = Legend(items=[
        LegendItem(label='Droplet FC', renderers=[r_scat]),
        LegendItem(label='FC Moving Avg', renderers=[r_ma]),
        LegendItem(label='Metapopulation FC', renderers=[r_meta]),
        LegendItem(label='Vc', renderers=[r_vc]),
        LegendItem(label='Baseline (0)', renderers=[r_base])
    ], location='top_right')
    p.add_layout(legend, 'right')
    return p

def plot_n0_vs_volume(df, Vc):
    plot_df = df.copy()
    plot_df['DropletID'] = plot_df.index
    source = ColumnDataSource(plot_df)
    p = figure(x_axis_type='log', y_axis_type='log',
               x_axis_label='Volume (μm³)', y_axis_label='Initial Biomass',
               output_backend="webgl", width=1200, height=800,
               tools="pan,wheel_zoom,reset,save")
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    r_scat = p.scatter('Volume', 'Biomass', source=source, color='gray', alpha=0.6, size=5,
                       legend_label='Biomass vs. Volume')
    hover = HoverTool(tooltips=[('Volume', '@Volume{0,0}'), ('Biomass', '@Biomass{0.00}'), ('ID', '@DropletID')], renderers=[r_scat])
    p.add_tools(hover)
    filtered_df = plot_df[(plot_df['Biomass'] > 0) & (plot_df['Volume'] >= Vc)]
    stats_text = "Insufficient data for regression"
    if len(filtered_df) > 2:
        x = np.log10(filtered_df['Volume'])
        y = np.log10(filtered_df['Biomass'])
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_values = np.linspace(plot_df['Volume'].min(), plot_df['Volume'].max(), 100)
        y_values = 10 ** (intercept + slope * np.log10(x_values))
        p.line(x_values, y_values, color='red', legend_label='Linear Regression', line_width=3)
        stats_text = f'<b>Regression (V > Vc):</b><br>y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_value ** 2:.3f}'
    p.legend.location = "top_left"
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {
        'text-align': 'center', 'margin': '10px auto', 'font-size': '14pt',
        'font-family': 'Arial, sans-serif', 'color': 'black',
        'background-color': '#f0f2f6', 'border': '1px solid #ccc',
        'padding': '15px', 'border-radius': '10px', 'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
    }
    return column(p, stats_div)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():
    configure_page()

    MEAN_PIXELS = 5.5
    STD_PIXELS = 1.0

    if "sim_results" not in st.session_state: st.session_state.sim_results = None
    if "baseline_results" not in st.session_state: st.session_state.baseline_results = None
    if "run_history" not in st.session_state: st.session_state.run_history = []
    if "trigger_run" not in st.session_state: st.session_state.trigger_run = False
    
    # 1. Render Sidebar (Handle Navigation)
    sidebar_result, current_mode = render_sidebar()
    
    # If mode is Equation Lab, render it and stop
    if current_mode == "Equation Lab":
        render_equation_lab_page()
        return

    # Otherwise, proceed with Simulator
    params = sidebar_result
    
    # 2. METRICS CONTAINER (Placeholder for Top Dashboard)
    metrics_container = st.container()

    # 3. CONTROLS (Button Below Metrics)
    col_btn, _ = st.columns([1,2])
    with col_btn:
        manual_run = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    # 4. Logic: Run Simulation
    should_run = manual_run or st.session_state.trigger_run or st.session_state.sim_results is None

    if should_run:
        st.session_state.trigger_run = False 
        st.session_state.heatmap_data = None
        
        # --- A. RUN MAIN SIMULATION ---
        with st.spinner("Processing main droplet population..."):
            vols, counts, initial_biomass, total_vols = generate_population(
                params['mean_log10'], params['std_log10'], params['n_samples'],
                params['concentration'], MEAN_PIXELS, STD_PIXELS
            )
            sort_idx = np.argsort(vols)
            vols = vols[sort_idx]
            counts = counts[sort_idx]
            initial_biomass = initial_biomass[sort_idx]
            n_trimmed = len(total_vols)
            N_occupied = len(vols)
            
            sim_output = None
            df_density = pd.DataFrame()
            vc_val = 0.0

            if N_occupied > 0:
                df_density, vc_val = calculate_vc_and_density(vols, initial_biomass, params['concentration'], MEAN_PIXELS)
                sim_output = run_simulation(
                    vols, initial_biomass, (total_vols.min(), total_vols.max()), params
                )
                if sim_output:
                    row = {k: v for k, v in params.items() if isinstance(v, (int, float, str))}
                    row['Timestamp'] = datetime.now().strftime("%H:%M:%S")
                    
                    model = params['model']
                    all_specifics = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']
                    if model == "Effective Concentration": relevant = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max']
                    elif model == "Linear Lysis Rate": relevant = ['a', 'b', 'K_A0', 'n_hill']
                    elif model == "Combined Model": relevant = ['K_on', 'K_off', 'K_D', 'n_hill', 'a', 'b']
                    else: relevant = []
                    for key in all_specifics:
                        if key not in relevant: row[key] = None 

                    if not st.session_state.run_history or row['Timestamp'] != st.session_state.run_history[0].get('Timestamp'):
                           st.session_state.run_history.insert(0, row)
                    if len(st.session_state.run_history) > 20:
                        st.session_state.run_history = st.session_state.run_history[:20]

            st.session_state.sim_results = {
                "vols": vols, "counts": counts, "initial_biomass": initial_biomass,
                "total_vols": total_vols, "n_trimmed": n_trimmed, "N_occupied": N_occupied,
                "df_density": df_density, "vc_val": vc_val, "sim_output": sim_output, "params": params
            }

        # --- B. RUN HEATMAP SCAN (IMMEDIATELY AFTER) ---
        if N_occupied > 0:
            with st.spinner("Calculating Survival Landscape..."):
                min_log = np.log10(total_vols.min())
                max_log = np.log10(total_vols.max())
                vol_grid = np.logspace(min_log, max_log, 50) 
                
                conc_for_generation = params["concentration"] 
                expected_counts = np.maximum(vol_grid * conc_for_generation, 1.0)
                init_biomass_grid = expected_counts * MEAN_PIXELS
                
                n_concs = 20
                max_conc = 40.0
                conc_grid = np.linspace(0, max_conc, n_concs)
                heatmap_matrix = np.zeros((n_concs, len(vol_grid)))
                
                vol_centers = vol_grid 

                scan_bar = st.progress(0, text="Simulating across concentration gradients...")
                for i, c_val in enumerate(conc_grid):
                    temp_params = params.copy()
                    temp_params['A0'] = c_val
                    
                    sim_out = run_simulation(vol_grid, init_biomass_grid, (vol_grid.min(), vol_grid.max()), temp_params)
                    
                    if sim_out:
                        final_biomass_run = sim_out[2]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            fc = np.log2(final_biomass_run / init_biomass_grid)
                        fc = np.nan_to_num(fc, nan=-6.0, posinf=6.0, neginf=-6.0)
                        heatmap_matrix[i, :] = fc
                        
                    scan_bar.progress((i + 1) / n_concs)
                scan_bar.empty()
                
                st.session_state.heatmap_data = {
                    "conc_grid": conc_grid,
                    "vol_centers": vol_centers,
                    "matrix": heatmap_matrix
                }
                st.session_state.heatmap_params = params.copy()

    # 5. Populate Metrics Container (With Fresh Data)
    if st.session_state.sim_results is not None:
        data = st.session_state.sim_results
        n_trimmed = data["n_trimmed"]
        N_occupied = data["N_occupied"]
        pct = (N_occupied / n_trimmed * 100) if n_trimmed > 0 else 0.0
        
        with metrics_container:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Droplets", f"{n_trimmed:,}")
            c2.metric("Occupied Droplets", f"{N_occupied:,}", f"{pct:.2f}%")
            c3.metric("Antibiotic Conc", f"{data['params']['A0']} μg/ml")
            c4.metric("Sim Duration", f"{data['params']['t_end']} h")
            st.markdown("---")

    # 6. Render Output Tabs
    data = st.session_state.sim_results
    tab_viz, tab_hist = st.tabs(["📊 Visualization", "📜 Run History"])

    with tab_viz:
        # Improved Compare Segment UI
        with st.expander("🛠️ Plot & Comparison Settings", expanded=True):
            col_main, col_tools = st.columns([3, 1])
            
            with col_main:
                selected_plot = st.selectbox("Select Figure:", PLOT_LIST, key="viz_plot_selection")
                st.caption(PLOT_OPTIONS[selected_plot])
            
            with col_tools:
                st.markdown("**Comparison Tools**")
                has_baseline = st.session_state.baseline_results is not None
                if has_baseline:
                    st.success("❄️ Baseline Active")
                else:
                    st.info("⚪ No Baseline")
                
                c_f, c_c = st.columns(2)
                c_f.button("❄️ Freeze", on_click=on_freeze_baseline, use_container_width=True, help="Snapshot current run")
                c_c.button("🧹 Clear", on_click=on_clear_baseline, use_container_width=True, help="Remove snapshot")

        if data is None or data["N_occupied"] == 0:
            st.info("👋 Welcome! Please configure settings in the sidebar and click **'Run Simulation'** to start.")
        else:
            (bin_sums, bin_counts, final_biomass, t_eval, bin_edges,
             a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums, s_bin_sums, alive_bin_sums) = data["sim_output"]

            with st.container():
                p = None
                
                if selected_plot == "Growth/Death Heatmap": 
                    if st.session_state.get("heatmap_data"):
                        hd = st.session_state.heatmap_data
                        p = plot_heatmap(hd["conc_grid"], hd["vol_centers"], hd["matrix"])
                    else:
                        st.warning("Heatmap data unavailable. Please Run Simulation.")

                elif selected_plot == "Survival Probability":
                    p = plot_survival_probability(t_eval, alive_bin_sums, bin_counts, bin_edges)
                elif selected_plot == "MIC vs Volume (Inoculum Effect)":
                    if st.session_state.get("heatmap_data"):
                        p = plot_mic_vs_volume(st.session_state.heatmap_data, data["params"])
                    else:
                        st.warning("Heatmap data unavailable. Please Run Simulation.")
                elif selected_plot == "Population Dynamics":
                    p = plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges, st.session_state.baseline_results)
                elif selected_plot == "Droplet Distribution":
                    p = plot_distribution(data["total_vols"], data["vols"])
                elif selected_plot == "Initial Density & Vc":
                    p = plot_initial_density_vc(data["df_density"], data["vc_val"], data["params"]['concentration'])
                elif selected_plot == "Fold Change":
                    p, df_fc = plot_fold_change(data["vols"], data["initial_biomass"], final_biomass, data["vc_val"])
                    st.download_button("📥 Download CSV", data=convert_df(df_fc), file_name="fold_change_data.csv", mime="text/csv")
                elif selected_plot == "N0 vs Volume":
                    p = plot_n0_vs_volume(data["df_density"], data["vc_val"])
                elif selected_plot == "Net Growth Rate (μ - λ)":
                    p = plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges)
                elif selected_plot == "Substrate Dynamics":
                    p = plot_substrate_dynamics(t_eval, s_bin_sums, bin_counts, bin_edges)
                elif selected_plot == "Antibiotic Dynamics":
                    p = plot_a_eff_dynamics(t_eval, a_eff_bin_sums, bin_counts, bin_edges, data["params"])
                elif selected_plot == "Density Dynamics":
                    p = plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges)
                elif selected_plot == "Bound Antibiotic":
                    if data["params"]['model'] == "Linear Lysis Rate":
                        st.warning("This model does not simulate binding kinetics.")
                        p = None
                    else:
                        p = plot_abound_dynamics(t_eval, a_bound_bin_sums, bin_counts, bin_edges)

                # --- RENDER ALL PLOTS HERE ---
                if p is not None:
                    streamlit_bokeh(p, use_container_width=True)

    with tab_hist:
        st.subheader("Simulation History")
        st.info("👆 **Select a row** in the table below, then click the **'Rerun Selected'** button.")
        
        if st.session_state.run_history:
            df_hist = pd.DataFrame(st.session_state.run_history)
            if 'Timestamp' in df_hist.columns:
                cols_order = ['Timestamp', 'model']
                other_cols = [c for c in df_hist.columns if c not in cols_order]
                df_hist = df_hist[cols_order + other_cols]

            selection = st.dataframe(
                df_hist,
                key="history_table",
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            c1, c2 = st.columns([1, 4])
            if selection.selection.rows:
                c1.button("🔄 Rerun Selected", type="primary", on_click=on_rerun_click)
            
            if c2.button("Clear History", type="secondary"):
                st.session_state.run_history = []
                st.rerun()
        else:
            st.warning("Run a simulation to populate the history table.")

if __name__ == "__main__":
    main()
