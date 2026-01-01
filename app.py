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
PLOT_OPTIONS = [
    "Population Dynamics",
    "Droplet Distribution",
    "Initial Density & Vc",
    "Fold Change",
    "N0 vs Volume",
    "Net Growth Rate (μ - λ)",
    "Substrate Dynamics",
    "Antibiotic Dynamics",
    "Density Dynamics",
    "Bound Antibiotic",
    "Growth/Death Heatmap"
]

# ==========================================
# 1. PAGE CONFIG
# ==========================================

def configure_page():
    st.set_page_config(page_title="Growth - Lysis Model simulation", layout="wide")
    st.title("Growth - Lysis Model simulation")

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
    """Updates session state keys with values from the selected history row."""
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
        'a': 'a', 'b': 'b', 'K_A0': 'K_A0'
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
    """Callback for Rerun button"""
    if "history_table" in st.session_state and st.session_state["history_table"]["selection"]["rows"]:
        idx = st.session_state["history_table"]["selection"]["rows"][0]
        if idx < len(st.session_state.run_history):
            selected_row = st.session_state.run_history[idx]
            load_params_from_history(selected_row)
            st.session_state.trigger_run = True

# ==========================================
# 4. UI COMPONENTS
# ==========================================

def render_sidebar():
    st.sidebar.header("Simulation Settings")
    params = {}

    params['model'] = st.sidebar.selectbox(
        "Select Model", 
        ["Effective Concentration", "Linear Lysis Rate", "Combined Model"], 
        key='model_select'
    )

    st.sidebar.subheader("Time Settings")
    col1, col2, col3 = st.sidebar.columns(3)
    
    params['t_start'] = col1.number_input("Start (h)", value=st.session_state.get('t_start', 0.0), step=1.0, key='t_start')
    params['t_end'] = col2.number_input("End (h)", value=st.session_state.get('t_end', 24.0), step=1.0, key='t_end')
    params['dt'] = col3.number_input("Step (h)", value=st.session_state.get('dt', 1.0), min_value=0.01, step=0.5, key='dt')

    st.sidebar.subheader("Population Generation")
    params['mean_log10'] = st.sidebar.number_input("Mean Log10 Volume", 1.0, 8.0, st.session_state.get('mean_log10', 3.0), 0.1, key='mean_log10')
    params['std_log10'] = st.sidebar.number_input("Std Dev Log10", 0.1, 3.0, st.session_state.get('std_log10', 1.2), 0.1, key='std_log10')
    params['n_samples'] = st.sidebar.number_input("N Samples (Droplets)", 1000, 100000, st.session_state.get('n_samples', 17000), 1000, key='n_samples')
    
    params['conc_exp'] = st.sidebar.slider("Concentration Exp (10^x)", -7.0, -1.0, st.session_state.get('conc_exp', -4.3), 0.1, key='conc_exp')
    params['concentration'] = 10 ** params['conc_exp']

    st.sidebar.subheader("Global Parameters")
    tab1, tab2 = st.sidebar.tabs(["Growth", "Drugs/Lysis"])

    with tab1:
        params['mu_max'] = st.number_input("mu_max", value=st.session_state.get('mu_max', 0.7), key='mu_max')
        params['Y'] = st.number_input("Yield (Y)", value=st.session_state.get('Y', 0.001), format="%.4f", key='Y')
        params['S0'] = st.number_input("Initial S0", value=st.session_state.get('S0', 1.0), key='S0')
        params['Ks'] = st.number_input("Ks", value=st.session_state.get('Ks', 2.0), key='Ks')

    with tab2:
        params['A0'] = st.number_input("Initial Antibiotic (A0)", value=st.session_state.get('A0', 10.0), key='A0')
        
        defaults = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']
        for key in defaults: params[key] = 0.0

        if params['model'] in ["Effective Concentration", "Combined Model"]:
            params['K_on'] = st.number_input("K_on", value=st.session_state.get('K_on', 750.0), key='K_on')
            params['K_off'] = st.number_input("K_off", value=st.session_state.get('K_off', 0.01), key='K_off')
            params['K_D'] = st.number_input("K_D", value=st.session_state.get('K_D', 12000.0), key='K_D')
            if 'n_hill' not in st.session_state: st.session_state.n_hill = 20.0
            params['n_hill'] = st.number_input("Hill coeff (n)", value=st.session_state.get('n_hill_1', st.session_state.n_hill), key='n_hill_1')

        if params['model'] == "Effective Concentration":
            params['lambda_max'] = st.number_input("lambda_max", value=st.session_state.get('lambda_max', 1.0), key='lambda_max')

        if params['model'] in ["Linear Lysis Rate", "Combined Model"]:
            params['a'] = st.number_input("a (Growth Lysis)", value=st.session_state.get('a', 3.0), key='a')
            params['b'] = st.number_input("b (Base Lysis)", value=st.session_state.get('b', 0.1), key='b')

        if params['model'] == "Linear Lysis Rate":
            params['K_A0'] = st.number_input("K_A0", value=st.session_state.get('K_A0', 10.0), key='K_A0')
            val_n_hill = st.session_state.get('n_hill_2', 20.0)
            params['n_hill'] = st.number_input("Hill coeff (n)", value=val_n_hill, key='n_hill_2')

        if 'n_hill_1' in params: params['n_hill'] = params['n_hill_1']
        elif 'n_hill_2' in params: params['n_hill'] = params['n_hill_2']

    return params

# ==========================================
# 5. CORE LOGIC
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
                                  bin_counts, bin_edges, vols,
                                  batch_blive_T, batch_a_eff_T, batch_density_T,
                                  batch_abound_T, batch_net_rate, batch_S_T):
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

    final_c = np.mean(batch_blive[-2:, :], axis=0) if N_STEPS > 2 else batch_blive[-1, :]
    final_c = np.where(final_c < 2.0, 0.0, final_c)

    fast_accumulate_bins_serial(
        local_bin_sums, local_a_eff, local_density,
        local_a_bound, local_net_rate, local_s_sums,
        local_bin_counts, bin_edges, batch_vols,
        batch_blive_T, batch_a_eff_T, batch_density_T,
        batch_abound_T, batch_net_rate, batch_S_T
    )

    return (idx, final_c[0], local_bin_sums, local_bin_counts, 
            local_a_eff, local_density, local_a_bound, local_net_rate, local_s_sums)

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
    total_bin_counts = np.zeros(n_bins)
    final_counts_all = np.zeros(N_occupied)

    # Progress bar only if a large simulation is running
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
                (idx, final_c_val, l_sums, l_counts, l_a_eff, l_dens, l_abound, l_net, l_s) = future.result()
                final_counts_all[idx] = final_c_val
                total_bin_sums += l_sums
                total_bin_counts += l_counts
                total_a_eff += l_a_eff
                total_density += l_dens
                total_a_bound += l_abound
                total_net_rate += l_net
                total_s_sums += l_s
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
            total_a_eff, total_density, total_a_bound, total_net_rate, total_s_sums)

def run_simulation(vols, initial_biomass, total_vols_range, params):
    return _compute_simulation_core(vols, initial_biomass, total_vols_range, params)

# ==========================================
# 6. PLOTTING FUNCTIONS
# ==========================================

def int_to_superscript(n):
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))

def plot_heatmap(conc_grid, vol_centers, data_matrix):
    """
    Plots a 2D Heatmap of Fold Change with adaptive row heights.
    X: Volume (Log), Y: Concentration, Color: Fold Change
    """
    x_list = []
    y_list = []
    c_list = []
    h_list = []
    
    # 1. Calculate boundaries to determine row heights (Adaptive Grid)
    # Midpoints between concentration levels
    mids = (conc_grid[:-1] + conc_grid[1:]) / 2
    # Extend boundaries to cover the first and last points appropriately
    lower_bound = conc_grid[0] - (mids[0] - conc_grid[0]) if len(mids) > 0 else conc_grid[0] - 0.5
    upper_bound = conc_grid[-1] + (conc_grid[-1] - mids[-1]) if len(mids) > 0 else conc_grid[-1] + 0.5
    boundaries = np.concatenate(([lower_bound], mids, [upper_bound]))
    # Heights are the difference between boundaries
    row_heights = np.diff(boundaries)
    
    # Width is uniform in log space
    w = (np.log10(vol_centers.max()) - np.log10(vol_centers.min())) / len(vol_centers)

    for i, conc in enumerate(conc_grid):
        current_h = row_heights[i]
        for j, vol in enumerate(vol_centers):
            val = data_matrix[i, j]
            if not np.isnan(val):
                x_list.append(np.log10(vol))
                y_list.append(conc)
                c_list.append(val)
                h_list.append(current_h)

    source = ColumnDataSource(data={
        'x': x_list, 'y': y_list, 'fc': c_list, 'h': h_list,
        'vol': 10**np.array(x_list)
    })

    mapper = linear_cmap(field_name='fc', palette=cc.CET_D1[::-1], low=-5, high=5)

    p = figure(
        title="Survival Landscape: Volume vs Antibiotic Dose",
        x_axis_label="Volume (µm³)",
        y_axis_label="Antibiotic Concentration (µg/ml)",
        width=1200, height=800,
        # REMOVED crosshair and hover
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above"
    )

    # Use adaptive height 'h' from source
    p.rect(x='x', y='y', width=w*1.02, height='h', source=source,
           fill_color=mapper, line_color=None)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0),
                         title="Log2 FC")
    p.add_layout(color_bar, 'right')
    
    overrides = {}
    for i in range(15): # Cover range 10^0 to 10^15
        overrides[i] = f"10{int_to_superscript(i)}"
    p.xaxis.major_label_overrides = overrides
    
    return p

def plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Normalized Biomass (B/B₀)",
               height=800, width=1200, tools="pan,wheel_zoom,reset,save")
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
                    line_dash="dashed", alpha=1.0)
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
    return p, df_fc

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
    if "run_history" not in st.session_state: st.session_state.run_history = []
    if "trigger_run" not in st.session_state: st.session_state.trigger_run = False
    
    # 1. Render Sidebar
    params = render_sidebar()

    # 2. Display Metrics
    if st.session_state.sim_results is not None:
        data = st.session_state.sim_results
        n_trimmed = data["n_trimmed"]
        N_occupied = data["N_occupied"]
        pct = (N_occupied / n_trimmed * 100) if n_trimmed > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Droplets", f"{n_trimmed:,}")
        col2.metric("Occupied", f"{N_occupied:,} ({pct:.2f}%)")
        col3.metric("Antibiotic Conc", f"{data['params']['A0']}") 
        st.divider()

    # 3. Header and Controls
    st.subheader("Results Analysis")
    col_btn, _ = st.columns([2,6])
    with col_btn:
        manual_run = st.button("Run Simulation", type="primary")
    
    # 4. Logic: Run Simulation
    should_run = manual_run or st.session_state.trigger_run or st.session_state.sim_results is None

    if should_run:
        st.session_state.trigger_run = False 
        st.session_state.heatmap_data = None # Clear old heatmap data if parameters change
        
        # --- A. RUN MAIN SIMULATION ---
        with st.spinner("Running main simulation..."):
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
            with st.spinner("Generating Survival Landscape (Heatmap)..."):
                # 1. Define Synthetic Volume Grid (50 Points)
                min_log = np.log10(total_vols.min())
                max_log = np.log10(total_vols.max())
                vol_grid = np.logspace(min_log, max_log, 50) 
                
                # 2. Define Idealized Biomass (Corrected for Occupied State)
                conc_for_generation = params["concentration"] 
                expected_counts = np.maximum(vol_grid * conc_for_generation, 1.0)
                init_biomass_grid = expected_counts * MEAN_PIXELS
                
                # 3. Define Concentration Grid with MANDATORY POINTS
                # Ensure 0, 3.3, 10, 30 are always included
                mandatory_concs = np.array([0.0, 3.3, 10.0, 30.0])
                base_grid = np.linspace(0, 40.0, 20)
                conc_grid = np.unique(np.concatenate((mandatory_concs, base_grid)))
                conc_grid.sort()
                
                n_concs_total = len(conc_grid)
                heatmap_matrix = np.zeros((n_concs_total, len(vol_grid)))
                
                vol_centers = vol_grid 

                # 4. Scan Loop
                scan_bar = st.progress(0, text="Scanning antibiotic concentrations...")
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
                        
                    scan_bar.progress((i + 1) / n_concs_total)
                scan_bar.empty()
                
                st.session_state.heatmap_data = {
                    "conc_grid": conc_grid,
                    "vol_centers": vol_centers,
                    "matrix": heatmap_matrix
                }
                st.session_state.heatmap_params = params.copy()

    # 5. Render Output Tabs
    data = st.session_state.sim_results
    tab_viz, tab_hist = st.tabs(["📊 Visualization", "📜 Run History"])

    with tab_viz:
        selected_plot = st.selectbox(
            "Select Figure to Display:", 
            PLOT_OPTIONS, 
            key="viz_plot_selection"
        )
        
        if data is None or data["N_occupied"] == 0:
            st.error("No occupied droplets found. Try increasing Concentration or Mean Volume.")
        else:
            (bin_sums, bin_counts, final_biomass, t_eval, bin_edges,
             a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums, s_bin_sums) = data["sim_output"]

            with st.container():
                p = None
                
                if selected_plot == PLOT_OPTIONS[10]: # Heatmap
                    if st.session_state.get("heatmap_data"):
                        hd = st.session_state.heatmap_data
                        p = plot_heatmap(hd["conc_grid"], hd["vol_centers"], hd["matrix"])
                    else:
                        st.warning("Heatmap data unavailable. Please Run Simulation.")

                elif selected_plot == PLOT_OPTIONS[0]: # Population Dynamics
                    p = plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges)
                elif selected_plot == PLOT_OPTIONS[1]: # Droplet Distribution
                    p = plot_distribution(data["total_vols"], data["vols"])
                elif selected_plot == PLOT_OPTIONS[2]: # Initial Density & Vc
                    p = plot_initial_density_vc(data["df_density"], data["vc_val"], data["params"]['concentration'])
                elif selected_plot == PLOT_OPTIONS[3]: # Fold Change
                    p, df_fc = plot_fold_change(data["vols"], data["initial_biomass"], final_biomass, data["vc_val"])
                    st.download_button("Download CSV", data=convert_df(df_fc), file_name="fold_change_data.csv", mime="text/csv")
                elif selected_plot == PLOT_OPTIONS[4]: # N0 vs Volume
                    p = plot_n0_vs_volume(data["df_density"], data["vc_val"])
                elif selected_plot == PLOT_OPTIONS[5]: # Net Growth Rate (μ - λ)
                    p = plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges)
                elif selected_plot == PLOT_OPTIONS[6]: # Substrate Dynamics
                    p = plot_substrate_dynamics(t_eval, s_bin_sums, bin_counts, bin_edges)
                elif selected_plot == PLOT_OPTIONS[7]: # Antibiotic Dynamics
                    p = plot_a_eff_dynamics(t_eval, a_eff_bin_sums, bin_counts, bin_edges, data["params"])
                elif selected_plot == PLOT_OPTIONS[8]: # Density Dynamics
                    p = plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges)
                elif selected_plot == PLOT_OPTIONS[9]: # Bound Antibiotic
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
            
            if selection.selection.rows:
                st.button("🔄 Rerun Selected Configuration", type="primary", on_click=on_rerun_click)
            
            if st.button("Clear History", type="secondary"):
                st.session_state.run_history = []
                st.rerun()
        else:
            st.warning("Run a simulation to populate the history table.")

if __name__ == "__main__":
    main()
