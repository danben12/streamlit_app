import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Span, Div
from bokeh.palettes import linear_palette
import colorcet as cc
from streamlit_bokeh import streamlit_bokeh
import time
from scipy.stats import linregress
from bokeh.layouts import column

# Numba Imports
from numba import jit, prange

# ==========================================
# 1. PAGE CONFIG & STATE MANAGEMENT
# ==========================================

def configure_page():
    st.set_page_config(page_title="Growth - Lysis Model simulation", layout="wide")
    st.title("Growth - Lysis Model simulation (Numba Accelerated)")

def init_session_state():
    if "last_change_time" not in st.session_state:
        st.session_state.last_change_time = time.time()

def input_changed():
    st.session_state.last_change_time = time.time()

def check_debounce(delay=0.7):
    # Simple debounce to prevent run on every keystroke
    while time.time() - st.session_state.last_change_time < delay:
        time.sleep(0.1)

# ==========================================
# 2. NUMBA OPTIMIZED ODE SOLVERS
# ==========================================

# --- A. Derivative Functions (JIT Compiled) ---

@jit(nopython=True)
def deriv_effective_conc(state, t, vol, mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n):
    # state: [A_free, A_bound, B_live, B_dead, S]
    A_free, A_bound, B_live, _, S = state
    
    density = B_live / vol
    # Protect against div by zero
    A_eff = A_bound / max(density, 1e-12) 
    
    mu = mu_max * S / (Ks + S)
    
    A_eff_n = A_eff ** n
    K_D_n = K_D ** n
    hill_term = A_eff_n / (K_D_n + A_eff_n + 1e-12)
    
    lambda_D = lambda_max * hill_term
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    
    return np.array([dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt])

@jit(nopython=True)
def deriv_linear_lysis(state, t, vol, A0, mu_max, Ks, Y, a, b, K_A0, n):
    # state: [B_live, B_dead, S]
    B_live, _, S = state
    
    density = B_live / vol
    mu = mu_max * S / (Ks + S)
    
    A0_n = A0 ** n
    K_A0_n = K_A0 ** n
    term_A0 = A0_n / (K_A0_n + A0_n + 1e-12)
    
    lambda_D = a * term_A0 * mu + b
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    return np.array([dB_live_dt, dB_dead_dt, dS_dt])

@jit(nopython=True)
def deriv_combined(state, t, vol, A0, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b):
    # state: [A_free, A_bound, B_live, B_dead, S]
    A_free, A_bound, B_live, _, S = state
    
    density = B_live / vol
    A_eff = A_bound / max(density, 1e-12)
    
    mu = mu_max * S / (Ks + S)
    
    A_eff_n = A_eff ** n
    K_D_n = K_D ** n
    hill_term = A_eff_n / (K_D_n + A_eff_n + 1e-12)
    
    lambda_D = a * hill_term * mu + b
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    
    return np.array([dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt])

# --- B. Parallel RK4 Solver (The Engine) ---

@jit(nopython=True, parallel=True)
def solve_ode_batch_numba(y0_array, vols, t_eval, model_type, 
                          params_array):
    """
    model_type: 1=Effective, 2=Linear, 3=Combined
    params_array: Array of floats matching the specific model's signature
    """
    n_droplets = y0_array.shape[0]
    n_vars = y0_array.shape[1]
    n_steps = len(t_eval)
    
    # Pre-allocate output: (Time, Droplets, Variables)
    result = np.zeros((n_steps, n_droplets, n_vars))
    
    # Initialize t=0
    for i in range(n_droplets):
        result[0, i, :] = y0_array[i, :]

    # Unpack Params based on model_type to keep inside the loop clean
    # Note: Numba requires consistency, so we unpack inside the loop or pass args individually.
    # We will simply pass the array and index inside.
    
    # PARALLEL LOOP OVER DROPLETS
    for i in prange(n_droplets):
        vol = vols[i]
        current_y = y0_array[i, :].copy()
        
        for t_idx in range(n_steps - 1):
            t = t_eval[t_idx]
            dt = t_eval[t_idx+1] - t
            
            # RK4 Integration Step
            if model_type == 1: # Effective
                k1 = deriv_effective_conc(current_y, t, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k2 = deriv_effective_conc(current_y + 0.5*dt*k1, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k3 = deriv_effective_conc(current_y + 0.5*dt*k2, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k4 = deriv_effective_conc(current_y + dt*k3, t + dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
            
            elif model_type == 2: # Linear
                # params: A0, mu_max, Ks, Y, a, b, K_A0, n
                k1 = deriv_linear_lysis(current_y, t, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k2 = deriv_linear_lysis(current_y + 0.5*dt*k1, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k3 = deriv_linear_lysis(current_y + 0.5*dt*k2, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                k4 = deriv_linear_lysis(current_y + dt*k3, t + dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7])
                
            else: # Combined
                # params: A0, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b
                k1 = deriv_combined(current_y, t, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7], params_array[8], params_array[9])
                k2 = deriv_combined(current_y + 0.5*dt*k1, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7], params_array[8], params_array[9])
                k3 = deriv_combined(current_y + 0.5*dt*k2, t + 0.5*dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7], params_array[8], params_array[9])
                k4 = deriv_combined(current_y + dt*k3, t + dt, vol, params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5], params_array[6], params_array[7], params_array[8], params_array[9])

            current_y = current_y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Clamp negatives (optional stability)
            for k in range(n_vars):
                if current_y[k] < 0: current_y[k] = 0.0
                
            result[t_idx+1, i, :] = current_y

    return result

# ==========================================
# 3. UI COMPONENTS (SIDEBAR)
# ==========================================

def render_sidebar():
    st.sidebar.header("Simulation Settings")
    params = {}
    
    # --- Model Selection ---
    model_choices = ["Effective Concentration", "Linear Lysis Rate", "Combined Model"]
    params['model'] = st.sidebar.selectbox("Select Model", model_choices, on_change=input_changed)

    # --- Time Settings ---
    st.sidebar.subheader("Time Settings")
    col_t1, col_t2, col_t3 = st.sidebar.columns(3)
    with col_t1:
        params['t_start'] = st.number_input("Start (h)", value=0.0, step=1.0, on_change=input_changed)
    with col_t2:
        params['t_end'] = st.number_input("End (h)", value=24.0, step=1.0, on_change=input_changed)
    with col_t3:
        params['dt'] = st.number_input("Step (h)", value=0.1, min_value=0.001, step=0.05, format="%.3f", on_change=input_changed)

    # --- Population Gen ---
    st.sidebar.subheader("Population Generation")
    params['mean_log10'] = st.sidebar.number_input("Mean Log10 Volume", 1.0, 8.0, 3.0, 0.1, on_change=input_changed)
    params['std_log10'] = st.sidebar.number_input("Std Dev Log10", 0.1, 3.0, 1.2, 0.1, on_change=input_changed)
    params['n_samples'] = st.sidebar.number_input("N Samples (Droplets)", 1000, 200000, 10000, 1000, on_change=input_changed)
    params['conc_exp'] = st.sidebar.slider("Concentration Exp (10^x)", -7.0, -1.0, -4.3, 0.1, on_change=input_changed)
    params['concentration'] = 10 ** params['conc_exp']
    
    # --- Global Params ---
    st.sidebar.subheader("Global Parameters")
    tab1, tab2 = st.sidebar.tabs(["Growth", "Drugs/Lysis"])

    with tab1:
        params['mu_max'] = st.number_input("mu_max", value=0.7, on_change=input_changed)
        params['Y'] = st.number_input("Yield (Y)", value=0.001, format="%.4f", on_change=input_changed)
        params['S0'] = st.number_input("Initial S0", value=1.0, on_change=input_changed)
        params['Ks'] = st.number_input("Ks", value=2.0, on_change=input_changed)

    with tab2:
        params['A0'] = st.number_input("Initial Antibiotic (A0)", value=10.0, on_change=input_changed)
        
        # Initialize all to 0.0 to safely pass to array
        for key in ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']:
            params[key] = 0.0

        if params['model'] in ["Effective Concentration", "Combined Model"]:
            params['K_on'] = st.number_input("K_on", value=750.0, on_change=input_changed)
            params['K_off'] = st.number_input("K_off", value=0.01, on_change=input_changed)
            params['K_D'] = st.number_input("K_D", value=12000.0, on_change=input_changed)
            params['n_hill'] = st.number_input("Hill coeff (n)", value=20.0, on_change=input_changed)

        if params['model'] == "Effective Concentration":
            params['lambda_max'] = st.number_input("lambda_max", value=1.0, on_change=input_changed)

        if params['model'] in ["Linear Lysis Rate", "Combined Model"]:
            params['a'] = st.number_input("a (Growth Lysis)", value=3.0, on_change=input_changed)
            params['b'] = st.number_input("b (Base Lysis)", value=0.1, on_change=input_changed)

        if params['model'] == "Linear Lysis Rate":
            params['K_A0'] = st.number_input("K_A0", value=10.0, on_change=input_changed)
            params['n_hill'] = st.number_input("Hill coeff (n)", value=20.0, on_change=input_changed)
            
    return params


# ==========================================
# 4. CORE LOGIC & CALCULATION FUNCTIONS
# ==========================================

@st.cache_data(show_spinner=False)
def generate_population(mean, std, n, conc, mean_pix, std_pix):
    """
    Generates population with integer pixel rounding (camera simulation).
    Cached by Streamlit to avoid regen on plot switch.
    """
    # 1. Droplet Volumes
    log_data = np.random.normal(loc=mean, scale=std, size=int(n))
    volume_data = 10 ** log_data
    
    mask_vol = (volume_data >= 1000) & (volume_data <= 1e8)
    trimmed_vol = volume_data[mask_vol]
    
    # 2. Poisson loading (Cell Counts)
    lambdas = trimmed_vol * conc
    cell_counts = np.random.poisson(lam=lambdas)
    
    # Filter occupied
    occupied_mask = cell_counts > 0
    final_vols = trimmed_vol[occupied_mask].copy()
    final_counts = cell_counts[occupied_mask].copy()
    
    n_occupied = len(final_counts)
    
    # 3. Biomass Conversion with Noise
    base_biomass = final_counts * mean_pix
    
    # Add noise 
    noise_scale = np.sqrt(final_counts) * std_pix
    noise = np.random.normal(0, 1, n_occupied) * noise_scale
    
    raw_biomass = base_biomass + noise
    
    # --- DISCRETIZATION STEP ---
    final_biomass = np.round(raw_biomass)
    final_biomass = np.maximum(final_biomass, 1.0) 
    
    return final_vols, final_counts, final_biomass, trimmed_vol


def calculate_vc_and_density(vols, biomass, theoretical_conc, mean_pix):
    # Effective count for density comparison
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
    
    closest_index = differences.idxmin()
    
    for i in range(len(differences) - convergence_window):
        window_mean_diff = differences.iloc[i:i + convergence_window].mean()
        if window_mean_diff <= tolerance:
            closest_index = i + convergence_window // 2
            break
            
    vc_val = df.loc[closest_index, 'Volume']
    return df, vc_val

# --- POST-PROCESSING HELPERS ---
def calculate_derived_metrics(sol_reshaped, vols, params):
    """
    Calculates Lambda, A_eff, Net Rate from the solution trajectories.
    Vectorized NumPy operations (fast enough without Numba).
    """
    model = params['model']
    # sol_reshaped is (Time, Droplets, Vars)
    
    if model == "Linear Lysis Rate":
        B_live = sol_reshaped[:, :, 0]
        S = sol_reshaped[:, :, 2]
        A_bound = np.zeros_like(B_live) # No binding in this model
    else: 
        A_bound = sol_reshaped[:, :, 1]
        B_live = sol_reshaped[:, :, 2]
        S = sol_reshaped[:, :, 4]
        
    density = B_live / vols # Broadcasting vols
    mu_mat = params['mu_max'] * S / (params['Ks'] + S)
    
    lambda_matrix = np.zeros_like(mu_mat)
    A_eff = np.zeros_like(mu_mat)

    if model == "Effective Concentration":
        A_eff = A_bound / np.maximum(density, 1e-12)
        A_eff_n = np.power(A_eff, params['n_hill'])
        K_D_n = params['K_D'] ** params['n_hill']
        hill = A_eff_n / (K_D_n + A_eff_n + 1e-12)
        lambda_matrix = params['lambda_max'] * hill

    elif model == "Linear Lysis Rate":
        A_eff[:] = params['A0'] # Logic: External conc is effective conc
        A0_n = np.power(params['A0'], params['n_hill'])
        K_A0_n = params['K_A0'] ** params['n_hill']
        term_A0 = A0_n / (K_A0_n + A0_n + 1e-12)
        lambda_matrix = params['a'] * term_A0 * mu_mat + params['b']

    elif model == "Combined Model":
        A_eff = A_bound / np.maximum(density, 1e-12)
        A_eff_n = np.power(A_eff, params['n_hill'])
        K_D_n = params['K_D'] ** params['n_hill']
        hill = A_eff_n / (K_D_n + A_eff_n + 1e-12)
        lambda_matrix = params['a'] * hill * mu_mat + params['b']
        
    net_rate = mu_mat - lambda_matrix
    return lambda_matrix, A_eff, density, A_bound, net_rate, B_live


@st.cache_data(show_spinner=False)
def run_simulation_cached(vols, initial_biomass, total_vols_range, params):
    """
    Main simulation driver, now Cached. 
    Uses Numba RK4 instead of Scipy odeint.
    """
    t_eval = np.arange(params['t_start'], params['t_end'] + params['dt']/100.0, params['dt'])
    if len(t_eval) < 2: t_eval = np.linspace(params['t_start'], params['t_end'], 2)
    N_STEPS = len(t_eval)
    N_occupied = len(vols)
    
    # --- Prepare Binning ---
    min_exp = int(np.floor(np.log10(total_vols_range[0])))
    max_exp = int(np.ceil(np.log10(total_vols_range[1])))
    bin_edges_log = np.arange(min_exp, max_exp + 1)
    bin_edges = 10 ** bin_edges_log
    n_bins = len(bin_edges) - 1
    
    # --- Prepare Numba Inputs ---
    model_name = params['model']
    
    if model_name == "Effective Concentration":
        model_type = 1
        n_vars = 5
        # Params: mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n
        p_arr = np.array([params['mu_max'], params['Ks'], params['Y'], params['K_on'], 
                          params['K_off'], params['lambda_max'], params['K_D'], params['n_hill']])
        
        y0 = np.zeros((N_occupied, 5))
        y0[:, 0] = params['A0'] # A_free
        y0[:, 2] = initial_biomass
        y0[:, 4] = params['S0']
        
    elif model_name == "Linear Lysis Rate":
        model_type = 2
        n_vars = 3
        # Params: A0, mu_max, Ks, Y, a, b, K_A0, n
        p_arr = np.array([params['A0'], params['mu_max'], params['Ks'], params['Y'], 
                          params['a'], params['b'], params['K_A0'], params['n_hill']])
        
        y0 = np.zeros((N_occupied, 3))
        y0[:, 0] = initial_biomass
        y0[:, 2] = params['S0']
        
    else: # Combined
        model_type = 3
        n_vars = 5
        # Params: A0, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b
        p_arr = np.array([params['A0'], params['mu_max'], params['Ks'], params['Y'], 
                          params['K_on'], params['K_off'], params['K_D'], params['n_hill'], 
                          params['a'], params['b']])
        
        y0 = np.zeros((N_occupied, 5))
        y0[:, 0] = params['A0']
        y0[:, 2] = initial_biomass
        y0[:, 4] = params['S0']

    # --- RUN NUMBA SOLVER ---
    # This replaces the loop over batches and odeint
    # It runs in parallel on all available cores
    sol_reshaped = solve_ode_batch_numba(y0, vols, t_eval, model_type, p_arr)

    # --- Calculate Derived Metrics (Vectorized) ---
    lambda_mat, A_eff_mat, density_mat, A_bound_mat, net_rate_mat, B_live_cont = calculate_derived_metrics(sol_reshaped, vols, params)

    # --- Discretization (Camera Sim) ---
    B_live_disc = np.round(B_live_cont)
    final_counts_all = np.mean(B_live_disc[-2:, :], axis=0) if N_STEPS > 2 else B_live_disc[-1, :]
    final_counts_all = np.where(final_counts_all < 2.0, 0.0, final_counts_all)

    # --- Binning Aggregation ---
    bin_sums = np.zeros((n_bins, N_STEPS))
    a_eff_bin_sums = np.zeros((n_bins, N_STEPS))
    density_bin_sums = np.zeros((n_bins, N_STEPS))
    a_bound_bin_sums = np.zeros((n_bins, N_STEPS))
    net_rate_bin_sums = np.zeros((n_bins, N_STEPS))
    bin_counts = np.zeros(n_bins)
    
    # Transpose for easier binning logic (Droplets, Time) -> (Time, Droplets) in original plot logic
    # The plotting functions expect (Bins, Time). My matrices are (Time, Droplets).
    
    for b_idx in range(n_bins):
        low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
        mask = (vols >= low) & (vols < high)
        count_in_bin = np.sum(mask)
        bin_counts[b_idx] = count_in_bin
        
        if count_in_bin > 0:
            # Sum over droplets in this bin (axis 1 of transpose)
            bin_sums[b_idx, :] = np.sum(B_live_disc[:, mask], axis=1)
            a_eff_bin_sums[b_idx, :] = np.sum(A_eff_mat[:, mask], axis=1)
            density_bin_sums[b_idx, :] = np.sum(density_mat[:, mask], axis=1)
            a_bound_bin_sums[b_idx, :] = np.sum(A_bound_mat[:, mask], axis=1)
            net_rate_bin_sums[b_idx, :] = np.sum(net_rate_mat[:, mask], axis=1)

    return (bin_sums, bin_counts, final_counts_all, t_eval, bin_edges, 
            a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums)

# ==========================================
# 5. PLOTTING FUNCTIONS 
# ==========================================
# (Kept mostly identical to preserve your styling)

def int_to_superscript(n):
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))

def plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Normalized Biomass (B/B₀)", 
               height=600, width=1000, tools="pan,wheel_zoom,reset,save")
    
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []

    legend_items = []
    color_idx = 0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_traj = bin_sums[i, :] / bin_counts[i]
            initial_val = mean_traj[0]
            norm_traj = mean_traj / initial_val if initial_val > 1e-9 else mean_traj
            norm_traj = np.where(norm_traj <= 0, np.nan, norm_traj)
            
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)} (n={int(bin_counts[i])})"
            
            r = p.line(t_eval, norm_traj, line_color=colors[color_idx], line_width=3, alpha=0.9)
            legend_items.append((label, [r]))
            color_idx += 1

    total_biomass_traj = np.sum(bin_sums, axis=0)
    total_N0 = total_biomass_traj[0]
    meta_norm = total_biomass_traj / total_N0 if total_N0 > 1e-9 else total_biomass_traj

    r_meta = p.line(t_eval, meta_norm, line_color="white", line_width=4, line_dash="dashed", alpha=1.0)
    legend_items.insert(0, ("Metapopulation (Avg)", [r_meta]))
            
    legend = Legend(items=legend_items, title="Volume Bins", click_policy="hide")
    p.add_layout(legend, 'right')
    return p

def plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Net Growth Rate (μ - λ) [1/h]", 
               height=600, width=1000, tools="pan,wheel_zoom,reset,save",
               title="Net Growth Rate (μ - λ) Dynamics")
    
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []

    color_idx = 0
    legend_items = []
    p.add_layout(Span(location=0, dimension='width', line_color='white', line_dash='dotted', line_width=2))

    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_net_rate = net_rate_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i+1]))
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
               height=600, width=1000, tools="pan,wheel_zoom,reset,save", title=title_text)
    
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []

    color_idx = 0
    legend_items = []

    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_a_eff = a_eff_bin_sums[i, :] / bin_counts[i]
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_a_eff, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1

    threshold_val = params['K_A0'] if params['model'] == "Linear Lysis Rate" else params['K_D']
    label_text = "K_A0" if params['model'] == "Linear Lysis Rate" else "K_D"

    p.add_layout(Span(location=threshold_val, dimension='width', line_color='red', line_dash='dotted', line_width=3))
    r_thresh_dummy = p.line([], [], line_color='red', line_dash='dotted', line_width=3)
    legend_items.insert(0, (f"{label_text} ({threshold_val:.0f})", [r_thresh_dummy]))

    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Cell Density (Biomass/Volume)", 
               height=600, width=1000, tools="pan,wheel_zoom,reset,save",
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
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            r = p.line(t_eval, mean_vals, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1

    legend = Legend(items=legend_items, location="top_right", click_policy="hide", title="Volume Bins")
    p.add_layout(legend, 'right')
    return p

def plot_abound_dynamics(t_eval, abound_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Bound Antibiotic (Molecules/Droplet)", 
               height=600, width=1000, tools="pan,wheel_zoom,reset,save",
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
            high_exp = int(np.log10(bin_edges[i+1]))
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
               x_axis_type="log", height=600, width=1000, tools="pan,wheel_zoom,reset,save")
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
               width=1000, height=600, output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    
    r_dens = p.scatter('Volume', 'InitialDensity', source=source, color='silver', alpha=0.6, size=4)
    r_roll = p.line(df_density['Volume'], df_density['RollingMeanDensity'], color='red', line_width=3)
    min_v, max_v = df_density['Volume'].min(), df_density['Volume'].max()
    r_theo = p.line([min_v, max_v], [theoretical_density, theoretical_density], color='white', line_width=3)
    p.add_layout(Span(location=vc_val, dimension='height', line_color='white', line_dash='dashed', line_width=3))
    
    legend = Legend(items=[
        LegendItem(label='Initial Density', renderers=[r_dens]),
        LegendItem(label='Rolling Mean', renderers=[r_roll]),
        LegendItem(label='Inoculum Density', renderers=[r_theo]),
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
    if not df_sub.empty:
        df_sub['MovingAverage'] = df_sub['FoldChange'].rolling(window=100, min_periods=1).mean()
    else:
        df_sub['MovingAverage'] = np.nan

    total_initial = np.sum(initial_biomass)
    total_final = np.sum(final_biomass)
    meta_fc = np.log2(total_final / total_initial) if total_final > 0 else min_fc
    
    source = ColumnDataSource(df_fc)
    sub_source = ColumnDataSource(df_sub)
    
    p = figure(x_axis_type='log', y_axis_type='linear', 
               x_axis_label='Volume (μm³)', y_axis_label='Log2 biomass Fold Change',
               width=1000, height=600, y_range=(-7, 9), output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    
    r_scat = p.scatter('Volume', 'FoldChange', source=source, color='silver', alpha=0.6, size=4)
    r_ma = p.line('Volume', 'MovingAverage', source=sub_source, color='red', line_width=3) if not df_sub.empty else p.line([], [], color='red')
    
    min_v, max_v = df_fc['Volume'].min(), df_fc['Volume'].max()
    r_meta = p.line([min_v, max_v], [meta_fc, meta_fc], color='white', line_width=3)
    p.line([min_v, max_v], [0, 0], color='white', line_dash='dashdot', line_width=3)
    p.line([vc_val, vc_val], [-10, 10], color='white', line_dash='dashed', line_width=3)
    
    p.add_tools(HoverTool(tooltips=[('Volume', '@Volume{0,0}'), ('Fold Change', '@FoldChange{0.00}'), ('ID', '@DropletID')], renderers=[r_scat]))
    
    legend = Legend(items=[
        LegendItem(label='Droplet FC', renderers=[r_scat]),
        LegendItem(label='FC Moving Avg', renderers=[r_ma]),
        LegendItem(label='Metapopulation FC', renderers=[r_meta]),
    ], location='top_right')
    p.add_layout(legend, 'right')
    return p, df_fc

def plot_n0_vs_volume(df, Vc):
    plot_df = df.copy()
    plot_df['DropletID'] = plot_df.index
    source = ColumnDataSource(plot_df)
    
    p = figure(x_axis_type='log', y_axis_type='log',
               x_axis_label='Volume (μm³)', y_axis_label='Initial Biomass', 
               output_backend="webgl", width=1000, height=600, 
               tools="pan,wheel_zoom,reset,save")

    r_scat = p.scatter('Volume', 'Biomass', source=source, color='gray', alpha=0.6, size=5)
    p.add_tools(HoverTool(tooltips=[('Volume', '@Volume{0,0}'), ('Biomass', '@Biomass{0.00}')], renderers=[r_scat]))

    filtered_df = plot_df[(plot_df['Biomass'] > 0) & (plot_df['Volume'] >= Vc)]
    stats_text = "Insufficient data for regression"
    
    if len(filtered_df) > 2:
        x = np.log10(filtered_df['Volume'])
        y = np.log10(filtered_df['Biomass'])
        slope, intercept, r_value, _, _ = linregress(x, y)
        x_values = np.linspace(plot_df['Volume'].min(), plot_df['Volume'].max(), 100)
        y_values = 10 ** (intercept + slope * np.log10(x_values))
        p.line(x_values, y_values, color='red', line_width=3)
        stats_text = f'<b>Regression (V > Vc):</b><br>y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_value ** 2:.3f}'

    stats_div = Div(text=stats_text, width=400, height=80, styles={'text-align': 'center', 'background-color': '#f0f2f6', 'padding': '10px', 'border-radius': '5px'})
    return column(p, stats_div)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    configure_page()
    init_session_state()
    
    MEAN_PIXELS = 5.5
    STD_PIXELS = 1.0
    
    params = render_sidebar()
    
    check_debounce()
    
    # --- 1. CACHED POPULATION GEN ---
    start_gen = time.time()
    vols, counts, initial_biomass, total_vols = generate_population(
        params['mean_log10'], params['std_log10'], params['n_samples'], 
        params['concentration'], MEAN_PIXELS, STD_PIXELS
    )
    
    n_trimmed = len(total_vols)
    N_occupied = len(vols)
    pct = (N_occupied / n_trimmed * 100) if n_trimmed > 0 else 0.0
    
    # Header Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Droplets", f"{n_trimmed:,}")
    col2.metric("Occupied Droplets", f"{N_occupied:,} ({pct:.2f}%)")
    col3.metric("Sim Duration", "Calculating...")

    if N_occupied == 0:
        st.error("No occupied droplets found. Increase Concentration or Mean Volume.")
        st.stop()
        
    df_density, vc_val = calculate_vc_and_density(vols, initial_biomass, params['concentration'], MEAN_PIXELS)
    
    # --- 2. CACHED SIMULATION (NUMBA) ---
    sim_start = time.time()
    status_text = st.empty()
    status_text.text(f"Simulating {N_occupied} droplets using Numba (Parallel)...")
    
    (bin_sums, bin_counts, final_biomass, t_eval, bin_edges, 
     a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums) = run_simulation_cached(
        vols, initial_biomass, (total_vols.min(), total_vols.max()), params
    )
    
    sim_dur = time.time() - sim_start
    status_text.empty()
    col3.metric("Sim Duration", f"{sim_dur:.3f} s")
    
    st.divider()
    
    # --- 3. UI PLOTS ---
    st.subheader("Results Analysis")
    plot_options = [
        "Population Dynamics", "Droplet Distribution", "Initial Density & Vc", 
        "Fold Change", "N0 vs Volume", "Net Growth Rate (μ - λ)", 
        "Antibiotic Dynamics", "Density Dynamics", "Bound Antibiotic"
    ]
    selected_plot = st.selectbox("Select Figure:", plot_options)

    with st.container():
        if selected_plot == "Population Dynamics":
            p = plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p, use_container_width=True)

        elif selected_plot == "Droplet Distribution":
            p = plot_distribution(total_vols, vols)
            streamlit_bokeh(p, use_container_width=True)

        elif selected_plot == "Initial Density & Vc":
            p = plot_initial_density_vc(df_density, vc_val, params['concentration'])
            streamlit_bokeh(p, use_container_width=True)

        elif selected_plot == "Fold Change":
            p, df_fc = plot_fold_change(vols, initial_biomass, final_biomass, vc_val)
            streamlit_bokeh(p, use_container_width=True)

        elif selected_plot == "N0 vs Volume":
            p = plot_n0_vs_volume(df_density, vc_val)
            streamlit_bokeh(p, use_container_width=True)

        elif selected_plot == "Net Growth Rate (μ - λ)":
            p_net = plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p_net, use_container_width=True)

        elif selected_plot == "Antibiotic Dynamics":
            p_aeff = plot_a_eff_dynamics(t_eval, a_eff_bin_sums, bin_counts, bin_edges, params)
            streamlit_bokeh(p_aeff, use_container_width=True)

        elif selected_plot == "Density Dynamics":
            p_dens = plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p_dens, use_container_width=True)

        elif selected_plot == "Bound Antibiotic":
            if params['model'] == "Linear Lysis Rate":
                st.warning("Model does not support binding kinetics.")
            else:
                p_abound = plot_abound_dynamics(t_eval, a_bound_bin_sums, bin_counts, bin_edges)
                streamlit_bokeh(p_abound, use_container_width=True)

if __name__ == "__main__":
    main()
