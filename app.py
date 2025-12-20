import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Span
from bokeh.palettes import linear_palette 
import colorcet as cc 
from streamlit_bokeh import streamlit_bokeh
import time
from scipy.stats import linregress
from bokeh.layouts import column
from bokeh.models import Div

# ==========================================
# 1. PAGE CONFIG & STATE MANAGEMENT
# ==========================================

def configure_page():
    st.set_page_config(page_title="Growth - Lysis Model simulation", layout="wide")
    st.title("Growth - Lysis Model simulation")

def init_session_state():
    if "last_change_time" not in st.session_state:
        st.session_state.last_change_time = time.time()

def input_changed():
    st.session_state.last_change_time = time.time()

def check_debounce(delay=0.7):
    while time.time() - st.session_state.last_change_time < delay:
        time.sleep(0.1)

# ==========================================
# 2. ODE MATH MODELS
# ==========================================

def vec_effective_concentration(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n):
    y = y_flat.reshape(N, 5)
    A_free = y[:, 0]
    A_bound = y[:, 1]
    B_live = y[:, 2]
    S = y[:, 4]
    
    density = B_live / V
    A_eff = A_bound / np.maximum(density, 1e-12) 
    mu = mu_max * S / (Ks + S)
    
    A_eff_n = np.power(A_eff, n)
    K_D_n = K_D ** n
    hill_term = A_eff_n / (K_D_n + A_eff_n + 1e-12)
    lambda_D = lambda_max * hill_term
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    
    dY = np.stack([dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt], axis=1)
    return dY.flatten()


def vec_linear_lysis(y_flat, t, N, V, A0_vec, mu_max, Ks, Y, a, b, K_A0, n):
    y = y_flat.reshape(N, 3)
    B_live = y[:, 0]
    S = y[:, 2]
    
    density = B_live / V
    mu = mu_max * S / (Ks + S)
    
    A0_n = np.power(A0_vec, n)
    K_A0_n = K_A0 ** n
    term_A0 = A0_n / (K_A0_n + A0_n + 1e-12)
    
    lambda_D = a * term_A0 * mu + b
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dY = np.stack([dB_live_dt, dB_dead_dt, dS_dt], axis=1)
    return dY.flatten()


def vec_combined_model(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b):
    y = y_flat.reshape(N, 5)
    A_free = y[:, 0]
    A_bound = y[:, 1]
    B_live = y[:, 2]
    S = y[:, 4]
    
    density = B_live / V
    A_eff = A_bound / np.maximum(density, 1e-12)
    mu = mu_max * S / (Ks + S)
    
    A_eff_n = np.power(A_eff, n)
    K_D_n = K_D ** n
    hill_term = A_eff_n / (K_D_n + A_eff_n + 1e-12)
    
    lambda_D = a * hill_term * mu + b
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    
    dY = np.stack([dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt], axis=1)
    return dY.flatten()


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
        params['dt'] = st.number_input("Step (h)", value=1.0, min_value=0.01, step=0.5, on_change=input_changed)

    # --- Population Gen ---
    st.sidebar.subheader("Population Generation")
    params['mean_log10'] = st.sidebar.number_input("Mean Log10 Volume", 1.0, 8.0, 3.0, 0.1, on_change=input_changed)
    params['std_log10'] = st.sidebar.number_input("Std Dev Log10", 0.1, 3.0, 1.2, 0.1, on_change=input_changed)
    params['n_samples'] = st.sidebar.number_input("N Samples (Droplets)", 1000, 100000, 17000, 1000, on_change=input_changed)
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
        
        defaults = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']
        for key in defaults:
            params[key] = 0

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

@st.cache_data
def generate_population(mean, std, n, conc, mean_pix, std_pix):
    """
    Generates population with integer pixel rounding (camera simulation).
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
    # Round to nearest integer to simulate pixel counting camera
    final_biomass = np.round(raw_biomass)
    
    # Physics check: Biomass cannot be zero if occupied
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


def calculate_batch_lambda(sol_reshaped, t_eval, vols, params, N_batch):
    """Reconstructs the Lambda_D (lysis rate) history from the solved states."""
    model = params['model']
    n_steps = sol_reshaped.shape[0]
    lambda_matrix = np.zeros((n_steps, N_batch))
    
    if model == "Linear Lysis Rate":
        B_live = sol_reshaped[:, :, 0]
        S = sol_reshaped[:, :, 2]
    else: # Eff Conc or Combined
        A_bound = sol_reshaped[:, :, 1]
        B_live = sol_reshaped[:, :, 2]
        S = sol_reshaped[:, :, 4]
        
    density = B_live / vols
    
    # Growth Rate (mu) history
    mu_mat = params['mu_max'] * S / (params['Ks'] + S)

    if model == "Effective Concentration":
        A_eff = A_bound / np.maximum(density, 1e-12)
        A_eff_n = np.power(A_eff, params['n_hill'])
        K_D_n = params['K_D'] ** params['n_hill']
        hill = A_eff_n / (K_D_n + A_eff_n + 1e-12)
        lambda_matrix = params['lambda_max'] * hill

    elif model == "Linear Lysis Rate":
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
        
    return lambda_matrix

def calculate_batch_a_eff(sol_reshaped, vols, params, N_batch):
    """Reconstructs the Effective Antibiotic Concentration (A_eff) history."""
    model = params['model']
    n_steps = sol_reshaped.shape[0]
    
    if model == "Linear Lysis Rate":
        # For simple model, "Effective Conc" is just A0
        return np.full((n_steps, N_batch), params['A0'])
        
    # For models with A_bound
    A_bound = sol_reshaped[:, :, 1]
    B_live = sol_reshaped[:, :, 2]
    
    density = B_live / vols
    
    # A_eff = A_bound / density
    A_eff = A_bound / np.maximum(density, 1e-12)
    
    return A_eff

def run_simulation(vols, initial_biomass, total_vols_range, params):
    BATCH_SIZE = 2000
    t_eval = np.arange(params['t_start'], params['t_end'] + params['dt']/100.0, params['dt'])
    
    if len(t_eval) < 2:
        t_eval = np.linspace(params['t_start'], params['t_end'], 2)

    N_STEPS = len(t_eval)
    N_occupied = len(vols)
    
    min_exp = int(np.floor(np.log10(total_vols_range[0])))
    max_exp = int(np.ceil(np.log10(total_vols_range[1])))
    bin_edges_log = np.arange(min_exp, max_exp + 1)
    bin_edges = 10 ** bin_edges_log
    n_bins = len(bin_edges) - 1
    
    # --- ACCUMULATORS ---
    bin_sums = np.zeros((n_bins, N_STEPS))
    a_eff_bin_sums = np.zeros((n_bins, N_STEPS))
    density_bin_sums = np.zeros((n_bins, N_STEPS))
    a_bound_bin_sums = np.zeros((n_bins, N_STEPS))
    net_rate_bin_sums = np.zeros((n_bins, N_STEPS))

    bin_counts = np.zeros(n_bins)
    final_counts_all = np.zeros(N_occupied)

    model = params['model']
    if model == "Linear Lysis Rate":
        idx_Blive = 0
        idx_S = 2
        num_vars = 3
        func = vec_linear_lysis
    else:
        idx_Blive = 2
        idx_S = 4
        num_vars = 5
        func = vec_effective_concentration if model == "Effective Concentration" else vec_combined_model

    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    n_batches = int(np.ceil(N_occupied / BATCH_SIZE))

    for i_batch in range(n_batches):
        start_idx = i_batch * BATCH_SIZE
        end_idx = min((i_batch + 1) * BATCH_SIZE, N_occupied)
        current_batch_size = end_idx - start_idx
        
        status_text.text(f"Simulating Batch {i_batch + 1}/{n_batches} ({current_batch_size} droplets)...")
        
        batch_vols = vols[start_idx:end_idx]
        batch_biomass = initial_biomass[start_idx:end_idx] 
        
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

        try:
            sol = odeint(func, y0_flat, t_eval, args=args)
            sol_reshaped = sol.reshape(N_STEPS, current_batch_size, num_vars)
            
            # --- CALCULATE INTERMEDIATES ---
            batch_lambda_vals = calculate_batch_lambda(sol_reshaped, t_eval, batch_vols, params, current_batch_size)
            batch_lambda_T = batch_lambda_vals.T
            
            # A_eff
            batch_a_eff_vals = calculate_batch_a_eff(sol_reshaped, batch_vols, params, current_batch_size)
            batch_a_eff_T = batch_a_eff_vals.T

            # Continuous Biomass and Substrate
            batch_blive_cont = sol_reshaped[:, :, idx_Blive]
            batch_S = sol_reshaped[:, :, idx_S]

            # --- NEW CALCULATIONS ---
            # 1. Growth Rate (mu)
            batch_mu = params['mu_max'] * batch_S / (params['Ks'] + batch_S)
            batch_mu_T = batch_mu.T

            # 2. Density (Biomass / Volume)
            batch_density = batch_blive_cont / batch_vols
            batch_density_T = batch_density.T

            # 3. Bound Antibiotic (if applicable)
            if model == "Linear Lysis Rate":
                batch_abound_T = np.zeros_like(batch_density_T)
            else:
                batch_abound_T = sol_reshaped[:, :, 1].T # Index 1 is A_bound

            # *** 4. NET RATE (mu - lambda) ***
            batch_net_rate = batch_mu_T - batch_lambda_T

            # --- DISCRETIZATION STEP (CAMERA) ---
            batch_blive = np.round(batch_blive_cont)
            
            final_c = np.mean(batch_blive[-2:, :], axis=0) if N_STEPS > 2 else batch_blive[-1, :]
            final_c = np.where(final_c < 2.0, 0.0, final_c)
            final_counts_all[start_idx:end_idx] = final_c
            
            batch_blive_T = batch_blive.T
            
            # Binning
            for b_idx in range(n_bins):
                low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
                mask = (batch_vols >= low) & (batch_vols < high)
                count_in_bin = np.sum(mask)
                
                if count_in_bin > 0:
                    bin_sums[b_idx, :] += np.sum(batch_blive_T[mask, :], axis=0)
                    a_eff_bin_sums[b_idx, :] += np.sum(batch_a_eff_T[mask, :], axis=0)
                    
                    # New Accumulators
                    density_bin_sums[b_idx, :] += np.sum(batch_density_T[mask, :], axis=0)
                    a_bound_bin_sums[b_idx, :] += np.sum(batch_abound_T[mask, :], axis=0)
                    
                    # Net Rate Accumulator
                    net_rate_bin_sums[b_idx, :] += np.sum(batch_net_rate[mask, :], axis=0)
                    
                    bin_counts[b_idx] += count_in_bin

        except Exception as e:
            st.error(f"Error in batch {i_batch}: {e}")
        
        progress_bar.progress((i_batch + 1) / n_batches)

    status_text.text(f"Simulation complete in {time.time() - start_time:.2f} seconds.")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return (bin_sums, bin_counts, final_counts_all, t_eval, bin_edges, 
            a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums)


# ==========================================
# 5. PLOTTING FUNCTIONS
# ==========================================

def int_to_superscript(n):
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))

def plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Normalized Biomass (B/B₀)", 
               height=800, width=1200, tools="pan,wheel_zoom,reset,save")
    
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    
    unique_bins = sum(1 for c in bin_counts if c > 0)
    
    if unique_bins > 0:
        colors = linear_palette(high_contrast_color_map, unique_bins)
    else:
        colors = []

    legend_items = []
    
    color_idx = 0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_traj = bin_sums[i, :] / bin_counts[i]
            initial_val = mean_traj[0]
            
            if initial_val > 1e-9:
                norm_traj = mean_traj / initial_val
            else:
                norm_traj = mean_traj 

            norm_traj = np.where(norm_traj <= 0, np.nan, norm_traj)
            
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)} (n={int(bin_counts[i])})"
            
            r = p.line(t_eval, norm_traj, line_color=colors[color_idx], line_width=3, alpha=0.9)
            legend_items.append((label, [r]))
            color_idx += 1

    total_biomass_traj = np.sum(bin_sums, axis=0)
    total_N0 = total_biomass_traj[0]

    if total_N0 > 1e-9:
        meta_norm = total_biomass_traj / total_N0
    else:
        meta_norm = total_biomass_traj

    r_meta = p.line(t_eval, meta_norm, line_color="white", line_width=4, 
                    line_dash="dashed", alpha=1.0)
    
    legend_items.insert(0, ("Metapopulation (Avg)", [r_meta]))
            
    legend = Legend(items=legend_items, title="Volume Bins", click_policy="hide")
    p.add_layout(legend, 'right')
    return p

# *** NEW PLOTTING FUNCTION FOR NET RATE ***
def plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges):
    p = figure(x_axis_label="Time (h)", y_axis_label="Net Growth Rate (μ - λ) [1/h]", 
               height=800, width=1200, tools="pan,wheel_zoom,reset,save",
               title="Net Growth Rate (μ - λ) Dynamics")
    
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    
    unique_bins = sum(1 for c in bin_counts if c > 0)
    colors = linear_palette(high_contrast_color_map, max(1, unique_bins)) if unique_bins > 0 else []

    color_idx = 0
    legend_items = []

    # Zero line for reference
    zero_line = Span(location=0, dimension='width', line_color='white', line_dash='dotted', line_width=2)
    p.add_layout(zero_line)

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
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)}"
            
            r = p.line(t_eval, mean_a_eff, line_color=colors[color_idx], line_width=3, alpha=0.8)
            legend_items.append((label, [r]))
            color_idx += 1

    # --- ADD THRESHOLD LINE (K_D or K_A0) ---
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
            # Log scale handling for zero
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
               width=1200, height=800, y_range=(-7, 9), output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    
    r_scat = p.scatter('Volume', 'FoldChange', source=source, color='silver', alpha=0.6, size=4)
    
    if not df_sub.empty:
        r_ma = p.line('Volume', 'MovingAverage', source=sub_source, color='red', line_width=3)
    else:
        r_ma = p.line([], [], color='red')
    
    min_v, max_v = df_fc['Volume'].min(), df_fc['Volume'].max()
    r_meta = p.line([min_v, max_v], [meta_fc, meta_fc], color='white', line_width=3)
    r_base = p.line([min_v, max_v], [0, 0], color='white', line_dash='dashdot', line_width=3)
    r_vc = p.line([vc_val, vc_val], [-10, 10], color='white', line_dash='dashed', line_width=3)
    
    p.add_tools(HoverTool(tooltips=[('Volume', '@Volume{0,0}'), ('Fold Change', '@FoldChange{0.00}'), ('ID', '@DropletID')], 
                          renderers=[r_scat]))
    
    legend = Legend(items=[
        LegendItem(label='Droplet FC', renderers=[r_scat]),
        LegendItem(label='FC Moving Avg', renderers=[r_ma]),
        LegendItem(label='Metapopulation FC', renderers=[r_meta]),
        LegendItem(label='Vc', renderers=[r_vc]),
        LegendItem(label='Baseline (0)', renderers=[r_base])
    ], location='top_right')
    p.add_layout(legend, 'right')
    
    # RETURN BOTH PLOT AND DATAFRAME FOR DOWNLOAD
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
    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "14pt"

    r_scat = p.scatter('Volume', 'Biomass', source=source, color='gray', alpha=0.6, size=5,
                       legend_label='Biomass vs. Volume')

    hover = HoverTool(tooltips=[('Volume', '@Volume{0,0}'), 
                                ('Biomass', '@Biomass{0.00}'), 
                                ('ID', '@DropletID')],
                      renderers=[r_scat])
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

    p.legend.label_text_font_size = "14pt"
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
    """Helper to convert DF to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 6. MAIN EXECUTION (Refactored for Single Figure Display)
# ==========================================

def main():
    configure_page()
    init_session_state()
    
    MEAN_PIXELS = 5.5
    STD_PIXELS = 1.0
    
    params = render_sidebar()
    
    check_debounce()
    
    # --- Generate Data ---
    vols, counts, initial_biomass, total_vols = generate_population(
        params['mean_log10'], params['std_log10'], params['n_samples'], 
        params['concentration'], MEAN_PIXELS, STD_PIXELS
    )
    
    n_trimmed = len(total_vols)
    N_occupied = len(vols)
    
    pct = (N_occupied / n_trimmed * 100) if n_trimmed > 0 else 0.0
    
    # --- Header Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Droplets (Simulated)", f"{n_trimmed:,}")
    col2.metric("Occupied Droplets", f"{N_occupied:,} ({pct:.2f}%)")
    col3.metric("Antibiotic Conc (A0)", f"{params['A0']}")

    if N_occupied == 0:
        st.error("No occupied droplets found. Try increasing Concentration or Mean Volume.")
        st.stop()
        
    df_density, vc_val = calculate_vc_and_density(vols, initial_biomass, params['concentration'], MEAN_PIXELS)
    
    # --- Run Simulation ---
    # Updated unpack to include net_rate_bin_sums and removed mu/lambda sums
    (bin_sums, bin_counts, final_biomass, t_eval, bin_edges, 
     a_eff_bin_sums, density_bin_sums, a_bound_bin_sums, net_rate_bin_sums) = run_simulation(
        vols, initial_biomass, (total_vols.min(), total_vols.max()), params
    )
    
    st.divider()
    
    # ==========================================
    # NEW UI LAYOUT: SELECTOR INSTEAD OF TABS
    # ==========================================
    
    st.subheader("Results Analysis")

    # The list of available figures
    plot_options = [
        "Population Dynamics", 
        "Droplet Distribution", 
        "Initial Density & Vc", 
        "Fold Change", 
        "N0 vs Volume",
        "Net Growth Rate (μ - λ)", # Merged Figure
        "Antibiotic Dynamics",
        "Density Dynamics",
        "Bound Antibiotic"
    ]

    # The Selector - Acts as the "Tab Switcher"
    selected_plot = st.selectbox("Select Figure to Display:", plot_options)

    # Container for the single figure
    with st.container():
        
        # --- 1. Population Dynamics ---
        if selected_plot == "Population Dynamics":
            st.markdown("#### Mean Growth curves (Normalized Biomass)")
            
            # Prepare Data
            data_dyn = {'Time': t_eval}
            for i in range(len(bin_counts)):
                if bin_counts[i] > 0:
                    mean_traj = bin_sums[i, :] / bin_counts[i]
                    initial_val = mean_traj[0]
                    norm_traj = mean_traj / initial_val if initial_val > 1e-9 else mean_traj
                    
                    low_exp = int(np.log10(bin_edges[i]))
                    high_exp = int(np.log10(bin_edges[i+1]))
                    col_name = f"Bin_10^{low_exp}_to_10^{high_exp}_(n={int(bin_counts[i])})"
                    data_dyn[col_name] = norm_traj
                    
            df_dynamics = pd.DataFrame(data_dyn)
            
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button("Download CSV", data=convert_df(df_dynamics), 
                                   file_name="dynamics_curves.csv", mime="text/csv")
            
            p = plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p, use_container_width=True)

        # --- 2. Droplet Distribution ---
        elif selected_plot == "Droplet Distribution":
            st.markdown("#### Droplet Distribution: Total vs Occupied")
            
            max_len = max(len(total_vols), len(vols))
            total_vols_pad = np.pad(total_vols, (0, max_len - len(total_vols)), constant_values=np.nan)
            occupied_vols_pad = np.pad(vols, (0, max_len - len(vols)), constant_values=np.nan)
            df_dist = pd.DataFrame({'Total_Droplet_Volumes': total_vols_pad, 'Occupied_Droplet_Volumes': occupied_vols_pad})
            
            st.download_button("Download CSV", data=convert_df(df_dist), 
                               file_name="volume_distribution.csv", mime="text/csv")
            
            p = plot_distribution(total_vols, vols)
            streamlit_bokeh(p, use_container_width=True)

        # --- 3. Initial Density & Vc ---
        elif selected_plot == "Initial Density & Vc":
            st.markdown("#### Initial Density & Vc Calculation")
            st.download_button("Download CSV", data=convert_df(df_density), 
                               file_name="initial_density_data.csv", mime="text/csv")
            
            p = plot_initial_density_vc(df_density, vc_val, params['concentration'])
            streamlit_bokeh(p, use_container_width=True)

        # --- 4. Fold Change ---
        elif selected_plot == "Fold Change":
            st.markdown("#### Biomass Fold Change vs Volume")
            p, df_fc = plot_fold_change(vols, initial_biomass, final_biomass, vc_val)
            
            st.download_button("Download CSV", data=convert_df(df_fc), 
                               file_name="fold_change_data.csv", mime="text/csv")
            streamlit_bokeh(p, use_container_width=True)

        # --- 5. N0 vs Volume ---
        elif selected_plot == "N0 vs Volume":
            st.markdown(f"#### Initial Biomass (N0) vs Volume")
            st.info(f"Regression is calculated for Volumes ≥ Vc ({vc_val:.1f} μm³)")
            st.download_button("Download CSV", data=convert_df(df_density), 
                               file_name="n0_vs_vol_data.csv", mime="text/csv")
            
            p = plot_n0_vs_volume(df_density, vc_val)
            streamlit_bokeh(p, use_container_width=True)

        # --- 6. NET GROWTH RATE (Merged Figure) ---
        elif selected_plot == "Net Growth Rate (μ - λ)":
            st.markdown("#### Net Growth Rate ($\mu - \lambda_D$)")
            st.info("This plot shows the effective growth rate. Positive values mean net growth, negative values mean net death (lysis > growth).")
            
            p_net = plot_net_growth_dynamics(t_eval, net_rate_bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p_net, use_container_width=True)

        # --- 7. Antibiotic Dynamics ---
        elif selected_plot == "Antibiotic Dynamics":
            st.markdown("#### Effective Antibiotic Concentration ($A_{bound} / \\rho$)")
            st.info("This metric normalizes bound antibiotic by cell density.")
            p_aeff = plot_a_eff_dynamics(t_eval, a_eff_bin_sums, bin_counts, bin_edges, params)
            streamlit_bokeh(p_aeff, use_container_width=True)

        # --- 8. Density Dynamics ---
        elif selected_plot == "Density Dynamics":
            st.markdown("#### Cell Density Dynamics ($B/V$)")
            st.info("Log-scale plot. Shows how biomass density changes over time for each bin.")
            p_dens = plot_density_dynamics(t_eval, density_bin_sums, bin_counts, bin_edges)
            streamlit_bokeh(p_dens, use_container_width=True)

        # --- 9. Bound Antibiotic ---
        elif selected_plot == "Bound Antibiotic":
            st.markdown("#### Bound Antibiotic ($A_{bound}$)")
            st.info("Shows the absolute amount of antibiotic bound to cells in a droplet.")
            if params['model'] == "Linear Lysis Rate":
                st.warning("This model does not simulate binding kinetics.")
            else:
                p_abound = plot_abound_dynamics(t_eval, a_bound_bin_sums, bin_counts, bin_edges)
                streamlit_bokeh(p_abound, use_container_width=True)

if __name__ == "__main__":
    main()


