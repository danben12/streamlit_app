import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Span
from bokeh.palettes import Category10
from streamlit_bokeh import streamlit_bokeh
import time
from scipy.stats import linregress
from bokeh.layouts import column
from bokeh.models import Div

# ==========================================
# 1. PAGE CONFIG & STATE MANAGEMENT
# ==========================================

def configure_page():
    """Sets up the basic page title and layout."""
    st.set_page_config(page_title="Growth - Lysis Model simulation", layout="wide")
    st.title("Growth - Lysis Model simulation")

def init_session_state():
    """Initializes session variables if they don't exist."""
    if "last_change_time" not in st.session_state:
        st.session_state.last_change_time = time.time()

def input_changed():
    """Callback to update the debounce timer whenever a user changes an input."""
    st.session_state.last_change_time = time.time()

def check_debounce(delay=0.7):
    """
    Pauses execution until the user stops interacting for 'delay' seconds.
    This prevents the heavy simulation from running while typing.
    """
    while time.time() - st.session_state.last_change_time < delay:
        time.sleep(0.1)

# ==========================================
# 2. ODE MATH MODELS (VECTORIZED)
# ==========================================

def vec_effective_concentration(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n):
    """
    Model A: Effective Concentration.
    Antibiotic efficacy depends on the ratio of Bound Antibiotic to Biomass Density.
    """
    y = y_flat.reshape(N, 5)
    A_free = y[:, 0]
    A_bound = y[:, 1]
    B_live = y[:, 2]
    # B_dead = y[:, 3] (unused in calculation but present in array)
    S = y[:, 4]
    
    # Calculate Density and Effective Concentration
    density = B_live / V
    A_eff = A_bound / np.maximum(density, 1e-12) # Protect against div/0
    
    # Monod Growth
    mu = mu_max * S / (Ks + S)
    
    # Hill Function for Lysis
    A_eff_n = np.power(A_eff, n)
    K_D_n = K_D ** n
    hill_term = A_eff_n / (K_D_n + A_eff_n + 1e-12)
    lambda_D = lambda_max * hill_term
    
    # Derivatives
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dA_free_dt = -K_on * A_free * density + K_off * A_bound + lambda_D * A_bound
    dA_bound_dt = K_on * A_free * density - K_off * A_bound - lambda_D * A_bound
    
    dY = np.stack([dA_free_dt, dA_bound_dt, dB_live_dt, dB_dead_dt, dS_dt], axis=1)
    return dY.flatten()


def vec_linear_lysis(y_flat, t, N, V, A0_vec, mu_max, Ks, Y, a, b, K_A0, n):
    """
    Model B: Linear Lysis Rate.
    Lysis rate is linearly dependent on growth rate (mu).
    """
    y = y_flat.reshape(N, 3)
    B_live = y[:, 0]
    # B_dead = y[:, 1]
    S = y[:, 2]
    
    density = B_live / V
    mu = mu_max * S / (Ks + S)
    
    # Hill function on Initial Antibiotic (A0)
    A0_n = np.power(A0_vec, n)
    K_A0_n = K_A0 ** n
    term_A0 = A0_n / (K_A0_n + A0_n + 1e-12)
    
    # Lysis depends on growth rate (mu) + basal rate (b)
    lambda_D = a * term_A0 * mu + b
    
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    
    dY = np.stack([dB_live_dt, dB_dead_dt, dS_dt], axis=1)
    return dY.flatten()


def vec_combined_model(y_flat, t, N, V, mu_max, Ks, Y, K_on, K_off, K_D, n, a, b):
    """
    Model C: Combined Model.
    Uses effective concentration kinetics BUT lysis is growth-dependent.
    """
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
    
    # Lysis depends on Hill term * Growth Rate
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
    """
    Renders the sidebar inputs and returns a dictionary of all simulation parameters.
    """
    st.sidebar.header("Simulation Settings")
    
    params = {}
    
    # --- Model Selection ---
    model_choices = ["Effective Concentration", "Linear Lysis Rate", "Combined Model"]
    params['model'] = st.sidebar.selectbox("Select Model", model_choices, on_change=input_changed)

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
        
        # Initialize optional params to 0
        defaults = ['K_on', 'K_off', 'K_D', 'n_hill', 'lambda_max', 'a', 'b', 'K_A0']
        for key in defaults:
            params[key] = 0

        # Conditional Inputs based on Model Selection
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
def generate_population(mean, std, n, conc):
    """
    Generates droplet volumes using Log-Normal distribution and 
    loads bacteria using Poisson distribution.
    """
    log_data = np.random.normal(loc=mean, scale=std, size=int(n))
    volume_data = 10 ** log_data
    
    # Trim unreasonable volumes
    mask_vol = (volume_data >= 1000) & (volume_data <= 1e8)
    trimmed_vol = volume_data[mask_vol]
    
    # Poisson loading
    lambdas = trimmed_vol * conc
    bacteria_counts = np.random.poisson(lam=lambdas)
    
    # Filter occupied droplets
    occupied_mask = bacteria_counts > 0
    final_vols = trimmed_vol[occupied_mask].copy()
    final_bact = bacteria_counts[occupied_mask].copy()
    
    return final_vols, final_bact, trimmed_vol


def calculate_vc_and_density(vols, bacts, theoretical_conc):
    """
    Calculates the Initial Density statistics and finds the Critical Volume (Vc).
    Vc is where the rolling mean density converges to the theoretical bulk density.
    """
    df = pd.DataFrame({'Volume': vols, 'Count': bacts})
    df['InitialDensity'] = df['Count'] / df['Volume']
    df = df.sort_values(by='Volume').reset_index(drop=True)

    # Rolling Mean (Window 100)
    log_density = np.log10(df['InitialDensity'])
    df['RollingLogMean'] = log_density.rolling(window=100, min_periods=1).mean()
    df['RollingMeanDensity'] = 10 ** df['RollingLogMean']

    # --- Find Vc (Convergence Point) ---
    convergence_window = 2
    tolerance = 0.05
    # Relative difference from theoretical
    differences = np.abs(1 - (df['RollingMeanDensity'] / theoretical_conc))
    
    closest_index = differences.idxmin() # Default fallback if no convergence found
    
    # Scan for first stable convergence
    for i in range(len(differences) - convergence_window):
        window_mean_diff = differences.iloc[i:i + convergence_window].mean()
        if window_mean_diff <= tolerance:
            closest_index = i + convergence_window // 2
            break
            
    vc_val = df.loc[closest_index, 'Volume']
    return df, vc_val


def run_simulation(vols, bacts, total_vols_range, params):
    """
    Runs the main simulation loop using batch processing to handle thousands of droplets.
    """
    BATCH_SIZE = 2000
    N_STEPS = 250
    t_eval = np.linspace(0, 24, N_STEPS)
    N_occupied = len(vols)
    
    # --- Binning setup for "Dynamics" plot ---
    # We want to group droplets by volume decade (10^3, 10^4, etc.)
    min_exp = int(np.floor(np.log10(total_vols_range[0])))
    max_exp = int(np.ceil(np.log10(total_vols_range[1])))
    bin_edges_log = np.arange(min_exp, max_exp + 1)
    bin_edges = 10 ** bin_edges_log
    n_bins = len(bin_edges) - 1
    
    # Storage for binning
    bin_sums = np.zeros((n_bins, N_STEPS))
    bin_counts = np.zeros(n_bins)
    final_counts_all = np.zeros(N_occupied)

    # --- Determine Function & Vars based on Model ---
    model = params['model']
    if model == "Linear Lysis Rate":
        idx_Blive = 0
        num_vars = 3
        func = vec_linear_lysis
    else:
        idx_Blive = 2
        num_vars = 5
        func = vec_effective_concentration if model == "Effective Concentration" else vec_combined_model

    # --- Batch Processing Loop ---
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
        batch_bacts = bacts[start_idx:end_idx]
        
        # Prepare Initial Conditions (y0) and Args
        args = ()
        y0_flat = None
        
        if model == "Effective Concentration":
            y0_mat = np.zeros((current_batch_size, 5))
            y0_mat[:, 0] = params['A0']   # A_free
            y0_mat[:, 2] = batch_bacts    # B_live
            y0_mat[:, 4] = params['S0']   # Substrate
            y0_flat = y0_mat.flatten()
            args = (current_batch_size, batch_vols, params['mu_max'], params['Ks'], params['Y'], 
                    params['K_on'], params['K_off'], params['lambda_max'], params['K_D'], params['n_hill'])
            
        elif model == "Linear Lysis Rate":
            y0_mat = np.zeros((current_batch_size, 3))
            y0_mat[:, 0] = batch_bacts    # B_live
            y0_mat[:, 2] = params['S0']   # Substrate
            y0_flat = y0_mat.flatten()
            A0_vec = np.full(current_batch_size, params['A0'])
            args = (current_batch_size, batch_vols, A0_vec, params['mu_max'], params['Ks'], params['Y'], 
                    params['a'], params['b'], params['K_A0'], params['n_hill'])
            
        elif model == "Combined Model":
            y0_mat = np.zeros((current_batch_size, 5))
            y0_mat[:, 0] = params['A0']
            y0_mat[:, 2] = batch_bacts
            y0_mat[:, 4] = params['S0']
            y0_flat = y0_mat.flatten()
            args = (current_batch_size, batch_vols, params['mu_max'], params['Ks'], params['Y'], 
                    params['K_on'], params['K_off'], params['K_D'], params['n_hill'], params['a'], params['b'])

        try:
            # SOLVE ODE
            sol = odeint(func, y0_flat, t_eval, args=args)
            
            # Reshape result: (Time, Droplets, Variables)
            sol_reshaped = sol.reshape(N_STEPS, current_batch_size, num_vars)
            
            # Extract Live Bacteria Count
            batch_blive = sol_reshaped[:, :, idx_Blive]
            
            # --- Store Final Counts ---
            # Take mean of last 4 timepoints to smooth small oscillations
            final_c = np.mean(batch_blive[-4:, :], axis=0)
            
            # Apply Extinction Threshold: < 1 cell = Dead
            final_c = np.where(final_c < 1.0, 0.0, final_c)
            final_counts_all[start_idx:end_idx] = final_c
            
            # --- Online Aggregation for Plotting ---
            batch_blive_T = batch_blive.T
            for b_idx in range(n_bins):
                low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
                mask = (batch_vols >= low) & (batch_vols < high)
                count_in_bin = np.sum(mask)
                
                if count_in_bin > 0:
                    # Sum all trajectories in this volume bin
                    bin_sums[b_idx, :] += np.sum(batch_blive_T[mask, :], axis=0)
                    bin_counts[b_idx] += count_in_bin

        except Exception as e:
            st.error(f"Error in batch {i_batch}: {e}")
        
        progress_bar.progress((i_batch + 1) / n_batches)

    status_text.text(f"Simulation complete in {time.time() - start_time:.2f} seconds.")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return bin_sums, bin_counts, final_counts_all, t_eval, bin_edges


# ==========================================
# 5. PLOTTING FUNCTIONS
# ==========================================

def int_to_superscript(n):
    """Helper for formatting exponents in plot legends."""
    return str(n).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))

def plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges):
    """Tab 1: Plots average growth curves for each volume decade."""
    p = figure(x_axis_label="Time (h)", y_axis_label="Mean Bacteria Count", y_axis_type="log", 
               height=800, width=1200, tools="pan,wheel_zoom,reset,save")
    colors = Category10[10]
    legend_items = []
    
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mean_traj = bin_sums[i, :] / bin_counts[i]
            # Handle zeros for log plot
            mean_traj = np.where(mean_traj <= 0, np.nan, mean_traj)
            
            low_exp = int(np.log10(bin_edges[i]))
            high_exp = int(np.log10(bin_edges[i+1]))
            label = f"10{int_to_superscript(low_exp)} - 10{int_to_superscript(high_exp)} (n={int(bin_counts[i])})"
            
            r = p.line(t_eval, mean_traj, line_color=colors[i % 10], line_width=3, alpha=0.9)
            legend_items.append((label, [r]))
            
    legend = Legend(items=legend_items, title="Volume Bins", click_policy="hide")
    p.add_layout(legend, 'right')
    return p

def plot_distribution(total_vols, occupied_vols):
    """Tab 2: Histogram of generated vs occupied droplets."""
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
    """Tab 3: Initial Density scatter plot with Critical Volume (Vc) marker."""
    source = ColumnDataSource(df_density)
    p = figure(x_axis_type='log', y_axis_type='log', x_axis_label='Volume (μm³)', y_axis_label='Initial Density (cells/μm³)',
               width=1200, height=800, output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    
    # Scatter points (Silver for dark mode visibility)
    r_dens = p.scatter('Volume', 'InitialDensity', source=source, color='silver', alpha=0.6, size=4)
    r_roll = p.line(df_density['Volume'], df_density['RollingMeanDensity'], color='red', line_width=3)
    
    # Theoretical Density Line
    min_v, max_v = df_density['Volume'].min(), df_density['Volume'].max()
    r_theo = p.line([min_v, max_v], [theoretical_density, theoretical_density], color='white', line_width=3)
    
    # Vc Line (Vertical)
    p.add_layout(Span(location=vc_val, dimension='height', line_color='white', line_dash='dashed', line_width=3))
    
    # Dummy line for legend entry (Visual fix for Span not appearing in legend)
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

def plot_fold_change(vols, initial_bacts, final_bacts, vc_val):
    """Tab 4: Fold Change (Final/Initial) scatter plot."""
    min_fc = -6.0
    
    # Calculate Fold Change (Log2)
    with np.errstate(divide='ignore', invalid='ignore'):
        fc_raw = final_bacts / initial_bacts
        fc_log2 = np.log2(fc_raw)
    
    # Handle infinities (Dead droplets)
    fc_log2 = np.where(np.isneginf(fc_log2) | np.isnan(fc_log2) | (fc_log2 < min_fc), min_fc, fc_log2)
    
    df_fc = pd.DataFrame({'Volume': vols, 'FoldChange': fc_log2, 'DropletID': np.arange(len(vols))})
    df_fc = df_fc.sort_values(by='Volume').reset_index(drop=True)
    
    # Calculate Moving Average for surviving droplets
    df_sub = df_fc[df_fc['FoldChange'] > min_fc].copy()
    if not df_sub.empty:
        df_sub['MovingAverage'] = df_sub['FoldChange'].rolling(window=100, min_periods=1).mean()
    else:
        df_sub['MovingAverage'] = np.nan

    # Metapopulation Average
    total_initial = np.sum(initial_bacts)
    total_final = np.sum(final_bacts)
    meta_fc = np.log2(total_final / total_initial) if total_final > 0 else min_fc
    
    source = ColumnDataSource(df_fc)
    sub_source = ColumnDataSource(df_sub)
    
    p = figure(x_axis_type='log', y_axis_type='linear', x_axis_label='Volume (μm³)', y_axis_label='Log2 Biomass Fold Change',
               width=1200, height=800, y_range=(-7, 9), output_backend="webgl", tools="pan,wheel_zoom,reset,save")
    
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    
    r_scat = p.scatter('Volume', 'FoldChange', source=source, color='silver', alpha=0.6, size=4)
    
    # Moving Average Line
    if not df_sub.empty:
        r_ma = p.line('Volume', 'MovingAverage', source=sub_source, color='red', line_width=3)
    else:
        r_ma = p.line([], [], color='red')
    
    # Reference Lines
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
    return p

def plot_n0_vs_volume(df, Vc):
    """
    Tab 5: N0 vs Volume with Linear Regression for V >= Vc.
    """
    # Create a local copy to avoid modifying the global dataframe
    plot_df = df.copy()
    plot_df['DropletID'] = plot_df.index  # Add ID for hover tool

    source = ColumnDataSource(plot_df)
    
    # Initialize Figure
    p = figure(x_axis_type='log', y_axis_type='log',
               x_axis_label='Volume (μm³)', y_axis_label='Initial Count (N0)', 
               output_backend="webgl", width=1200, height=800, 
               tools="pan,wheel_zoom,reset,save")

    # Styling
    p.xaxis.axis_label_text_font_size = "16pt"
    p.yaxis.axis_label_text_font_size = "16pt"
    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "14pt"

    # Scatter Plot
    r_scat = p.scatter('Volume', 'Count', source=source, color='gray', alpha=0.6, size=5,
                       legend_label='N0 vs. Volume')

    # Hover Tool
    hover = HoverTool(tooltips=[('Volume', '@Volume{0,0}'), 
                                ('Count', '@Count'), 
                                ('ID', '@DropletID')],
                      renderers=[r_scat])
    p.add_tools(hover)

    # --- Linear Regression (Only for V >= Vc and Count > 0) ---
    filtered_df = plot_df[(plot_df['Count'] > 0) & (plot_df['Volume'] >= Vc)]
    
    stats_text = "Insufficient data for regression"
    
    if len(filtered_df) > 2:
        x = np.log10(filtered_df['Volume'])
        y = np.log10(filtered_df['Count'])
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Create regression line points across the whole range
        x_values = np.linspace(plot_df['Volume'].min(), plot_df['Volume'].max(), 100)
        y_values = 10 ** (intercept + slope * np.log10(x_values))
        
        p.line(x_values, y_values, color='red', legend_label='Linear Regression', line_width=3)
        
        stats_text = f'<b>Regression (V > Vc):</b><br>y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_value ** 2:.3f}'

    # Legend Styling
    p.legend.label_text_font_size = "14pt"
    p.legend.location = "top_left"

    # Stats Div (Info Box)
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {
        'text-align': 'center', 'margin': '10px auto', 'font-size': '14pt',
        'font-family': 'Arial, sans-serif', 'color': 'black', 
        'background-color': '#f0f2f6', 'border': '1px solid #ccc', 
        'padding': '15px', 'border-radius': '10px', 'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
    }

    return column(p, stats_div)


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    # 1. Setup
    configure_page()
    init_session_state()
    
    # 2. Render Sidebar & Get Params
    params = render_sidebar()
    
    # 3. Debounce
    check_debounce()
    
    # 4. Population Logic
    vols, bacts, total_vols = generate_population(params['mean_log10'], params['std_log10'], 
                                                  params['n_samples'], params['concentration'])
    
    n_trimmed = len(total_vols)
    N_occupied = len(vols)
    
    # Display Stats
    pct = (N_occupied / n_trimmed * 100) if n_trimmed > 0 else 0.0
    st.write(f"**Simulation Stats:** **{n_trimmed}** remain after trimming ($10^3 < V < 10^8$). "
             f"**{N_occupied}** are occupied (**{pct:.2f}%** occupation).")
    st.markdown(f"### Antibiotic Concentration ($A_0$): {params['A0']}")
    
    if N_occupied == 0:
        st.error("No occupied droplets found. Try increasing Concentration or Mean Volume.")
        st.stop()
        
    # 5. Pre-Simulation Calculations (Density & Vc)
    df_density, vc_val = calculate_vc_and_density(vols, bacts, params['concentration'])
    
    # 6. Run Simulation
    bin_sums, bin_counts, final_counts, t_eval, bin_edges = run_simulation(
        vols, bacts, (total_vols.min(), total_vols.max()), params
    )
    
    # 7. Visualization Tabs
    st.subheader("Results Analysis")
    
    # --- UPDATE: Added "Scaling Law" to the tabs list ---
    t1, t2, t3, t4, t5 = st.tabs(["Population Dynamics", "Droplet Distribution", "Initial Density & Vc", "Fold Change", "N0 vs Volume"])
    
    with t1:
        st.markdown("##### Mean Growth curves per volume bin")
        p = plot_dynamics(t_eval, bin_sums, bin_counts, bin_edges)
        streamlit_bokeh(p, use_container_width=True)
        
    with t2:
        st.markdown("##### Droplet Distribution: Total vs Occupied")
        p = plot_distribution(total_vols, vols)
        streamlit_bokeh(p, use_container_width=True)
        
    with t3:
        p = plot_initial_density_vc(df_density, vc_val, params['concentration'])
        streamlit_bokeh(p, use_container_width=True)
        
    with t4:
        st.markdown("##### Biomass Fold Change vs Volume")
        p = plot_fold_change(vols, bacts, final_counts, vc_val)
        streamlit_bokeh(p, use_container_width=True)

    # --- UPDATE: Added Tab 5 content ---
    with t5:
        st.markdown(f"##### N0 vs Volume")
        st.info(f"Regression is calculated for Volumes ≥ Vc ({vc_val:.1f} μm³)")
        # df_density contains 'Volume' and 'Count', which is what we need
        p = plot_n0_vs_volume(df_density, vc_val)
        streamlit_bokeh(p, use_container_width=True)
if __name__ == "__main__":
    main()




