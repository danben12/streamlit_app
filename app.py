import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Bacteria Growth Models", layout="wide")
st.markdown(
    '<h1 style="font-size:32px; text-align:center;">Bacteria Growth Models under Different Antibiotic Concentrations and Volumes</h1>',
    unsafe_allow_html=True
)

with st.expander("About this App", expanded=False):
    st.markdown("""
    This application simulates and visualizes the growth of bacterial populations under varying antibiotic concentrations and droplet volumes using three different mathematical models.
    Adjust the parameters in the sidebar to see how they affect the growth dynamics.
    """)

# ------------------- Debounce Setup ------------------- #
if "last_change_time" not in st.session_state:
    st.session_state.last_change_time = time.time()

DEBOUNCE_DELAY = 0.6  # seconds

def input_changed():
    st.session_state.last_change_time = time.time()

# ------------------- Sidebar ------------------- #
st.sidebar.header("Model Parameters")

with st.sidebar.expander("Growth and Lysis Parameters", expanded=False):
    mu_max = st.number_input("Maximum Growth Rate (mu_max)", 0.1, 2.0, 0.7, 0.05, format="%.2f", on_change=input_changed)
    Y = st.number_input("Yield Coefficient (Y)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", on_change=input_changed)
    S0 = st.number_input("Initial substrate (S0)", 0.1, 5.0, 1.0, 0.1, format="%.2f", on_change=input_changed)
    Ks = st.number_input("Half-saturation Constant (Ks)", 0.1, 10.0, 2.0, 0.1, format="%.2f", on_change=input_changed)

with st.sidebar.expander("Antibiotic Parameters", expanded=False):
    K_on = st.number_input("Antibiotic binding rate (K_on)", 0, 2000, 750, 1, on_change=input_changed)
    K_off = st.number_input("Antibiotic unbinding rate (K_off)", 0.0, 0.1, 0.01, 0.001, format="%.3f", on_change=input_changed)
    K_D = st.number_input("Dissociation constant (K_D)", 0, 50000, 12000, 1, on_change=input_changed)
    lambda_max = st.number_input("Maximum Lysis Rate (lambda_max)", 0.1, 2.0, 1.0, 0.1, format="%.2f", on_change=input_changed)
    K_A0 = st.number_input("Initial antibiotic constant (K_A0)", 0, 50, 10, 1, on_change=input_changed)
    n = st.number_input("Hill coefficient (n)", 1, 50, 20, 1, on_change=input_changed)
    a = st.number_input("Baseline Lysis (a)", 0.0, 5.0, 3.0, 0.1, format="%.2f", on_change=input_changed)
    b = st.number_input("Growth-dependent Lysis (b)", 0.0, 1.0, 0.1, 0.05, format="%.2f", on_change=input_changed)

# ------------------- Debounce Check ------------------- #
while time.time() - st.session_state.last_change_time < DEBOUNCE_DELAY:
    time.sleep(0.1)
# ------------------- Models ------------------- #
def effective_concentration_model(y, t, mu_max, Ks, Y, K_on, K_off, lambda_max, V, K_D, n):
    """
    params:
    param y: vector of state variables [A_free, A_bound_live, B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param K_on: rate constant for antibiotic binding to live bacteria
    param K_off: rate constant for antibiotic unbinding from live bacteria
    param lambda_max: maximum lysis rate due to antibiotic
    param V: volume of the droplet
    param K_D: dissociation constant for antibiotic effect
    param n: Hill coefficient for antibiotic effect
    variables:
    A_free: concentration of free antibiotic
    A_bound_live: concentration of antibiotic bound to live bacteria
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    A_free, A_bound_live, B_live, B_dead, S = y
    density=B_live/V
    A_eff = A_bound_live / density
    mu = mu_max * S / (Ks + S)
    lambda_D = lambda_max * (A_eff ** n / (K_D ** n + A_eff ** n))
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound_live + lambda_D * A_bound_live
    dA_bound_live_dt = K_on * A_free * density - K_off * A_bound_live - lambda_D * A_bound_live
    return np.array([dA_free_dt, dA_bound_live_dt, dB_live_dt, dB_dead_dt, dS_dt])

def linear_Lysis_rate_model(y, t, mu_max, Ks, Y, a, b, V,A0,K_A0,n):
    """
        params:
    param y: vector of state variables [B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param a: baseline lysis rate
    param b: coefficient for growth-rate-dependent lysis
    param V: volume of the droplet
    param A0: initial antibiotic concentration
    param K_A0: dissociation constant for antibiotic effect
    param n: Hill coefficient for antibiotic effect
    variables:
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    B_live, B_dead, S = y
    density=B_live/V
    mu = mu_max * S / (Ks + S)
    lambda_D = a*(A0**n / (K_A0**n + A0**n))*mu+b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    return np.array([dB_live_dt, dB_dead_dt, dS_dt])

def combined_model (y, t, mu_max, Ks, Y, K_on, K_off, V, K_D, n, a, b):
    """
    params:
    param y: vector of state variables [A_free, A_bound_live, B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param K_on: rate constant for antibiotic binding to live bacteria
    param K_off: rate constant for antibiotic unbinding from live bacteria
    param lambda_max: maximum lysis rate due to antibiotic
    param V: volume of the droplet
    param K_D: dissociation constant for antibiotic effect based on effective concentration
    param n: Hill coefficient for antibiotic effect based on effective concentration
    param a: baseline lysis rate
    param b: coefficient for growth-rate-dependent lysis
    variables:
    A_free: concentration of free antibiotic
    A_bound_live: concentration of antibiotic bound to live bacteria
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    A_free, A_bound_live, B_live, B_dead, S = y
    density=B_live/V
    A_eff = A_bound_live / density
    mu = mu_max * S / (Ks + S)
    lambda_D = a*(A_eff ** n / (K_D ** n + A_eff ** n))*mu+b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound_live + lambda_D * A_bound_live
    dA_bound_live_dt = K_on * A_free * density - K_off * A_bound_live - lambda_D * A_bound_live
    return np.array([dA_free_dt, dA_bound_live_dt, dB_live_dt, dB_dead_dt, dS_dt])


# ------------------- Helper Function ------------------- #
def generate_subplot_data(model_func, V, A_free0, t, density, mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n, S0, a, b, K_A0):
    results = []
    for A0 in A_free0:
        if model_func == effective_concentration_model:
            y0 = [A0, 0, V*density, 0, S0]
            sol = odeint(model_func, y0, t, args=(mu_max, Ks, Y, K_on, K_off, lambda_max, V, K_D, n))
            data = pd.DataFrame(sol, columns=["A_free","A_bound_live","B_live","B_dead","S"])
        elif model_func == linear_Lysis_rate_model:
            y0 = [V*density, 0, S0]
            sol = odeint(model_func, y0, t, args=(mu_max, Ks, Y, a, b, V, A0, K_A0, n))
            data = pd.DataFrame(sol, columns=["B_live","B_dead","S"])
        elif model_func == combined_model:
            y0 = [A0, 0, V*density, 0, S0]
            sol = odeint(model_func, y0, t, args=(mu_max, Ks, Y, K_on, K_off, V, K_D, n, a, b))
            data = pd.DataFrame(sol, columns=["A_free","A_bound_live","B_live","B_dead","S"])
        data["time"] = t
        data["A0"] = A0
        data["normalized_B_live"] = data["B_live"] / data["B_live"].max()
        results.append(data)
    final_results = pd.concat(results)
    grid_data = final_results.pivot(index="A0", columns="time", values="normalized_B_live")
    return grid_data

# ------------------- Constants ------------------- #
volume = [1e3, 1e4, 1e5, 1e6, 1e7]
density_vals = [0.005742, 0.001325, 0.000619, 0.000499, 0.000402]
t = np.linspace(0, 24, 150)
A_free0 = np.linspace(0, 30, 100)

# ------------------- Generate Figure ------------------- #
model_funcs = [effective_concentration_model, linear_Lysis_rate_model, combined_model]
model_names = ["Effective Concentration", "Linear Lysis Rate", "Combined Model"]

fig, axes = plt.subplots(len(model_funcs), len(volume), figsize=(24, 14), dpi=300)
superscript_map = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')

for row, model_func in enumerate(model_funcs):
    for col, V in enumerate(volume):
        grid_data = generate_subplot_data(model_func, V, A_free0, t, density_vals[col],
                                         mu_max, Ks, Y, K_on, K_off, lambda_max, K_D, n, S0, a, b, K_A0)
        ax = axes[row, col]
        contour = ax.contourf(grid_data.columns, grid_data.index, grid_data.values, levels=200, cmap="jet", vmin=0, vmax=1)

        if row == 0:
            exponent = int(np.log10(V))
            exponent_sup = str(exponent).translate(superscript_map)
            ax.set_title(f"Volume = 10{exponent_sup}", fontsize=16, pad=18)

        if row == len(model_funcs) - 1:
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xticks(np.linspace(grid_data.columns.min(), grid_data.columns.max(), 5))
        else:
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(model_names[row], fontsize=16, labelpad=18)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_yticks(np.linspace(grid_data.index.min(), grid_data.index.max(), 7))
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12, wspace=0.15, hspace=0.18)
fig.text(0.04, 0.5, "Initial Antibiotic Concentration (A0)", va='center', rotation='vertical', fontsize=18)
fig.text(0.5, 0.06, "Time (hours)", ha='center', va='center', fontsize=18)

cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax, label="Normalized Live Bacteria (0-1)", orientation="vertical", ticks=[0,0.25,0.5,0.75,1])
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Normalized Live Bacteria (0-1)", fontsize=16)
plt.suptitle("Bacteria Population over Time and Antibiotic Concentration", y=0.97, fontsize=22)

st.pyplot(fig, clear_figure=True, use_container_width=True)

