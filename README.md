Vectorized Bacteria Growth & Lysis Simulation

Overview

This Streamlit application simulates bacterial population dynamics within water-in-oil emulsion droplets (or similar micro-compartments). It combines stochastic initial loading (Poisson distribution) with deterministic differential equations (ODEs) to model how bacteria grow and respond to antibiotics in confined volumes.

The simulation is highly optimized using vectorization, allowing it to solve coupled ODEs for tens of thousands of droplets simultaneously in seconds.

Features

Stochastic Initialization: Generates log-normal droplet volume distributions and loads bacteria using Poisson statistics.

Three Biological Models:

Effective Concentration: Models the inoculum effect where high bacterial density "dilutes" the available antibiotic.

Linear Lysis Rate: Assumes lysis is linearly dependent on the growth rate.

Combined Model: Merges binding kinetics with growth-dependent lysis.

High-Performance Solver: Uses scipy.integrate.odeint with vectorized batch processing.

Interactive Visualization:

Dynamics: Growth trajectories averaged by volume decade.

Distribution: Histograms of droplet sizes (Total vs. Occupied).

Analysis: Initial density plots and Critical Volume ($V_c$) calculation.

Fold Change: Biomass change ($\log_2$) vs. Volume.

Installation

Clone or Download this repository.

Ensure you have Python installed (3.8+ recommended).

Install the required dependencies:

pip install -r requirements.txt


Usage

Run the application using Streamlit:

streamlit run growth_lysis_simulation.py


The app will open in your default web browser (usually at http://localhost:8501).

Model Details

1. Effective Concentration

Assumes antibiotic efficacy depends on the ratio of bound antibiotic to biomass density.

Key parameters: $K_{on}$, $K_{off}$, $K_D$, Hill coefficient ($n$).

Logic: Small droplets with high initial density may survive because the effective antibiotic concentration per cell drops below the MIC.

2. Linear Lysis Rate

Assumes the rate of cell death (lysis) is proportional to the growth rate ($\mu$).

Equation: $\lambda_D = a \cdot \text{Hill}(A_0) \cdot \mu + b$

Logic: Antibiotics often work best on actively dividing cells. Fast-growing populations (usually in larger droplets with lower density) are killed faster.

3. Combined Model

A hybrid approach that uses the binding kinetics of Model 1 to determine the "Effective" concentration, but applies the growth-dependent lysis logic of Model 2.

Visual Outputs

The app provides four analysis tabs:

Population Dynamics: Shows mean growth curves over 24 hours, grouped by droplet volume (e.g., $10^3, 10^4, \dots \mu m^3$).

Droplet Distribution: A histogram showing the total population of droplets generated vs. the subset that actually contain bacteria.

Initial Density & $V_c$: Plots Initial Density vs. Volume. The Critical Volume ($V_c$) is the threshold where stochastic density fluctuations stabilize to the bulk average.

Fold Change: A "volcano-style" plot showing the Log2 Fold Change in biomass for every individual droplet.
