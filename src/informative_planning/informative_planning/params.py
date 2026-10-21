import numpy as np

# ── Environment / Map ──────────────────────────────────────────────────────────
origin = [-532.0, 162.0]     # [x, y] world coordinates of map origin (m)
grid_size = [100, 100]       # map dimensions in cells [width, height]
resolution = 1.0             # cell size (m/cell)

# ── Sensor / Field of View ─────────────────────────────────────────────────────
d_max = 35                   # maximum sensor range (m)
theta = 5 * np.pi / 24      # half-angle of horizontal FOV (rad) — 37.5 deg

# Inverse sensor model parameters (logistic sigmoid)
k0, d0 = 0.1, 40            # occupied model: P(occ | z, d)
k1, d1 = 0.03, 40           # free model:     P(free | z, d)

# ── Occupancy Grid Mapping ─────────────────────────────────────────────────────
p_low, p_high = 0.15, 0.9   # probability clamp range
alpha, beta = 0.99, 0.05    # dynamic map prediction blend weights (alpha*pred + beta*current)
decay_rate = 0.01           # visit-score decay rate (per second)

# ── Planner Selection ──────────────────────────────────────────────────────────
# Global planner: 'mppi' | 'receding_horizon' | 'frontier' | 'coverage' | 'myopic'
planner = 'mppi'

# ── MPPI Global Planner ────────────────────────────────────────────────────────
# Horizon and sampling are set in mppi_config.yaml; time_step must match t_step there.
time_step = 2.0             # planning timestep (s)
track_time_step = 0.1       # tracking controller timestep (s)

# ── Cost Function Weights ──────────────────────────────────────────────────────
# Stage cost: C = a1*Ce + a2*Ct + a3*Cc + a4*Cp
a1 = 1.0    # exploration (visit-score information gain) weight
a2 = 0.45   # target-tracking information gain weight
a3 = 0.05   # boundary / collision cost weight
a4 = 0.1    # path smoothness weight

# Ablation study: set to None for the full cost function, or one of:
#   'exploration' | 'tracking' | 'path+exploration' | 'tracking_coefficients'
cost_ablation = None

# Sensitivity ablation: multipliers on a1..a4 (1.0 = unchanged).
scale_a1 = 1.0
scale_a2 = 1.0
scale_a3 = 1.0
scale_a4 = 1.0

# ── Mission ────────────────────────────────────────────────────────────────────
mission_time = 180           # mission timeout (s)
max_targets = 20             # stop sim after intercepting this many targets

# ── Mode Switching ─────────────────────────────────────────────────────────────
# Range at which the planner switches from exploration to intercept mode.
switching_radius = 8.0       # range to switch to intercept mode (m), use r = d_max (35) for greedy interception

# ── USV Motion ─────────────────────────────────────────────────────────────────
u = 1.5                      # nominal cruise speed (m/s)

# ── LOS / Waypoint Guidance ────────────────────────────────────────────────────
l_d = 8                      # LOS lookahead distance (m)
r_tol = 4.5                  # waypoint acceptance radius (m)
x_tol = 1                   # along-track switching tolerance (m)

# ── Myopic / Receding-Horizon Baseline Planners ────────────────────────────────
n_cand = 15                  # number of candidate headings for myopic greedy planner

# ── Data Logging ───────────────────────────────────────────────────────────────
save_name = 'trial_mppi'     # output subdirectory name for logged metrics
