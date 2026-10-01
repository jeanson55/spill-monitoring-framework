"""
GENERATE PHYSICS-BASED SPILL TRAJECTORIES
==========================================

Generates gravity-driven thin-film spill trajectories using

    dh/dt + div(q) = 0

where

    q = -(rho * g * h^3 / (3 * mu)) grad(h)

The numerical scheme is conservative and semi-implicit:

    (h_new - h_old) / dt
        = div(M_old * grad(h_new))

with

    M_old = rho*g*h_old^3/(3*mu)

This avoids the severe timestep restriction of a fully explicit
thin-film solver.

Outputs
-------
pinn/data/spill_trajectory_dataset.csv

Each row contains:
    trajectory_id
    time_s
    x
    y
    h
    rho
    mu
    g
    initial_xc
    initial_yc
    initial_sigma_x
    initial_sigma_y
    initial_amplitude
    visible
    xc
    yc
    area
    width
    height
    aspect_ratio
    volume
    max_thickness

The trajectory-level split is stored in:
    pinn/data/spill_trajectory_metadata.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "spill_trajectory_dataset.csv"
METADATA_PATH = DATA_DIR / "spill_trajectory_metadata.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)


# ============================================================
# NUMERICAL DOMAIN
# ============================================================

NX = 40
NY = 40

LX = 1.0          # m
LY = 1.0          # m

DX = LX / (NX - 1)
DY = LY / (NY - 1)

X = np.linspace(0.0, LX, NX)
Y = np.linspace(0.0, LY, NY)

XX, YY = np.meshgrid(X, Y, indexing="ij")


# ============================================================
# TIME SETTINGS
# ============================================================

N_TIME = 61
T_END = 300.0                 # seconds
OUTPUT_TIMES = np.linspace(0.0, T_END, N_TIME)

DT_INITIAL = 2.0
DT_MIN = 1e-3
DT_MAX = 10.0


# ============================================================
# TRAJECTORIES
# ============================================================

N_TRAJECTORIES = 30

# Number of trajectories in each partition
N_TRAIN = 21
N_VALID = 4
N_TEST = 5


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

G = 9.81                       # m/s^2

# Representative fluid-property ranges.
# These are simulation parameters, not experimental measurements.
RHO_RANGE = (800.0, 1100.0)    # kg/m^3
MU_RANGE = (0.005, 0.080)      # Pa.s


# ============================================================
# INITIAL SPILL GEOMETRY
# ============================================================

# Initial thickness range
AMPLITUDE_RANGE = (0.0015, 0.0060)       # m

# Initial Gaussian spread
SIGMA_X_RANGE = (0.035, 0.090)            # m
SIGMA_Y_RANGE = (0.035, 0.090)            # m

# Initial centroid constrained away from boundaries
XC_RANGE = (0.25, 0.75)
YC_RANGE = (0.25, 0.75)


# ============================================================
# VISIBLE-SPILL THRESHOLD
# ============================================================

# This threshold is used only to derive image-like geometry
# from the physical thickness field.
#
# It is NOT a detection threshold.
VISIBLE_THRESHOLD = 1e-5       # m


# ============================================================
# LINEAR SYSTEM CONSTRUCTION
# ============================================================

def build_implicit_matrix(mobility, dt):
    """
    Build the sparse matrix for

        h_new - dt * div(M grad(h_new)) = h_old

    using a conservative finite-volume discretization with
    zero-flux Neumann boundaries.

    The mobility is frozen at the previous timestep.
    """

    n = NX * NY

    A = lil_matrix((n, n), dtype=np.float64)

    def idx(i, j):
        return i * NY + j

    for i in range(NX):
        for j in range(NY):

            p = idx(i, j)

            diagonal = 1.0

            # East
            if i < NX - 1:
                m_face = 0.5 * (mobility[i, j] + mobility[i + 1, j])
                c = dt * m_face / DX**2
                A[p, idx(i + 1, j)] -= c
                diagonal += c

            # West
            if i > 0:
                m_face = 0.5 * (mobility[i, j] + mobility[i - 1, j])
                c = dt * m_face / DX**2
                A[p, idx(i - 1, j)] -= c
                diagonal += c

            # North
            if j < NY - 1:
                m_face = 0.5 * (mobility[i, j] + mobility[i, j + 1])
                c = dt * m_face / DY**2
                A[p, idx(i, j + 1)] -= c
                diagonal += c

            # South
            if j > 0:
                m_face = 0.5 * (mobility[i, j] + mobility[i, j - 1])
                c = dt * m_face / DY**2
                A[p, idx(i, j - 1)] -= c
                diagonal += c

            A[p, p] = diagonal

    return A.tocsr()


# ============================================================
# INITIAL CONDITION
# ============================================================

def make_initial_spill(
    xc,
    yc,
    sigma_x,
    sigma_y,
    amplitude,
):
    """
    Elliptical Gaussian spill thickness distribution.

    The background thickness is zero.
    """

    exponent = (
        ((XX - xc) ** 2) / (2.0 * sigma_x**2)
        + ((YY - yc) ** 2) / (2.0 * sigma_y**2)
    )

    h0 = amplitude * np.exp(-exponent)

    return h0


# ============================================================
# GEOMETRY EXTRACTION
# ============================================================

def extract_geometry(h):
    """
    Convert a physical thickness field into observable
    spill geometry.

    The geometry is derived from the visible region

        h >= VISIBLE_THRESHOLD
    """

    mask = h >= VISIBLE_THRESHOLD

    if not np.any(mask):
        return {
            "xc": np.nan,
            "yc": np.nan,
            "area": 0.0,
            "width": 0.0,
            "height": 0.0,
            "aspect_ratio": np.nan,
            "volume": float(np.sum(h) * DX * DY),
            "max_thickness": float(np.max(h)),
            "visible": 0,
        }

    xi = XX[mask]
    yi = YY[mask]

    xmin = np.min(xi)
    xmax = np.max(xi)
    ymin = np.min(yi)
    ymax = np.max(yi)

    width = xmax - xmin
    height = ymax - ymin

    area = float(np.sum(mask) * DX * DY)

    xc = float(np.sum(xi) / len(xi))
    yc = float(np.sum(yi) / len(yi))

    if height > 0.0:
        aspect_ratio = width / height
    else:
        aspect_ratio = np.nan

    volume = float(np.sum(h) * DX * DY)

    return {
        "xc": xc,
        "yc": yc,
        "area": area,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(aspect_ratio),
        "volume": volume,
        "max_thickness": float(np.max(h)),
        "visible": 1,
    }


# ============================================================
# SINGLE TRAJECTORY
# ============================================================

def simulate_trajectory(
    trajectory_id,
    rho,
    mu,
    xc,
    yc,
    sigma_x,
    sigma_y,
    amplitude,
):
    """
    Simulate one gravity-driven spill trajectory.
    """

    h = make_initial_spill(
        xc=xc,
        yc=yc,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        amplitude=amplitude,
    )

    # Ensure non-negative thickness.
    h = np.maximum(h, 0.0)

    records = []

    current_time = 0.0
    dt = DT_INITIAL

    for output_index, target_time in enumerate(OUTPUT_TIMES):

        while current_time < target_time - 1e-12:

            dt_use = min(dt, target_time - current_time)

            # Gravity-driven mobility:
            #
            # M(h) = rho*g*h^3/(3*mu)
            mobility = rho * G * np.maximum(h, 0.0)**3 / (3.0 * mu)

            A = build_implicit_matrix(mobility, dt_use)

            h_old = h.copy()

            h_new = spsolve(
                A,
                h_old.ravel()
            ).reshape(NX, NY)

            # Numerical protection.
            h_new = np.maximum(h_new, 0.0)

            # Relative change.
            denominator = np.maximum(
                np.max(np.abs(h_old)),
                1e-12
            )

            relative_change = (
                np.max(np.abs(h_new - h_old))
                / denominator
            )

            # Reject excessively large changes.
            if relative_change > 0.20 and dt_use > DT_MIN:

                dt = max(dt_use * 0.5, DT_MIN)
                continue

            h = h_new
            current_time += dt_use

            # Adapt timestep.
            if relative_change < 0.03:
                dt = min(dt * 1.25, DT_MAX)
            elif relative_change > 0.12:
                dt = max(dt * 0.70, DT_MIN)

        geometry = extract_geometry(h)

        # Store field values and trajectory-level geometry.
        for i in range(NX):
            for j in range(NY):

                records.append({
                    "trajectory_id": trajectory_id,
                    "time_s": float(target_time),

                    "x": float(X[i]),
                    "y": float(Y[j]),

                    "h": float(h[i, j]),

                    "rho": float(rho),
                    "mu": float(mu),
                    "g": float(G),

                    "initial_xc": float(xc),
                    "initial_yc": float(yc),
                    "initial_sigma_x": float(sigma_x),
                    "initial_sigma_y": float(sigma_y),
                    "initial_amplitude": float(amplitude),

                    "visible": int(geometry["visible"]),

                    "xc": geometry["xc"],
                    "yc": geometry["yc"],
                    "area": geometry["area"],
                    "width": geometry["width"],
                    "height": geometry["height"],
                    "aspect_ratio": geometry["aspect_ratio"],
                    "volume": geometry["volume"],
                    "max_thickness": geometry["max_thickness"],
                })

    return records


# ============================================================
# TRAJECTORY PARAMETERS
# ============================================================

def generate_parameters():

    parameters = []

    for trajectory_id in range(N_TRAJECTORIES):

        rho = rng.uniform(*RHO_RANGE)
        mu = rng.uniform(*MU_RANGE)

        xc = rng.uniform(*XC_RANGE)
        yc = rng.uniform(*YC_RANGE)

        sigma_x = rng.uniform(*SIGMA_X_RANGE)
        sigma_y = rng.uniform(*SIGMA_Y_RANGE)

        amplitude = rng.uniform(*AMPLITUDE_RANGE)

        parameters.append({
            "trajectory_id": trajectory_id,
            "rho": rho,
            "mu": mu,
            "initial_xc": xc,
            "initial_yc": yc,
            "initial_sigma_x": sigma_x,
            "initial_sigma_y": sigma_y,
            "initial_amplitude": amplitude,
        })

    return parameters


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHYSICS-BASED SPILL TRAJECTORY GENERATOR")
    print("=" * 70)

    print(f"Grid:              {NX} x {NY}")
    print(f"Domain:            {LX} m x {LY} m")
    print(f"Time:              0 - {T_END} s")
    print(f"Output times:      {N_TIME}")
    print(f"Trajectories:      {N_TRAJECTORIES}")
    print(f"Visible threshold: {VISIBLE_THRESHOLD:.2e} m")
    print()

    parameters = generate_parameters()

    # --------------------------------------------------------
    # Assign trajectory-level splits.
    # --------------------------------------------------------

    train_ids = list(range(0, N_TRAIN))
    valid_ids = list(range(N_TRAIN, N_TRAIN + N_VALID))
    test_ids = list(range(N_TRAIN + N_VALID, N_TRAJECTORIES))

    metadata = []

    for p in parameters:

        tid = p["trajectory_id"]

        if tid in train_ids:
            split = "train"
        elif tid in valid_ids:
            split = "validation"
        else:
            split = "test"

        metadata.append({
            **p,
            "split": split,
        })

    metadata_df = pd.DataFrame(metadata)

    # --------------------------------------------------------
    # Generate trajectories.
    # --------------------------------------------------------

    all_records = []

    for count, p in enumerate(parameters, start=1):

        tid = p["trajectory_id"]

        print(
            f"[{count:02d}/{N_TRAJECTORIES}] "
            f"Trajectory {tid:02d} | "
            f"rho={p['rho']:.1f} kg/m3 | "
            f"mu={p['mu']:.4f} Pa.s | "
            f"A={p['initial_amplitude']:.4f} m"
        )

        records = simulate_trajectory(
            trajectory_id=tid,
            rho=p["rho"],
            mu=p["mu"],
            xc=p["initial_xc"],
            yc=p["initial_yc"],
            sigma_x=p["initial_sigma_x"],
            sigma_y=p["initial_sigma_y"],
            amplitude=p["initial_amplitude"],
        )

        all_records.extend(records)

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    df = pd.DataFrame(all_records)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    metadata_df.to_csv(
        METADATA_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Basic integrity checks.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(f"Rows:              {len(df):,}")
    print(f"Columns:           {len(df.columns)}")
    print(f"Trajectories:      {df['trajectory_id'].nunique()}")
    print(f"Grid points/time:  {NX * NY:,}")
    print(f"Time points:       {N_TIME}")
    print()

    print("Trajectory split:")
    print(metadata_df["split"].value_counts().to_string())
    print()

    print("Thickness range:")
    print(
        f"min={df['h'].min():.6e}, "
        f"max={df['h'].max():.6e} m"
    )

    print("Volume range:")
    print(
        f"min={df['volume'].min():.6e}, "
        f"max={df['volume'].max():.6e} m3"
    )

    print("Area range:")
    print(
        f"min={df['area'].min():.6e}, "
        f"max={df['area'].max():.6e} m2"
    )

    print()

    # Mass conservation check using total volume.
    volume_by_traj = (
        df.groupby(["trajectory_id", "time_s"])["volume"]
        .first()
        .reset_index()
    )

    conservation_rows = []

    for tid, group in volume_by_traj.groupby("trajectory_id"):

        group = group.sort_values("time_s")

        initial_volume = group["volume"].iloc[0]
        final_volume = group["volume"].iloc[-1]

        relative_change = abs(
            final_volume - initial_volume
        ) / max(initial_volume, 1e-15)

        conservation_rows.append({
            "trajectory_id": tid,
            "initial_volume": initial_volume,
            "final_volume": final_volume,
            "relative_volume_change": relative_change,
        })

    conservation_df = pd.DataFrame(conservation_rows)

    print("Mass conservation:")
    print(
        f"maximum relative volume change = "
        f"{conservation_df['relative_volume_change'].max():.6e}"
    )

    print()

    print("Dataset:")
    print(OUTPUT_PATH)

    print("Metadata:")
    print(METADATA_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
