"""
STAGE 2 — GEOMETRY + MASS-CONSTRAINED SPILL PINN
=================================================

Purpose
-------
Stage 1 demonstrated that the neural field can learn the synthetic spill
thickness field using supervised reconstruction alone.

Stage 2 introduces two additional constraints:

    1. Geometry consistency
    2. Mass/volume conservation

The PDE physics term is deliberately DISABLED.

Objective:

    L_total =
        L_data
        + lambda_geometry(epoch) * L_geometry
        + lambda_mass(epoch)     * L_mass

with lambda_geometry / lambda_mass linearly warmed up from 0 -> 1 over
GEOMETRY_WARMUP_EPOCHS / MASS_WARMUP_EPOCHS, and:

    lambda_physics  = 0.0

CHANGE LOG (fixes for the stalled-training run)
------------------------------------------------
1. VISIBLE_TEMPERATURE raised from 1e-5 to a value tied to H_REF
   (0.05 * H_REF). At 1e-5 the soft-visibility sigmoid was saturated
   for essentially every pixel, so area/centroid gradients only
   reached the handful of pixels sitting exactly at the threshold.
   That starved the geometry loss of any real training signal.

2. Softplus output-activation beta lowered from 5.0 to 1.0, and the
   redundant "/5.0" rescale removed. At beta=5 the activation's
   derivative collapses to ~0 whenever the pre-activation goes
   negative, which happens naturally while the model is
   under-predicting thickness. That made it progressively harder to
   correct once the model started producing near-zero output.
   The bias initialization is re-derived analytically for beta=1
   so the model still starts near the empirical target mean.

3. Geometry/mass loss weights are now warmed up from 0 -> 1 over the
   first N epochs instead of being pinned to 1.0 from epoch 1. With
   raw loss magnitudes of roughly data=0.009, geometry=0.10,
   mass=1.0, equal weighting let mass/geometry dominate the gradient
   ~100:1 and ~10:1 over the actual spatial data fit, which is
   consistent with the near-flat validation curves observed.

4. EPOCHS raised substantially (10 -> 300) and a
   ReduceLROnPlateau scheduler added, since escaping the earlier
   near-flat regime needs materially more optimization steps even
   after fixes 1-3.

5. A one-time diagnostic print of the foreground (visible-spill)
   pixel fraction is added after dataset construction, so class
   imbalance in FOREGROUND_WEIGHT can be sanity-checked directly
   against the data rather than assumed.

Dataset
-------
Field:
    pinn/data/spill_pinn_field.csv

Geometry state:
    pinn/data/spill_pinn_geometry_state.csv

Merge key:
    trajectory_id + time_s

Trajectory split:
    Train: 0-20
    Validation: 21-24
    Test: 25-29

Model inputs
------------
x, y, time,
rho, mu,
xc, yc,
area, width, height, aspect_ratio

Target
------
h [m]

Output
------
h [m], constrained to be non-negative.

Author:
    Spill Detection / Physics-Informed Spill Modeling Project
"""

from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

ROOT = Path(__file__).resolve().parents[2]

FIELD_FILE = ROOT / "pinn" / "data" / "spill_pinn_field_ic_conditioned.csv"
STATE_FILE = ROOT / "pinn" / "data" / "spill_pinn_geometry_state_ic_conditioned.csv"

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# FIX (4): 10 epochs was nowhere near enough for the coupled objective to
# escape the near-flat regime, even after the gradient fixes below.
EPOCHS = 100

FIELD_BATCH_SIZE = 4096

# Number of complete trajectory/time states used for geometry/mass loss
STATE_BATCH_SIZE = 2

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

# FIX (3): warm the geometry/mass constraints in instead of pinning them
# to full strength from epoch 1, so the base data fit gets priority early.
#
# FIX (6): the previous 20-epoch warmup still let val RMSE degrade
# permanently from 2.3e-4 m (epoch ~5) to a stuck 8.2e-4 m plateau, because
# geometry loss's raw magnitude (~0.49) was ~20-25x larger than data loss's
# raw magnitude (~0.02) even at matched weight=1.0. A longer warmup combined
# with runtime loss-scale normalization (see loss_scale_geometry /
# loss_scale_mass below) gives the optimizer more room to settle without
# geometry's much larger raw scale overwhelming the data-fit gradient.
GEOMETRY_WARMUP_EPOCHS = 40
MASS_WARMUP_EPOCHS = 40

# FIX (6): clip gradient norm to damp the transition as warmup weights
# increase -- the previous run showed a data-loss spike right around the
# epoch where geometry_weight crossed ~0.55-0.60.
GRAD_CLIP_NORM = 1.0

# LR scheduler: halve the LR once validation RMSE plateaus.
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
SCHEDULER_MIN_LR = 1e-6


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

INPUT_DIM = 16
HIDDEN_DIM = 96
NUM_HIDDEN_LAYERS = 5

# FIX (2): beta=5 caused the Softplus derivative to collapse to ~0 whenever
# the pre-activation went negative (i.e. whenever the model was
# under-predicting), which made recovery from under-prediction very slow.
# beta=1 keeps a usable gradient over a much wider range of pre-activations.
OUTPUT_SOFTPLUS_BETA = 1.0


# =============================================================================
# PHYSICAL REFERENCE SCALES
# =============================================================================

H_REF = 0.006       # m
T_REF = 300.0       # s
L_REF = 1.0         # m

RHO_REF = 950.0     # kg/m3
MU_REF = 0.04       # Pa.s

AREA_REF = 0.5      # m2


# =============================================================================
# LOSS WEIGHTS
# =============================================================================

DATA_WEIGHT = 1.0

# FIX (6): these are now "relative to data loss" weights, not raw weights,
# because the training loop multiplies each one by a one-time-computed
# loss_scale_geometry / loss_scale_mass factor (see train()) that rescales
# geometry_loss and mass_loss's raw magnitude to match data_loss's raw
# magnitude. Previously GEOMETRY_WEIGHT=1.0 meant "geometry's ~0.49 raw
# loss vs data's ~0.02 raw loss", i.e. geometry was ~20-25x more influential
# than data at "equal" weight -- that mismatch is what caused val RMSE to
# degrade from 2.3e-4 m to a stuck 8.2e-4 m as the warmup schedule ramped up.
# A final weight of 0.3 here now means "about 30% as important as the data
# fit" in comparable units, not 30% of an already-20x-oversized quantity.
GEOMETRY_WEIGHT = 0.3
MASS_WEIGHT = 0.3

# Stage 2B: explicit trajectory-level conservation penalty.
# L_mass matches predicted volume to the sampled state's target volume.
# L_conservation instead matches every sampled state to its trajectory V0.
DEFAULT_CONSERVATION_WEIGHT = 0.3
CONSERVATION_WARMUP_EPOCHS = 40

# Deliberately zero in Stage 2.
PHYSICS_WEIGHT = 0.0


def geometry_weight_schedule(epoch: int) -> float:
    """Linear warmup 0 -> GEOMETRY_WEIGHT over GEOMETRY_WARMUP_EPOCHS."""
    if GEOMETRY_WARMUP_EPOCHS <= 0:
        return GEOMETRY_WEIGHT
    return GEOMETRY_WEIGHT * min(1.0, epoch / GEOMETRY_WARMUP_EPOCHS)


def mass_weight_schedule(epoch: int) -> float:
    """Linear warmup 0 -> MASS_WEIGHT over MASS_WARMUP_EPOCHS."""
    if MASS_WARMUP_EPOCHS <= 0:
        return MASS_WEIGHT
    return MASS_WEIGHT * min(1.0, epoch / MASS_WARMUP_EPOCHS)


def conservation_weight_schedule(epoch: int, conservation_weight: float) -> float:
    """Linear warmup 0 -> conservation_weight."""
    if CONSERVATION_WARMUP_EPOCHS <= 0:
        return conservation_weight
    return conservation_weight * min(1.0, epoch / CONSERVATION_WARMUP_EPOCHS)


# =============================================================================
# GEOMETRY CONFIGURATION
# =============================================================================

GRID_NX = 40
GRID_NY = 40

CELL_AREA = (
    1.0 / (GRID_NX * GRID_NY)
)

VISIBLE_THRESHOLD = 1e-5

# FIX (1): the soft-visibility mask is a sigmoid((h - threshold) / T).
# At T=1e-5, with target thicknesses in the 1e-6 .. 3e-3 m range, the
# sigmoid argument was deeply saturated for nearly every pixel, so its
# local derivative was ~0 almost everywhere and area/centroid losses could
# not meaningfully train the network. Tying T to H_REF keeps the soft
# transition band wide enough to carry a real gradient while still being
# narrow relative to the thickness scale.
VISIBLE_TEMPERATURE = 0.05 * H_REF  # = 3e-4, was a fixed 1e-5

FOREGROUND_WEIGHT = 5.0


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =============================================================================
# FIELD DATASET
# =============================================================================

class SpillFieldDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

        required_columns = [
            "trajectory_id",
            "time_s",
            "x",
            "y",
            "rho",
            "mu",
            "xc",
            "yc",
            "area",
            "width",
            "height",
            "aspect_ratio",
            "initial_xc",
            "initial_yc",
            "initial_sigma_x",
            "initial_sigma_y",
            "initial_amplitude",
            "h",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required field columns: "
                f"{missing}"
            )

        # ---------------------------------------------------------------------
        # Normalize model inputs.
        # ---------------------------------------------------------------------

        x = (
            self.df["x"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        y = (
            self.df["y"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        time = (
            self.df["time_s"]
            .to_numpy(dtype=np.float32)
            / T_REF
        )

        rho = (
            self.df["rho"]
            .to_numpy(dtype=np.float32)
            / RHO_REF
        )

        mu = (
            self.df["mu"]
            .to_numpy(dtype=np.float32)
            / MU_REF
        )

        xc = (
            self.df["xc"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        yc = (
            self.df["yc"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Physical area is normalized using AREA_REF.
        # ---------------------------------------------------------------------

        area = (
            self.df["area"]
            .to_numpy(dtype=np.float32)
            / AREA_REF
        )

        width = (
            self.df["width"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        height = (
            self.df["height"]
            .to_numpy(dtype=np.float32)
            / L_REF
        )

        aspect_ratio = (
            self.df["aspect_ratio"]
            .to_numpy(dtype=np.float32)
        )

        initial_xc = self.df["initial_xc"].to_numpy(dtype=np.float32) / L_REF
        initial_yc = self.df["initial_yc"].to_numpy(dtype=np.float32) / L_REF
        initial_sigma_x = self.df["initial_sigma_x"].to_numpy(dtype=np.float32) / L_REF
        initial_sigma_y = self.df["initial_sigma_y"].to_numpy(dtype=np.float32) / L_REF
        initial_amplitude = self.df["initial_amplitude"].to_numpy(dtype=np.float32) / H_REF

        self.inputs = np.column_stack(
            [
                x,
                y,
                time,
                rho,
                mu,
                xc,
                yc,
                area,
                width,
                height,
                aspect_ratio,
                initial_xc,
                initial_yc,
                initial_sigma_x,
                initial_sigma_y,
                initial_amplitude,
            ]
        ).astype(np.float32)

        # Physical thickness in metres.
        self.target = (
            self.df["h"]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )

        # Foreground indicator.
        self.foreground = (
            self.target[:, 0]
            > VISIBLE_THRESHOLD
        ).astype(np.float32)

        if not np.isfinite(
            self.inputs
        ).all():

            raise ValueError(
                "Non-finite model input detected."
            )

        if not np.isfinite(
            self.target
        ).all():

            raise ValueError(
                "Non-finite target detected."
            )

    def __len__(self):

        return len(self.target)

    def __getitem__(self, index):

        return (
            torch.from_numpy(
                self.inputs[index]
            ),

            torch.from_numpy(
                self.target[index]
            ),

            torch.tensor(
                self.foreground[index],
                dtype=torch.float32,
            ),
        )


# =============================================================================
# COMPLETE STATE DATASET
# =============================================================================

class SpillStateDataset(Dataset):
    """
    Dataset returning complete 40x40 spatial states.

    Each sample corresponds to one trajectory/time pair and contains:

        x, y, h
        rho, mu, g
        xc, yc, area, width, height, aspect_ratio
        target volume

    IMPORTANT:
    This dataset must be constructed from the FIELD dataframe because
    each trajectory/time state contains 1600 spatial cells.

    The geometry metadata are already repeated on those field rows,
    so the first row of each trajectory/time group is sufficient to
    recover the geometry target.
    """

    def __init__(self, field_df):
        self.states = []

        required_columns = [
            "trajectory_id",
            "time_s",
            "x",
            "y",
            "h",
            "rho",
            "mu",
            "g",
            "xc",
            "yc",
            "area",
            "width",
            "height",
            "aspect_ratio",
            "initial_xc",
            "initial_yc",
            "initial_sigma_x",
            "initial_sigma_y",
            "initial_amplitude",
        ]

        missing = [
            col for col in required_columns
            if col not in field_df.columns
        ]

        if missing:
            raise ValueError(
                "SpillStateDataset is missing required columns:\n"
                f"{missing}"
            )

        # --------------------------------------------------------------
        # Group the FIELD dataframe into complete spatial states
        # --------------------------------------------------------------
        grouped = field_df.groupby(
            ["trajectory_id", "time_s"],
            sort=True
        )

        for (trajectory_id, time_s), group in grouped:

            # ----------------------------------------------------------
            # Each trajectory/time state must contain the complete
            # 40 x 40 spatial grid.
            # ----------------------------------------------------------
            if len(group) != GRID_NX * GRID_NY:
                raise ValueError(
                    "Incomplete spatial state detected: "
                    f"trajectory={trajectory_id}, "
                    f"time={time_s}, "
                    f"cells={len(group)}, "
                    f"expected={GRID_NX * GRID_NY}"
                )

            # ----------------------------------------------------------
            # Sort spatial coordinates so the field has deterministic
            # ordering.
            # ----------------------------------------------------------
            group = group.sort_values(
                ["x", "y"]
            ).reset_index(drop=True)

            # ----------------------------------------------------------
            # Extract spatial field
            # ----------------------------------------------------------
            x = group["x"].to_numpy(dtype=np.float32)
            y = group["y"].to_numpy(dtype=np.float32)
            h = group["h"].to_numpy(dtype=np.float32)

            # ----------------------------------------------------------
            # Physical parameters
            # ----------------------------------------------------------
            rho = float(group["rho"].iloc[0])
            mu = float(group["mu"].iloc[0])
            g = float(group["g"].iloc[0])

            # ----------------------------------------------------------
            # Geometry metadata.
            #
            # These values are repeated across the 1600 field rows,
            # so taking the first row is sufficient.
            # ----------------------------------------------------------
            xc = float(group["xc"].iloc[0])
            yc = float(group["yc"].iloc[0])
            area = float(group["area"].iloc[0])
            width = float(group["width"].iloc[0])
            height = float(group["height"].iloc[0])
            aspect_ratio = float(
                group["aspect_ratio"].iloc[0]
            )
            initial_xc = float(group["initial_xc"].iloc[0])
            initial_yc = float(group["initial_yc"].iloc[0])
            initial_sigma_x = float(group["initial_sigma_x"].iloc[0])
            initial_sigma_y = float(group["initial_sigma_y"].iloc[0])
            initial_amplitude = float(group["initial_amplitude"].iloc[0])

            # ----------------------------------------------------------
            # Target volume.
            #
            # The field is represented on a 1 x 1 m domain with
            # uniform cell area.
            # ----------------------------------------------------------
            cell_area = 1.0 / (
                GRID_NX * GRID_NY
            )

            target_volume = float(
                np.sum(h) * cell_area
            )

            self.states.append(
                {
                    "trajectory_id": int(trajectory_id),
                    "time_s": float(time_s),

                    "x": x,
                    "y": y,
                    "h": h,

                    "rho": rho,
                    "mu": mu,
                    "g": g,

                    "xc": xc,
                    "yc": yc,
                    "area": area,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "initial_xc": initial_xc,
                    "initial_yc": initial_yc,
                    "initial_sigma_x": initial_sigma_x,
                    "initial_sigma_y": initial_sigma_y,
                    "initial_amplitude": initial_amplitude,

                    "volume": target_volume,
                }
            )

        if len(self.states) == 0:
            raise ValueError(
                "SpillStateDataset contains no complete states."
            )

        # Recover V0 explicitly from the earliest target state of each trajectory.
        initial_volume_by_trajectory = {}
        for trajectory_id in sorted({
            state["trajectory_id"] for state in self.states
        }):
            trajectory_states = [
                state for state in self.states
                if state["trajectory_id"] == trajectory_id
            ]
            initial_state = min(
                trajectory_states,
                key=lambda state: state["time_s"],
            )
            initial_volume_by_trajectory[trajectory_id] = initial_state["volume"]

        initial_state_by_trajectory = {}
        for trajectory_id in sorted({
            state["trajectory_id"] for state in self.states
        }):
            trajectory_states = [
                state for state in self.states
                if state["trajectory_id"] == trajectory_id
            ]
            initial_state_by_trajectory[trajectory_id] = min(
                trajectory_states,
                key=lambda state: state["time_s"],
            )

        for state in self.states:
            initial_state = initial_state_by_trajectory[state["trajectory_id"]]
            state["initial_volume"] = initial_state["volume"]
            state["initial_xc_state"] = initial_state["xc"]
            state["initial_yc_state"] = initial_state["yc"]
            state["initial_area_state"] = initial_state["area"]
            state["initial_width_state"] = initial_state["width"]
            state["initial_height_state"] = initial_state["height"]
            state["initial_aspect_ratio_state"] = initial_state["aspect_ratio"]

        print(
            f"Constructed SpillStateDataset: "
            f"{len(self.states):,} complete states"
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]

        return {
            "trajectory_id": state["trajectory_id"],
            "time_s": state["time_s"],

            "x": torch.from_numpy(
                state["x"].copy()
            ),

            "y": torch.from_numpy(
                state["y"].copy()
            ),

            "h": torch.from_numpy(
                state["h"].copy()
            ),

            "rho": torch.tensor(
                state["rho"],
                dtype=torch.float32
            ),

            "mu": torch.tensor(
                state["mu"],
                dtype=torch.float32
            ),

            "g": torch.tensor(
                state["g"],
                dtype=torch.float32
            ),

            "xc": torch.tensor(
                state["xc"],
                dtype=torch.float32
            ),

            "yc": torch.tensor(
                state["yc"],
                dtype=torch.float32
            ),

            "area": torch.tensor(
                state["area"],
                dtype=torch.float32
            ),

            "width": torch.tensor(
                state["width"],
                dtype=torch.float32
            ),

            "height": torch.tensor(
                state["height"],
                dtype=torch.float32
            ),

            "aspect_ratio": torch.tensor(
                state["aspect_ratio"],
                dtype=torch.float32
            ),

            "initial_xc": torch.tensor(state["initial_xc"], dtype=torch.float32),
            "initial_yc": torch.tensor(state["initial_yc"], dtype=torch.float32),
            "initial_sigma_x": torch.tensor(state["initial_sigma_x"], dtype=torch.float32),
            "initial_sigma_y": torch.tensor(state["initial_sigma_y"], dtype=torch.float32),
            "initial_amplitude": torch.tensor(state["initial_amplitude"], dtype=torch.float32),

            "volume": torch.tensor(
                state["volume"],
                dtype=torch.float32
            ),

            # Initial target volume V0 for the trajectory. This is used by
            # the trajectory-level conservation loss.
            "initial_volume": torch.tensor(
                state["initial_volume"],
                dtype=torch.float32
            ),

            "initial_xc_state": torch.tensor(state["initial_xc_state"], dtype=torch.float32),
            "initial_yc_state": torch.tensor(state["initial_yc_state"], dtype=torch.float32),
            "initial_area_state": torch.tensor(state["initial_area_state"], dtype=torch.float32),
            "initial_width_state": torch.tensor(state["initial_width_state"], dtype=torch.float32),
            "initial_height_state": torch.tensor(state["initial_height_state"], dtype=torch.float32),
            "initial_aspect_ratio_state": torch.tensor(
                state["initial_aspect_ratio_state"], dtype=torch.float32
            ),
        }

# =============================================================================
# COLLATE FUNCTION FOR COMPLETE STATES
# =============================================================================


def state_collate(batch):
    """
    Collate complete 40x40 spill states.

    Each item contains 1600 spatial cells plus scalar physical
    parameters and geometry targets.
    """

    # --------------------------------------------------------------
    # Spatial fields
    # --------------------------------------------------------------
    x = np.stack(
        [item["x"].numpy() for item in batch],
        axis=0
    )

    y = np.stack(
        [item["y"].numpy() for item in batch],
        axis=0
    )

    h = np.stack(
        [item["h"].numpy() for item in batch],
        axis=0
    )

    # --------------------------------------------------------------
    # Scalar physical parameters
    # --------------------------------------------------------------
    rho = np.asarray(
        [item["rho"].item() for item in batch],
        dtype=np.float32
    )

    mu = np.asarray(
        [item["mu"].item() for item in batch],
        dtype=np.float32
    )

    g = np.asarray(
        [item["g"].item() for item in batch],
        dtype=np.float32
    )

    # --------------------------------------------------------------
    # Geometry targets
    # --------------------------------------------------------------
    xc = np.asarray(
        [item["xc"].item() for item in batch],
        dtype=np.float32
    )

    yc = np.asarray(
        [item["yc"].item() for item in batch],
        dtype=np.float32
    )

    area = np.asarray(
        [item["area"].item() for item in batch],
        dtype=np.float32
    )

    width = np.asarray(
        [item["width"].item() for item in batch],
        dtype=np.float32
    )

    height = np.asarray(
        [item["height"].item() for item in batch],
        dtype=np.float32
    )

    aspect_ratio = np.asarray(
        [item["aspect_ratio"].item() for item in batch],
        dtype=np.float32
    )

    initial_xc = np.asarray([item["initial_xc"].item() for item in batch], dtype=np.float32)
    initial_yc = np.asarray([item["initial_yc"].item() for item in batch], dtype=np.float32)
    initial_sigma_x = np.asarray([item["initial_sigma_x"].item() for item in batch], dtype=np.float32)
    initial_sigma_y = np.asarray([item["initial_sigma_y"].item() for item in batch], dtype=np.float32)
    initial_amplitude = np.asarray([item["initial_amplitude"].item() for item in batch], dtype=np.float32)

    initial_xc_state = np.asarray(
        [item["initial_xc_state"].item() for item in batch], dtype=np.float32
    )
    initial_yc_state = np.asarray(
        [item["initial_yc_state"].item() for item in batch], dtype=np.float32
    )
    initial_area_state = np.asarray(
        [item["initial_area_state"].item() for item in batch], dtype=np.float32
    )
    initial_width_state = np.asarray(
        [item["initial_width_state"].item() for item in batch], dtype=np.float32
    )
    initial_height_state = np.asarray(
        [item["initial_height_state"].item() for item in batch], dtype=np.float32
    )
    initial_aspect_ratio_state = np.asarray(
        [item["initial_aspect_ratio_state"].item() for item in batch], dtype=np.float32
    )

    volume = np.asarray(
        [item["volume"].item() for item in batch],
        dtype=np.float32
    )

    initial_volume = np.asarray(
        [item["initial_volume"].item() for item in batch],
        dtype=np.float32
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------
    trajectory_id = np.asarray(
        [item["trajectory_id"] for item in batch],
        dtype=np.int64
    )

    time_s = np.asarray(
        [item["time_s"] for item in batch],
        dtype=np.float32
    )

    # --------------------------------------------------------------
    # Return tensors
    # --------------------------------------------------------------
    return {
        "trajectory_id": torch.from_numpy(
            trajectory_id
        ),

        "time_s": torch.from_numpy(
            time_s
        ),

        "x": torch.from_numpy(
            x
        ),

        "y": torch.from_numpy(
            y
        ),

        "h": torch.from_numpy(
            h
        ),

        "rho": torch.from_numpy(
            rho
        ),

        "mu": torch.from_numpy(
            mu
        ),

        "g": torch.from_numpy(
            g
        ),

        "xc": torch.from_numpy(
            xc
        ),

        "yc": torch.from_numpy(
            yc
        ),

        "area": torch.from_numpy(
            area
        ),

        "width": torch.from_numpy(
            width
        ),

        "height": torch.from_numpy(
            height
        ),

        "aspect_ratio": torch.from_numpy(
            aspect_ratio
        ),

        "initial_xc": torch.from_numpy(initial_xc),
        "initial_yc": torch.from_numpy(initial_yc),
        "initial_sigma_x": torch.from_numpy(initial_sigma_x),
        "initial_sigma_y": torch.from_numpy(initial_sigma_y),
        "initial_amplitude": torch.from_numpy(initial_amplitude),

        "volume": torch.from_numpy(
            volume
        ),

        "initial_volume": torch.from_numpy(
            initial_volume
        ),

        "initial_xc_state": torch.from_numpy(initial_xc_state),
        "initial_yc_state": torch.from_numpy(initial_yc_state),
        "initial_area_state": torch.from_numpy(initial_area_state),
        "initial_width_state": torch.from_numpy(initial_width_state),
        "initial_height_state": torch.from_numpy(initial_height_state),
        "initial_aspect_ratio_state": torch.from_numpy(initial_aspect_ratio_state),
    }


# =============================================================================
# MODEL
# =============================================================================

class GeometryIntegratedSpillPINN(
    nn.Module
):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    ):

        super().__init__()

        layers = []

        layers.append(
            nn.Linear(
                input_dim,
                hidden_dim,
            )
        )

        layers.append(
            nn.Tanh()
        )

        for _ in range(
            num_hidden_layers - 1
        ):

            layers.append(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.Tanh()
            )

        self.network = nn.Sequential(
            *layers
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            1,
        )

        # FIX (2): beta lowered from 5.0 to OUTPUT_SOFTPLUS_BETA (1.0).
        self.output_activation = nn.Softplus(
            beta=OUTPUT_SOFTPLUS_BETA
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        # ---------------------------------------------------------------------
        # Initialize near target mean.
        #
        # FIX (2): re-derived for the un-rescaled activation
        # normalized_h = softplus(beta * raw) / beta, beta = OUTPUT_SOFTPLUS_BETA.
        # Solve for raw such that softplus(beta * raw) / beta = normalized_target:
        #     raw = log(expm1(beta * normalized_target)) / beta
        # (Previously this was solved for beta=5 while the forward pass also
        # divided by a fixed 5.0; that extra division has been removed below,
        # so the two must be re-derived together for beta=1.)
        # ---------------------------------------------------------------------

        target_mean = 8.2e-5

        normalized_target = (
            target_mean
            / H_REF
        )

        beta = OUTPUT_SOFTPLUS_BETA

        raw_value = (
            np.log(
                np.expm1(
                    beta
                    * normalized_target
                )
            )
            / beta
        )

        with torch.no_grad():

            self.output_layer.bias.fill_(
                float(raw_value)
            )

    def forward(self, inputs):

        features = self.network(
            inputs
        )

        raw = self.output_layer(
            features
        )

        # FIX (2): the old code divided by a fixed 5.0 here, which was
        # only correct paired with beta=5. Now that beta is a named
        # constant, normalized_h = softplus(beta * raw) / beta directly
        # (nn.Softplus already applies beta internally), so no separate
        # rescale is needed.
        normalized_h = (
            self.output_activation(raw)
            / OUTPUT_SOFTPLUS_BETA
        )

        h = (
            H_REF
            * normalized_h
        )

        return h


# =============================================================================
# SUPERVISED FIELD LOSS
# =============================================================================

def supervised_field_loss(
    prediction,
    target,
    foreground,
):

    squared_error = (
        prediction - target
    ).pow(2)

    weights = torch.where(
        foreground > 0.5,

        torch.full_like(
            foreground,
            FOREGROUND_WEIGHT,
        ),

        torch.ones_like(
            foreground
        ),
    )

    normalized_error = (
        squared_error
        / (H_REF ** 2)
    )

    return (
        normalized_error[:, 0]
        * weights
    ).mean()


# =============================================================================
# GEOMETRY LOSS
# =============================================================================

def geometry_loss(
    model,
    state_batch,
):

    x = state_batch["x"].to(
        DEVICE
    )

    y = state_batch["y"].to(
        DEVICE
    )

    h_target = state_batch["h"].to(
        DEVICE
    )

    batch_size = x.shape[0]

    # -------------------------------------------------------------------------
    # Geometry conditioning variables.
    # These are constant over each spatial field.
    # -------------------------------------------------------------------------

    time = (
        state_batch["time_s"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    xc = (
        state_batch["xc"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    yc = (
        state_batch["yc"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    area = (
        state_batch["area"]
        .to(DEVICE)
        .reshape(batch_size, 1)
        / AREA_REF
    )

    width = (
        state_batch["width"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    height = (
        state_batch["height"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    aspect_ratio = (
        state_batch["aspect_ratio"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    # -------------------------------------------------------------------------
    # rho and mu are recovered from the merged geometry-state dataframe
    # constructed in load_and_merge_data(), which attaches per-trajectory
    # rho/mu onto the complete-state rows before SpillStateDataset is built.
    # -------------------------------------------------------------------------

    rho = state_batch["rho"].to(
        DEVICE
    ).reshape(batch_size, 1)

    mu = state_batch["mu"].to(
        DEVICE
    ).reshape(batch_size, 1)

    # -------------------------------------------------------------------------
    # Flatten spatial dimensions.
    # -------------------------------------------------------------------------

    x_flat = x.reshape(
        batch_size * GRID_NX * GRID_NY,
        1,
    )

    y_flat = y.reshape(
        batch_size * GRID_NX * GRID_NY,
        1,
    )

    # Repeat state variables for every grid cell.
    time_flat = (
        time
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
        / T_REF
    )

    rho_flat = (
        rho
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
        / RHO_REF
    )

    mu_flat = (
        mu
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
        / MU_REF
    )

    xc_flat = (
        xc
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    yc_flat = (
        yc
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    area_flat = (
        area
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    width_flat = (
        width
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    height_flat = (
        height
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    aspect_flat = (
        aspect_ratio
        .repeat(
            1,
            GRID_NX * GRID_NY,
        )
        .reshape(-1, 1)
    )

    initial_xc_flat = state_batch["initial_xc"].to(DEVICE).reshape(batch_size, 1).repeat(1, GRID_NX * GRID_NY).reshape(-1, 1) / L_REF
    initial_yc_flat = state_batch["initial_yc"].to(DEVICE).reshape(batch_size, 1).repeat(1, GRID_NX * GRID_NY).reshape(-1, 1) / L_REF
    initial_sigma_x_flat = state_batch["initial_sigma_x"].to(DEVICE).reshape(batch_size, 1).repeat(1, GRID_NX * GRID_NY).reshape(-1, 1) / L_REF
    initial_sigma_y_flat = state_batch["initial_sigma_y"].to(DEVICE).reshape(batch_size, 1).repeat(1, GRID_NX * GRID_NY).reshape(-1, 1) / L_REF
    initial_amplitude_flat = state_batch["initial_amplitude"].to(DEVICE).reshape(batch_size, 1).repeat(1, GRID_NX * GRID_NY).reshape(-1, 1) / H_REF

    model_inputs = torch.cat(
        [
            x_flat,
            y_flat,
            time_flat,
            rho_flat,
            mu_flat,
            xc_flat,
            yc_flat,
            area_flat,
            width_flat,
            height_flat,
            aspect_flat,
            initial_xc_flat,
            initial_yc_flat,
            initial_sigma_x_flat,
            initial_sigma_y_flat,
            initial_amplitude_flat,
        ],
        dim=1,
    )

    prediction = model(
        model_inputs
    )

    prediction = prediction.reshape(
        batch_size,
        GRID_NX * GRID_NY,
    )

    # -------------------------------------------------------------------------
    # Differentiable visible mask.
    #
    # FIX (1): VISIBLE_TEMPERATURE now scales with H_REF instead of a fixed
    # 1e-5, so this sigmoid has a usable gradient over a realistic band of
    # thickness values instead of being saturated almost everywhere.
    # -------------------------------------------------------------------------

    visible = torch.sigmoid(
        (
            prediction
            - VISIBLE_THRESHOLD
        )
        / VISIBLE_TEMPERATURE
    )

    # -------------------------------------------------------------------------
    # Predicted physical area.
    #
    # cell_area is physical m2 because domain = 1m x 1m.
    # -------------------------------------------------------------------------

    area_pred_physical = (
        visible.sum(dim=1)
        * CELL_AREA
    )

    area_pred_normalized = (
        area_pred_physical
        / AREA_REF
    )

    area_target_normalized = (
        state_batch["area"]
        .to(DEVICE)
        / AREA_REF
    )

    area_error = (
        area_pred_normalized
        - area_target_normalized
    )

    area_loss = (
        area_error.pow(2)
    ).mean()

    # -------------------------------------------------------------------------
    # Predicted centroid.
    # -------------------------------------------------------------------------

    x_grid = x_flat.reshape(
        batch_size,
        GRID_NX * GRID_NY,
    )

    y_grid = y_flat.reshape(
        batch_size,
        GRID_NX * GRID_NY,
    )

    visible_mass = (
        visible.sum(dim=1)
        + 1e-12
    )

    xc_pred = (
        (
            visible
            * x_grid
        ).sum(dim=1)
        / visible_mass
    )

    yc_pred = (
        (
            visible
            * y_grid
        ).sum(dim=1)
        / visible_mass
    )

    xc_target = (
        state_batch["xc"]
        .to(DEVICE)
    )

    yc_target = (
        state_batch["yc"]
        .to(DEVICE)
    )

    centroid_loss = (
        (
            xc_pred
            - xc_target
        ).pow(2)
        +
        (
            yc_pred
            - yc_target
        ).pow(2)
    ).mean()

    # -------------------------------------------------------------------------
    # Combined geometry consistency.
    #
    # Area and centroid are dimensionless/normalized quantities of comparable
    # order after the reference scaling.
    # -------------------------------------------------------------------------

    geometry = (
        area_loss
        + centroid_loss
    )

    return geometry


# =============================================================================
# MASS LOSS
# =============================================================================

def mass_loss(
    model,
    state_batch,
):

    x = state_batch["x"].to(
        DEVICE
    )

    y = state_batch["y"].to(
        DEVICE
    )

    batch_size = x.shape[0]

    time = (
        state_batch["time_s"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    rho = (
        state_batch["rho"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    mu = (
        state_batch["mu"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    xc = (
        state_batch["xc"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    yc = (
        state_batch["yc"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    area = (
        state_batch["area"]
        .to(DEVICE)
        .reshape(batch_size, 1)
        / AREA_REF
    )

    width = (
        state_batch["width"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    height = (
        state_batch["height"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    aspect_ratio = (
        state_batch["aspect_ratio"]
        .to(DEVICE)
        .reshape(batch_size, 1)
    )

    x_flat = x.reshape(
        batch_size * GRID_NX * GRID_NY,
        1,
    )

    y_flat = y.reshape(
        batch_size * GRID_NX * GRID_NY,
        1,
    )

    cells = (
        GRID_NX * GRID_NY
    )

    time_flat = (
        time
        .repeat(1, cells)
        .reshape(-1, 1)
        / T_REF
    )

    rho_flat = (
        rho
        .repeat(1, cells)
        .reshape(-1, 1)
        / RHO_REF
    )

    mu_flat = (
        mu
        .repeat(1, cells)
        .reshape(-1, 1)
        / MU_REF
    )

    xc_flat = (
        xc
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    yc_flat = (
        yc
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    area_flat = (
        area
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    width_flat = (
        width
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    height_flat = (
        height
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    aspect_flat = (
        aspect_ratio
        .repeat(1, cells)
        .reshape(-1, 1)
    )

    initial_xc_flat = state_batch["initial_xc"].to(DEVICE).reshape(batch_size, 1).repeat(1, cells).reshape(-1, 1) / L_REF
    initial_yc_flat = state_batch["initial_yc"].to(DEVICE).reshape(batch_size, 1).repeat(1, cells).reshape(-1, 1) / L_REF
    initial_sigma_x_flat = state_batch["initial_sigma_x"].to(DEVICE).reshape(batch_size, 1).repeat(1, cells).reshape(-1, 1) / L_REF
    initial_sigma_y_flat = state_batch["initial_sigma_y"].to(DEVICE).reshape(batch_size, 1).repeat(1, cells).reshape(-1, 1) / L_REF
    initial_amplitude_flat = state_batch["initial_amplitude"].to(DEVICE).reshape(batch_size, 1).repeat(1, cells).reshape(-1, 1) / H_REF

    model_inputs = torch.cat(
        [
            x_flat,
            y_flat,
            time_flat,
            rho_flat,
            mu_flat,
            xc_flat,
            yc_flat,
            area_flat,
            width_flat,
            height_flat,
            aspect_flat,
            initial_xc_flat,
            initial_yc_flat,
            initial_sigma_x_flat,
            initial_sigma_y_flat,
            initial_amplitude_flat,
        ],
        dim=1,
    )

    prediction = model(
        model_inputs
    )

    prediction = prediction.reshape(
        batch_size,
        cells,
    )

    # -------------------------------------------------------------------------
    # Integrate thickness over the complete spatial domain.
    #
    # V = integral h dA
    #
    # Since the domain is 1m x 1m and the grid is uniform:
    #
    # V ~= sum(h_i) * cell_area
    # -------------------------------------------------------------------------

    predicted_volume = (
        prediction.sum(dim=1)
        * CELL_AREA
    )

    target_volume = (
        state_batch["h"]
        .to(DEVICE)
        .sum(dim=1)
        * CELL_AREA
    )

    # Normalize volume by the maximum target volume scale present in the
    # generated dataset. The value is computed directly from the batch target
    # to avoid introducing another arbitrary physical constant.
    volume_scale = (
        target_volume.detach().abs()
        + 1e-8
    )

    relative_volume_error = (
        predicted_volume
        - target_volume
    ) / volume_scale

    return (
        relative_volume_error
        .pow(2)
        .mean()
    )


# =============================================================================
# TRAJECTORY-LEVEL CONSERVATION LOSS
# =============================================================================

def conservation_loss(
    model,
    state_batch,
):
    """Penalize trajectory-level self-inconsistency of predicted volume.

    Mass loss compares predicted volume at the sampled state with the target
    volume. This term is independent of that target: it compares the model's
    predicted volume at time t with the model's predicted volume at the same
    trajectory's initial state (t=0).
    """

    x = state_batch["x"].to(DEVICE)
    y = state_batch["y"].to(DEVICE)
    batch_size = x.shape[0]
    cells = GRID_NX * GRID_NY

    def repeat_scaled_tensor(value, scale=1.0):
        value = value.to(DEVICE).reshape(batch_size, 1)
        return value.repeat(1, cells).reshape(-1, 1) / scale

    def predict_volume(
        time_values,
        xc_values,
        yc_values,
        area_values,
        width_values,
        height_values,
        aspect_ratio_values,
    ):
        x_flat = x.reshape(batch_size * cells, 1)
        y_flat = y.reshape(batch_size * cells, 1)

        model_inputs = torch.cat(
            [
                x_flat,
                y_flat,
                repeat_scaled_tensor(time_values, T_REF),
                repeat_scaled_tensor(state_batch["rho"], RHO_REF),
                repeat_scaled_tensor(state_batch["mu"], MU_REF),
                repeat_scaled_tensor(xc_values),
                repeat_scaled_tensor(yc_values),
                repeat_scaled_tensor(area_values, AREA_REF),
                repeat_scaled_tensor(width_values),
                repeat_scaled_tensor(height_values),
                repeat_scaled_tensor(aspect_ratio_values),
                repeat_scaled_tensor(state_batch["initial_xc"], L_REF),
                repeat_scaled_tensor(state_batch["initial_yc"], L_REF),
                repeat_scaled_tensor(state_batch["initial_sigma_x"], L_REF),
                repeat_scaled_tensor(state_batch["initial_sigma_y"], L_REF),
                repeat_scaled_tensor(state_batch["initial_amplitude"], H_REF),
            ],
            dim=1,
        )

        prediction = model(model_inputs).reshape(batch_size, cells)
        return prediction.sum(dim=1) * CELL_AREA

    predicted_volume_t = predict_volume(
        state_batch["time_s"],
        state_batch["xc"],
        state_batch["yc"],
        state_batch["area"],
        state_batch["width"],
        state_batch["height"],
        state_batch["aspect_ratio"],
    )

    initial_time = torch.zeros_like(
        state_batch["time_s"].to(DEVICE)
    )
    predicted_volume_0 = predict_volume(
        initial_time,
        state_batch["initial_xc_state"],
        state_batch["initial_yc_state"],
        state_batch["initial_area_state"],
        state_batch["initial_width_state"],
        state_batch["initial_height_state"],
        state_batch["initial_aspect_ratio_state"],
    )

    relative_drift = (
        predicted_volume_t - predicted_volume_0
    ) / (predicted_volume_0.detach().abs() + 1e-8)

    return relative_drift.pow(2).mean()


# =============================================================================
# LOAD AND MERGE DATA
# =============================================================================

def load_and_merge_data():

    print()
    print("Loading field dataset...")

    field_df = pd.read_csv(
        FIELD_FILE
    )

    print(
        f"Field rows: {len(field_df):,}"
    )

    print(
        f"Field columns: {list(field_df.columns)}"
    )

    print()
    print("Loading geometry-state dataset...")

    state_df = pd.read_csv(
        STATE_FILE
    )

    print(
        f"State rows: {len(state_df):,}"
    )

    print(
        f"State columns: {list(state_df.columns)}"
    )

    merge_keys = [
        "trajectory_id",
        "time_s",
    ]

    duplicate_states = (
        state_df
        .duplicated(
            subset=merge_keys
        )
        .sum()
    )

    if duplicate_states != 0:

        raise ValueError(
            "Geometry-state dataset contains "
            f"{duplicate_states} duplicate states."
        )

    geometry_columns = [
        "xc",
        "yc",
        "area",
        "width",
        "height",
        "aspect_ratio",
    ]

    state_subset = state_df[
        merge_keys
        + geometry_columns
    ].copy()

    merged = field_df.merge(
        state_subset,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    if len(merged) != len(
        field_df
    ):

        raise ValueError(
            "Field row count changed during merge."
        )

    if merged[
        geometry_columns
    ].isna().any().any():

        raise ValueError(
            "Geometry merge contains missing values."
        )

    # -------------------------------------------------------------------------
    # Add rho and mu to the state dataset by taking their unique values from
    # each trajectory. They are physical trajectory parameters and constant
    # throughout a trajectory.
    # -------------------------------------------------------------------------

    trajectory_properties = (
        field_df
        .groupby(
            "trajectory_id"
        )
        .agg(
            rho=("rho", "first"),
            mu=("mu", "first"),
        )
        .reset_index()
    )

    # Verify rho and mu are constant within each trajectory.
    rho_counts = (
        field_df
        .groupby("trajectory_id")["rho"]
        .nunique()
    )

    mu_counts = (
        field_df
        .groupby("trajectory_id")["mu"]
        .nunique()
    )

    if (
        rho_counts.max() != 1
        or mu_counts.max() != 1
    ):

        raise ValueError(
            "rho or mu is not constant within at least one trajectory."
        )

    merged = merged.merge(
        trajectory_properties,
        on="trajectory_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_trajectory"),
    )

    # -------------------------------------------------------------------------
    # Construct state dataframe from one representative row per state plus
    # full field information.
    # -------------------------------------------------------------------------

    state_df_complete = (
        merged[
            [
                "trajectory_id",
                "time_s",
                "x",
                "y",
                "h",
                "xc",
                "yc",
                "area",
                "width",
                "height",
                "aspect_ratio",
                "initial_xc",
                "initial_yc",
                "initial_sigma_x",
                "initial_sigma_y",
                "initial_amplitude",
                "rho_trajectory",
                "mu_trajectory",
            ]
        ]
        .rename(
            columns={
                "rho_trajectory": "rho",
                "mu_trajectory": "mu",
            }
        )
    )

    print()
    print(
        "Geometry state successfully merged onto field data."
    )

    print(
        f"Merged field rows: {len(merged):,}"
    )

    print(
        f"Merged columns: {list(merged.columns)}"
    )

    return (
        merged,
        state_df_complete,
    )


# =============================================================================
# SPLIT
# =============================================================================


def split_by_trajectory(
    field_df,
    state_df,
    train_ids,
    val_ids,
    test_ids,
):
    """
    Split field and geometry-state data by trajectory.

    IMPORTANT:
    The field dataframe contains 1600 grid cells for every
    trajectory/time state. Therefore, when geometry-state information
    has been merged onto the field dataframe, each geometry state is
    repeated 1600 times.

    This function explicitly:
      1. Splits field data by trajectory.
      2. Deduplicates geometry states by trajectory_id + time_s.
      3. Verifies 61 complete states per trajectory.
      4. Verifies 97,600 field rows per trajectory
         (61 times x 40 x 40 grid).
      5. Checks train/validation/test trajectory isolation.
    """

    train_ids = sorted([int(x) for x in train_ids])
    val_ids = sorted([int(x) for x in val_ids])
    test_ids = sorted([int(x) for x in test_ids])

    # ------------------------------------------------------------------
    # 1. Check trajectory partitions
    # ------------------------------------------------------------------
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:
        raise ValueError("Training and validation trajectories overlap.")

    if train_set & test_set:
        raise ValueError("Training and test trajectories overlap.")

    if val_set & test_set:
        raise ValueError("Validation and test trajectories overlap.")

    all_expected_ids = train_set | val_set | test_set

    actual_field_ids = set(
        field_df["trajectory_id"].astype(int).unique()
    )

    actual_state_ids = set(
        state_df["trajectory_id"].astype(int).unique()
    )

    if actual_field_ids != all_expected_ids:
        raise ValueError(
            "Field trajectory IDs do not match the expected "
            "train/validation/test IDs."
        )

    if actual_state_ids != all_expected_ids:
        raise ValueError(
            "State trajectory IDs do not match the expected "
            "train/validation/test IDs."
        )

    # ------------------------------------------------------------------
    # 2. Split FIELD dataframe
    # ------------------------------------------------------------------
    field_train = field_df[
        field_df["trajectory_id"].isin(train_ids)
    ].copy()

    field_val = field_df[
        field_df["trajectory_id"].isin(val_ids)
    ].copy()

    field_test = field_df[
        field_df["trajectory_id"].isin(test_ids)
    ].copy()

    # ------------------------------------------------------------------
    # 3. Remove repeated geometry-state rows
    # ------------------------------------------------------------------
    #
    # If state_df came from the merged field dataframe, every
    # trajectory/time state appears once per grid cell.
    #
    # There should be exactly ONE geometry-state record for each:
    #
    #     trajectory_id + time_s
    #
    # Therefore explicitly deduplicate here.
    #
    state_unique = (
        state_df
        .sort_values(["trajectory_id", "time_s"])
        .drop_duplicates(
            subset=["trajectory_id", "time_s"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 4. Split GEOMETRY-STATE dataframe
    # ------------------------------------------------------------------
    state_train = state_unique[
        state_unique["trajectory_id"].isin(train_ids)
    ].copy()

    state_val = state_unique[
        state_unique["trajectory_id"].isin(val_ids)
    ].copy()

    state_test = state_unique[
        state_unique["trajectory_id"].isin(test_ids)
    ].copy()

    # ------------------------------------------------------------------
    # 5. Expected dimensions
    # ------------------------------------------------------------------
    expected_times = 61
    expected_grid_cells = GRID_NX * GRID_NY
    expected_field_rows_per_trajectory = (
        expected_times * expected_grid_cells
    )

    # ------------------------------------------------------------------
    # 6. Verify FIELD rows per trajectory
    # ------------------------------------------------------------------
    field_counts = (
        field_df
        .groupby("trajectory_id")
        .size()
        .sort_index()
    )

    bad_field = field_counts[
        field_counts != expected_field_rows_per_trajectory
    ]

    if len(bad_field) > 0:
        raise ValueError(
            "Trajectories have unexpected numbers of field rows:\n"
            f"{bad_field}"
        )

    # ------------------------------------------------------------------
    # 7. Verify GEOMETRY states per trajectory
    # ------------------------------------------------------------------
    state_counts = (
        state_unique
        .groupby("trajectory_id")
        .size()
        .sort_index()
    )

    bad_state = state_counts[
        state_counts != expected_times
    ]

    if len(bad_state) > 0:
        raise ValueError(
            "Trajectories have unexpected numbers of complete states:\n"
            f"{bad_state}"
        )

    # ------------------------------------------------------------------
    # 8. Verify no duplicate trajectory/time states
    # ------------------------------------------------------------------
    duplicate_states = (
        state_unique
        .duplicated(
            subset=["trajectory_id", "time_s"],
            keep=False
        )
        .sum()
    )

    if duplicate_states != 0:
        raise ValueError(
            f"Found {duplicate_states} duplicate trajectory/time "
            "geometry states after deduplication."
        )

    # ------------------------------------------------------------------
    # 9. Verify total field rows
    # ------------------------------------------------------------------
    expected_train_field = (
        len(train_ids) * expected_field_rows_per_trajectory
    )
    expected_val_field = (
        len(val_ids) * expected_field_rows_per_trajectory
    )
    expected_test_field = (
        len(test_ids) * expected_field_rows_per_trajectory
    )

    if len(field_train) != expected_train_field:
        raise ValueError(
            f"Unexpected training field count: {len(field_train)} "
            f"(expected {expected_train_field})"
        )

    if len(field_val) != expected_val_field:
        raise ValueError(
            f"Unexpected validation field count: {len(field_val)} "
            f"(expected {expected_val_field})"
        )

    if len(field_test) != expected_test_field:
        raise ValueError(
            f"Unexpected test field count: {len(field_test)} "
            f"(expected {expected_test_field})"
        )

    # ------------------------------------------------------------------
    # 10. Verify total geometry-state counts
    # ------------------------------------------------------------------
    expected_train_states = len(train_ids) * expected_times
    expected_val_states = len(val_ids) * expected_times
    expected_test_states = len(test_ids) * expected_times

    if len(state_train) != expected_train_states:
        raise ValueError(
            f"Unexpected training state count: {len(state_train)} "
            f"(expected {expected_train_states})"
        )

    if len(state_val) != expected_val_states:
        raise ValueError(
            f"Unexpected validation state count: {len(state_val)} "
            f"(expected {expected_val_states})"
        )

    if len(state_test) != expected_test_states:
        raise ValueError(
            f"Unexpected test state count: {len(state_test)} "
            f"(expected {expected_test_states})"
        )

    # ------------------------------------------------------------------
    # 11. Verify trajectory isolation
    # ------------------------------------------------------------------
    if (
        set(field_train["trajectory_id"].unique())
        & set(field_val["trajectory_id"].unique())
    ):
        raise ValueError("Field train/validation trajectory leakage.")

    if (
        set(field_train["trajectory_id"].unique())
        & set(field_test["trajectory_id"].unique())
    ):
        raise ValueError("Field train/test trajectory leakage.")

    if (
        set(field_val["trajectory_id"].unique())
        & set(field_test["trajectory_id"].unique())
    ):
        raise ValueError("Field validation/test trajectory leakage.")

    print("\n" + "=" * 72)
    print("TRAJECTORY SPLIT INTEGRITY CHECK")
    print("=" * 72)

    print(
        f"Training trajectories:   {train_ids}"
    )
    print(
        f"Validation trajectories: {val_ids}"
    )
    print(
        f"Test trajectories:       {test_ids}"
    )

    print("\nField rows:")
    print(
        f"  Train: {len(field_train):,} "
        f"(expected {expected_train_field:,})"
    )
    print(
        f"  Val:   {len(field_val):,} "
        f"(expected {expected_val_field:,})"
    )
    print(
        f"  Test:  {len(field_test):,} "
        f"(expected {expected_test_field:,})"
    )

    print("\nUnique geometry states:")
    print(
        f"  Train: {len(state_train):,} "
        f"(expected {expected_train_states:,})"
    )
    print(
        f"  Val:   {len(state_val):,} "
        f"(expected {expected_val_states:,})"
    )
    print(
        f"  Test:  {len(state_test):,} "
        f"(expected {expected_test_states:,})"
    )

    print("\nPer-trajectory integrity:")
    print(
        f"  Field rows/trajectory: "
        f"{expected_field_rows_per_trajectory:,}"
    )
    print(
        f"  Geometry states/trajectory: "
        f"{expected_times}"
    )

    print("\nPASS: no trajectory leakage.")
    print("PASS: field dimensions are correct.")
    print("PASS: geometry-state dimensions are correct.")
    print("PASS: no duplicate trajectory/time states.")
    print("=" * 72)

    return (
        field_train,
        field_val,
        field_test,
        state_train,
        state_val,
        state_test,
    )

# =============================================================================
# FIELD EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_field(
    model,
    loader,
):

    model.eval()

    squared_sum = 0.0
    absolute_sum = 0.0
    count = 0

    for (
        inputs,
        target,
        _,
    ) in loader:

        inputs = inputs.to(
            DEVICE,
            non_blocking=True,
        )

        target = target.to(
            DEVICE,
            non_blocking=True,
        )

        prediction = model(
            inputs
        )

        error = (
            prediction
            - target
        )

        squared_sum += (
            error.pow(2)
            .sum()
            .item()
        )

        absolute_sum += (
            error.abs()
            .sum()
            .item()
        )

        count += target.numel()

    mse = (
        squared_sum
        / count
    )

    rmse = np.sqrt(mse)

    mae = (
        absolute_sum
        / count
    )

    return rmse, mae


# =============================================================================
# STATE EVALUATION
# =============================================================================

def evaluate_constraints(
    model,
    state_loader,
):

    model.eval()

    geometry_values = []
    mass_values = []
    conservation_values = []

    with torch.no_grad():

        for state_batch in state_loader:

            geometry_values.append(
                geometry_loss(
                    model,
                    state_batch,
                ).item()
            )

            mass_values.append(
                mass_loss(
                    model,
                    state_batch,
                ).item()
            )

            conservation_values.append(
                conservation_loss(
                    model,
                    state_batch,
                ).item()
            )

    return (
        float(np.mean(geometry_values)),
        float(np.mean(mass_values)),
        float(np.mean(conservation_values)),
    )


# =============================================================================
# INITIAL DIAGNOSTIC
# =============================================================================

@torch.no_grad()
def initial_diagnostic(
    model,
    dataset,
):

    model.eval()

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(
            10000,
            len(dataset),
        ),
        dtype=int,
    )

    inputs = torch.from_numpy(
        dataset.inputs[
            indices
        ]
    ).to(DEVICE)

    target = (
        dataset.target[
            indices
        ]
        .reshape(-1)
    )

    prediction = (
        model(inputs)
        .cpu()
        .numpy()
        .reshape(-1)
    )

    print()
    print(
        "Initial prediction diagnostic"
    )

    print(
        f"Initial h min:  "
        f"{prediction.min():.6e} m"
    )

    print(
        f"Initial h max:  "
        f"{prediction.max():.6e} m"
    )

    print(
        f"Initial h mean: "
        f"{prediction.mean():.6e} m"
    )

    print(
        f"Target h min:   "
        f"{target.min():.6e} m"
    )

    print(
        f"Target h max:   "
        f"{target.max():.6e} m"
    )

    print(
        f"Target h mean:  "
        f"{target.mean():.6e} m"
    )


# =============================================================================
# TRAINING
# =============================================================================

def train(conservation_weight):

    print("=" * 72)

    print(
        "STAGE 2B — IC-CONDITIONED + TRAJECTORY-CONSERVATION SPILL PINN"
    )

    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Conservation weight (final): {conservation_weight:.3f}"
    )

    print(
        f"Conservation warmup epochs: {CONSERVATION_WARMUP_EPOCHS}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()
    print(
        "Loss weights (final, after warmup, RELATIVE to data-loss scale "
        "-- see 'Computed loss-scale normalization factors' below):"
    )

    print(
        f"  Data:     {DATA_WEIGHT}"
    )

    print(
        f"  Geometry: {GEOMETRY_WEIGHT} "
        f"(warmup over {GEOMETRY_WARMUP_EPOCHS} epochs)"
    )

    print(
        f"  Mass:     {MASS_WEIGHT} "
        f"(warmup over {MASS_WARMUP_EPOCHS} epochs)"
    )

    print(
        f"  Physics:  {PHYSICS_WEIGHT}"
    )

    print(
        f"  Grad clip norm: {GRAD_CLIP_NORM}"
    )

    print()
    print(
        f"Visible temperature: {VISIBLE_TEMPERATURE:.3e} "
        f"(tied to H_REF = {H_REF})"
    )

    print(
        f"Output softplus beta: {OUTPUT_SOFTPLUS_BETA}"
    )

    train_ids = list(range(0, 21))
    val_ids = list(range(21, 25))
    test_ids = list(range(25, 30))
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    (
        merged_field,
        complete_state_df,
    ) = load_and_merge_data()


    (
        train_field,
        val_field,
        test_field,
        train_state,
        val_state,
        test_state,
    ) = split_by_trajectory(
        merged_field,
        complete_state_df,
        train_ids,
        val_ids,
        test_ids,
    )


    # -------------------------------------------------------------------------
    # Field datasets
    # -------------------------------------------------------------------------

    train_field_dataset = (
        SpillFieldDataset(
            train_field
        )
    )

    val_field_dataset = (
        SpillFieldDataset(
            val_field
        )
    )

    test_field_dataset = (
        SpillFieldDataset(
            test_field
        )
    )

    # FIX (5): print the foreground (visible-spill) pixel fraction so the
    # FOREGROUND_WEIGHT choice can be sanity-checked against the actual
    # class balance instead of assumed.
    print()
    print(
        "Foreground pixel fraction "
        f"(train): {train_field_dataset.foreground.mean():.4%}"
    )
    print(
        "Foreground pixel fraction "
        f"(val):   {val_field_dataset.foreground.mean():.4%}"
    )
    print(
        "Foreground pixel fraction "
        f"(test):  {test_field_dataset.foreground.mean():.4%}"
    )
    print(
        f"Current FOREGROUND_WEIGHT: {FOREGROUND_WEIGHT} "
        "-- if the foreground fraction above is well under ~10%, consider "
        "raising this weight further."
    )

    train_field_loader = DataLoader(
        train_field_dataset,
        batch_size=FIELD_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_field_loader = DataLoader(
        val_field_dataset,
        batch_size=FIELD_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_field_loader = DataLoader(
        test_field_dataset,
        batch_size=FIELD_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------------------
    # Complete-state datasets/loaders
    # -------------------------------------------------------------------------


    train_state_dataset = SpillStateDataset(
        train_field
    )

    val_state_dataset = SpillStateDataset(
        val_field
    )

    test_state_dataset = SpillStateDataset(
        test_field
    )


    train_state_loader = DataLoader(
        train_state_dataset,
        batch_size=STATE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=state_collate,
    )

    val_state_loader = DataLoader(
        val_state_dataset,
        batch_size=STATE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=state_collate,
    )

    test_state_loader = DataLoader(
        test_state_dataset,
        batch_size=STATE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=state_collate,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = (
        GeometryIntegratedSpillPINN()
        .to(DEVICE)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable model parameters: "
        f"{parameter_count:,}"
    )

    initial_diagnostic(
        model,
        train_field_dataset,
    )

    # -------------------------------------------------------------------------
    # Optimizer + scheduler
    # -------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # FIX (4): decay LR once validation RMSE plateaus, since a fixed LR
    # over 300 epochs is unlikely to be optimal throughout training.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    history = []

    best_val_rmse = float(
        "inf"
    )

    best_epoch = None

    best_model_path = (
        MODELS_DIR
        / "spill_pinn_stage2_ic_conservation_best.pt"
    )

    # -------------------------------------------------------------------------
    # FIX (6): loss-scale normalization state.
    #
    # Computed once, from the very first training batch's raw (unweighted)
    # losses, then held fixed for the rest of training. This converts
    # GEOMETRY_WEIGHT / MASS_WEIGHT from "raw weight against whatever
    # magnitude geometry_loss/mass_loss happen to have" into "weight
    # relative to data_loss's own magnitude", which is what actually
    # prevents geometry (raw ~0.49) from silently outweighing data
    # (raw ~0.02) by ~20-25x once its warmup weight ramps up.
    # -------------------------------------------------------------------------

    loss_scale_geometry = None
    loss_scale_mass = None
    loss_scale_conservation = None

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        # FIX (3): current epoch's warmed-up geometry/mass weights.
        current_geometry_weight = geometry_weight_schedule(epoch)
        current_mass_weight = mass_weight_schedule(epoch)
        current_conservation_weight = conservation_weight_schedule(epoch, conservation_weight)

        # ==============================================================
        # FIELD ITERATOR
        # ==============================================================

        field_iterator = iter(
            train_field_loader
        )

        state_iterator = iter(
            train_state_loader
        )

        num_field_batches = len(
            train_field_loader
        )

        num_state_batches = len(
            train_state_loader
        )

        total_batches = max(
            num_field_batches,
            num_state_batches,
        )

        epoch_total = 0.0
        epoch_data = 0.0
        epoch_geometry = 0.0
        epoch_mass = 0.0
        epoch_conservation = 0.0

        for batch_index in range(
            total_batches
        ):

            # ----------------------------------------------------------
            # Cycle through field loader.
            # ----------------------------------------------------------

            try:

                field_batch = next(
                    field_iterator
                )

            except StopIteration:

                field_iterator = iter(
                    train_field_loader
                )

                field_batch = next(
                    field_iterator
                )

            inputs = field_batch[
                0
            ].to(
                DEVICE,
                non_blocking=True,
            )

            target = field_batch[
                1
            ].to(
                DEVICE,
                non_blocking=True,
            )

            foreground = field_batch[
                2
            ].to(
                DEVICE,
                non_blocking=True,
            )

            # ----------------------------------------------------------
            # Cycle through complete-state loader.
            # ----------------------------------------------------------

            try:

                state_batch = next(
                    state_iterator
                )

            except StopIteration:

                state_iterator = iter(
                    train_state_loader
                )

                state_batch = next(
                    state_iterator
                )

            # ----------------------------------------------------------
            # Clear gradients.
            # ----------------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ==========================================================
            # DATA LOSS
            # ==========================================================

            prediction = model(
                inputs
            )

            data = (
                supervised_field_loss(
                    prediction,
                    target,
                    foreground,
                )
            )

            # ==========================================================
            # GEOMETRY LOSS
            # ==========================================================

            geometry = geometry_loss(
                model,
                state_batch,
            )

            # ==========================================================
            # MASS LOSS
            # ==========================================================

            mass = mass_loss(
                model,
                state_batch,
            )

            # ==========================================================
            # TRAJECTORY CONSERVATION LOSS
            # ==========================================================

            conservation = conservation_loss(
                model,
                state_batch,
            )

            # ==========================================================
            # FIX (6): compute the one-time loss-scale normalization
            # factors from the very first batch's raw losses, before any
            # weighting is applied. Held fixed afterward so the relative
            # weighting stays consistent and interpretable across epochs.
            # ==========================================================

            if loss_scale_geometry is None:

                loss_scale_geometry = (
                    (data.item() + 1e-8)
                    / (geometry.item() + 1e-8)
                )

                loss_scale_mass = (
                    (data.item() + 1e-8)
                    / (mass.item() + 1e-8)
                )

                loss_scale_conservation = (
                    (data.item() + 1e-8)
                    / (conservation.item() + 1e-8)
                )

                print()
                print(
                    "Computed loss-scale normalization factors "
                    "(held fixed for the rest of training):"
                )
                print(
                    f"  geometry: {loss_scale_geometry:.6e} "
                    f"(raw geometry loss: {geometry.item():.6e})"
                )
                print(
                    f"  mass:     {loss_scale_mass:.6e} "
                    f"(raw mass loss: {mass.item():.6e})"
                )
                print(
                    f"  conservation: {loss_scale_conservation:.6e} "
                    f"(raw conservation loss: {conservation.item():.6e})"
                )

            # ==========================================================
            # TOTAL LOSS
            #
            # FIX (3): geometry/mass use their warmed-up per-epoch
            # weights instead of the flat GEOMETRY_WEIGHT/MASS_WEIGHT.
            #
            # FIX (6): each term is additionally multiplied by its
            # loss_scale_* factor so current_geometry_weight/
            # current_mass_weight are relative to data_loss's own
            # magnitude, not raw multipliers on an already much larger
            # quantity.
            # ==========================================================

            total = (
                DATA_WEIGHT
                * data
                +
                current_geometry_weight
                * loss_scale_geometry
                * geometry
                +
                current_mass_weight
                * loss_scale_mass
                * mass
                +
                current_conservation_weight
                * loss_scale_conservation
                * conservation
            )

            # Physics is explicitly zero in Stage 2.
            #
            # No PDE residual is calculated.

            total.backward()

            # FIX (6): clip gradient norm to damp any sharp transition as
            # the warmup weights climb, rather than letting one bad batch
            # kick the model out of a good basin (as seen around the
            # epoch where geometry_weight crossed ~0.55-0.60 previously).
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )

            optimizer.step()

            # ----------------------------------------------------------
            # Accumulate diagnostics.
            # ----------------------------------------------------------

            epoch_total += (
                total.item()
            )

            epoch_data += (
                data.item()
            )

            epoch_geometry += (
                geometry.item()
            )

            epoch_mass += (
                mass.item()
            )

            epoch_conservation += (
                conservation.item()
            )

        # -----------------------------------------------------------------
        # Mean training losses.
        # -----------------------------------------------------------------

        epoch_total /= total_batches
        epoch_data /= total_batches
        epoch_geometry /= total_batches
        epoch_mass /= total_batches

        # -----------------------------------------------------------------
        # Validation field metrics.
        # -----------------------------------------------------------------

        val_rmse, val_mae = (
            evaluate_field(
                model,
                val_field_loader,
            )
        )

        # -----------------------------------------------------------------
        # Validation constraint metrics.
        # -----------------------------------------------------------------

        val_geometry, val_mass, val_conservation = (
            evaluate_constraints(
                model,
                val_state_loader,
            )
        )

        # -----------------------------------------------------------------
        # Step the LR scheduler on validation RMSE.
        # -----------------------------------------------------------------

        scheduler.step(val_rmse)

        # -----------------------------------------------------------------
        # Save best model according to validation RMSE.
        # -----------------------------------------------------------------

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_rmse_m":
                        val_rmse,

                    "val_mae_m":
                        val_mae,

                    "val_geometry":
                        val_geometry,

                    "val_mass":
                        val_mass,

                    "seed": SEED,

                    "loss_weights": {
                        "data":
                            DATA_WEIGHT,

                        "geometry":
                            current_geometry_weight,

                        "mass":
                            current_mass_weight,

                        "conservation":
                            current_conservation_weight,

                        "physics":
                            PHYSICS_WEIGHT,
                    },

                    "loss_scale_geometry":
                        loss_scale_geometry,

                    "loss_scale_mass":
                        loss_scale_mass,

                    "loss_scale_conservation":
                        loss_scale_conservation,
                },
                best_model_path,
            )

        history.append(
            {
                "epoch": epoch,

                "train_total":
                    epoch_total,

                "train_data":
                    epoch_data,

                "train_geometry":
                    epoch_geometry,

                "train_mass":
                    epoch_mass,

                "train_conservation":
                    epoch_conservation,

                "geometry_weight":
                    current_geometry_weight,

                "mass_weight":
                    current_mass_weight,

                "conservation_weight":
                    current_conservation_weight,

                "loss_scale_geometry":
                    loss_scale_geometry,

                "loss_scale_mass":
                    loss_scale_mass,

                "loss_scale_conservation":
                    loss_scale_conservation,

                "lr":
                    optimizer.param_groups[0]["lr"],

                "val_rmse_m":
                    val_rmse,

                "val_mae_m":
                    val_mae,

                "val_geometry":
                    val_geometry,

                "val_mass":
                    val_mass,

                "val_conservation":
                    val_conservation,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Total {epoch_total:.6e} | "
            f"Data {epoch_data:.6e} | "
            f"Geometry {epoch_geometry:.6e} (w={current_geometry_weight:.3f}) | "
            f"Mass {epoch_mass:.6e} (w={current_mass_weight:.3f}) | "
            f"Conservation {epoch_conservation:.6e} (w={current_conservation_weight:.3f}) | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Val RMSE {val_rmse:.6e} m | "
            f"Val MAE {val_mae:.6e} m | "
            f"Val Geometry {val_geometry:.6e} | "
            f"Val Mass {val_mass:.6e}"
        )

    # =========================================================================
    # RESTORE BEST MODEL
    # =========================================================================

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =========================================================================
    # FINAL TEST
    # =========================================================================

    test_rmse, test_mae = (
        evaluate_field(
            model,
            test_field_loader,
        )
    )

    test_geometry, test_mass, test_conservation = (
        evaluate_constraints(
            model,
            test_state_loader,
        )
    )

    print()
    print("=" * 72)
    print("STAGE 2 COMPLETE")
    print("=" * 72)

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation RMSE: "
        f"{best_val_rmse:.6e} m"
    )

    print()
    print(
        f"Final test RMSE: "
        f"{test_rmse:.6e} m"
    )

    print(
        f"Final test MAE: "
        f"{test_mae:.6e} m"
    )

    print(
        f"Final test geometry loss: "
        f"{test_geometry:.6e}"
    )

    print(
        f"Final test mass loss: "
        f"{test_mass:.6e}"
    )

    print(
        f"Final test conservation loss: "
        f"{test_conservation:.6e}"
    )

    # =========================================================================
    # SAVE HISTORY
    # =========================================================================

    history_df = pd.DataFrame(
        history
    )

    weight_tag = f"{conservation_weight:.3f}".replace(".", "p")
    history_file = (
        RESULTS_DIR
        / f"spill_pinn_stage2_ic_conservation_training_history_w{weight_tag}.csv"
    )

    history_df.to_csv(
        history_file,
        index=False,
    )

    # =========================================================================
    # SAVE TEST PREDICTIONS
    # =========================================================================

    model.eval()

    predictions = []

    with torch.no_grad():

        for (
            inputs,
            target,
            _,
        ) in test_field_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )

            prediction = (
                model(inputs)
                .cpu()
                .numpy()
                .reshape(-1)
            )

            target_np = (
                target.numpy()
                .reshape(-1)
            )

            predictions.extend(
                zip(
                    prediction,
                    target_np,
                )
            )

    prediction_df = pd.DataFrame(
        predictions,
        columns=[
            "h_pred_m",
            "h_true_m",
        ],
    )

    prediction_file = (
        RESULTS_DIR
        / f"spill_pinn_stage2_ic_conservation_test_predictions_w{weight_tag}.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    print()
    print(
        f"Best model saved to:\n"
        f"{best_model_path}"
    )

    print(
        f"Training history saved to:\n"
        f"{history_file}"
    )

    print(
        f"Test predictions saved to:\n"
        f"{prediction_file}"
    )

    print("=" * 72)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train the IC-conditioned Stage 2B spill PINN with an explicit "
            "trajectory-level conservation penalty."
        )
    )
    parser.add_argument(
        "--conservation-weight",
        type=float,
        default=DEFAULT_CONSERVATION_WEIGHT,
        help=(
            "Final relative conservation-loss weight. It is warmed up over "
            "CONSERVATION_WARMUP_EPOCHS."
        ),
    )
    args = parser.parse_args()

    if args.conservation_weight < 0:
        raise ValueError("--conservation-weight must be non-negative.")

    train(args.conservation_weight)