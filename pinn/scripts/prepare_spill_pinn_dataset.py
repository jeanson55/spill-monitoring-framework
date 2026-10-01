#'```python
"""
PREPARE SPILL PINN DATASETS
===========================

Converts the physics-generated spill trajectory dataset into two
PINN-ready representations:

1. FIELD DATASET
   One row per trajectory x time x spatial point.

2. GEOMETRY/STATE DATASET
   One row per trajectory x time.

The trajectory-level train/validation/test split is preserved from
spill_trajectory_metadata.csv.

No random row-level splitting is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SOURCE_PATH = DATA_DIR / "spill_trajectory_dataset.csv"
METADATA_PATH = DATA_DIR / "spill_trajectory_metadata.csv"

FIELD_PATH = DATA_DIR / "spill_pinn_field_ic_conditioned.csv"
STATE_PATH = DATA_DIR / "spill_pinn_geometry_state_ic_conditioned.csv"
SPLIT_PATH = DATA_DIR / "spill_pinn_split.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PREPARE SPILL PINN DATASETS")
print("=" * 70)

print()
print("Loading source dataset...")

df = pd.read_csv(SOURCE_PATH)
metadata = pd.read_csv(METADATA_PATH)

print(f"Source rows:        {len(df):,}")
print(f"Source columns:     {len(df.columns)}")
print(f"Trajectories:       {df['trajectory_id'].nunique()}")
print(f"Time points:        {df['time_s'].nunique()}")
print()


# ============================================================
# VERIFY TRAJECTORY SPLITS
# ============================================================

required_metadata = {
    "trajectory_id",
    "split",
    "rho",
    "mu",
}

missing = required_metadata.difference(metadata.columns)

if missing:
    raise ValueError(
        f"Metadata is missing required columns: {sorted(missing)}"
    )


source_ids = set(df["trajectory_id"].unique())
metadata_ids = set(metadata["trajectory_id"].unique())

if source_ids != metadata_ids:
    raise ValueError(
        "Trajectory IDs in source dataset and metadata do not match."
    )


split_counts = metadata["split"].value_counts()

required_splits = {
    "train",
    "validation",
    "test",
}

if set(split_counts.index) != required_splits:
    raise ValueError(
        f"Unexpected split labels: {sorted(split_counts.index)}"
    )


print("Trajectory split:")
print(split_counts.to_string())
print()


# ============================================================
# SPLIT TABLE
# ============================================================

split_df = metadata[
    [
        "trajectory_id",
        "split",
    ]
].sort_values("trajectory_id")

split_df.to_csv(
    SPLIT_PATH,
    index=False,
)


# ============================================================
# FIELD DATASET
# ============================================================

print("Preparing field dataset...")

field_columns = [
    "trajectory_id",
    "time_s",
    "x",
    "y",
    "h",
    "rho",
    "mu",
    "g",
    "initial_xc",
    "initial_yc",
    "initial_sigma_x",
    "initial_sigma_y",
    "initial_amplitude",
]

field_df = df[field_columns].copy()

# Add trajectory-level split.
field_df = field_df.merge(
    split_df,
    on="trajectory_id",
    how="left",
    validate="many_to_one",
)

# Reorder.
field_df = field_df[
    [
        "trajectory_id",
        "split",
        "time_s",
        "x",
        "y",
        "h",
        "rho",
        "mu",
        "g",
        "initial_xc",
        "initial_yc",
        "initial_sigma_x",
        "initial_sigma_y",
        "initial_amplitude",
    ]
]

field_df = field_df.sort_values(
    [
        "trajectory_id",
        "time_s",
        "x",
        "y",
    ]
).reset_index(drop=True)

field_df.to_csv(
    FIELD_PATH,
    index=False,
)

print(f"Field rows:        {len(field_df):,}")
print(f"Field columns:     {len(field_df.columns)}")
print()


# ============================================================
# GEOMETRY / STATE DATASET
# ============================================================

print("Preparing geometry/state dataset...")

state_columns = [
    "trajectory_id",
    "time_s",
    "xc",
    "yc",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "volume",
    "max_thickness",
    "rho",
    "mu",
    "g",
    "visible",
    "initial_xc",
    "initial_yc",
    "initial_sigma_x",
    "initial_sigma_y",
    "initial_amplitude",
]

state_df = (
    df[state_columns]
    .drop_duplicates(
        subset=[
            "trajectory_id",
            "time_s",
        ]
    )
    .copy()
)

# Add split.
state_df = state_df.merge(
    split_df,
    on="trajectory_id",
    how="left",
    validate="many_to_one",
)

state_df = state_df[
    [
        "trajectory_id",
        "split",
        "time_s",
        "xc",
        "yc",
        "area",
        "width",
        "height",
        "aspect_ratio",
        "volume",
        "max_thickness",
        "rho",
        "mu",
        "g",
        "visible",
        "initial_xc",
        "initial_yc",
        "initial_sigma_x",
        "initial_sigma_y",
        "initial_amplitude",
    ]
]

state_df = state_df.sort_values(
    [
        "trajectory_id",
        "time_s",
    ]
).reset_index(drop=True)

state_df.to_csv(
    STATE_PATH,
    index=False,
)

print(f"State rows:        {len(state_df):,}")
print(f"State columns:     {len(state_df.columns)}")
print()


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print("=" * 70)
print("INTEGRITY CHECKS")
print("=" * 70)

expected_field_rows = (
    df["trajectory_id"].nunique()
    * df["time_s"].nunique()
    * df["x"].nunique()
    * df["y"].nunique()
)

expected_state_rows = (
    df["trajectory_id"].nunique()
    * df["time_s"].nunique()
)

print()
print(f"Expected field rows: {expected_field_rows:,}")
print(f"Actual field rows:   {len(field_df):,}")

if len(field_df) != expected_field_rows:
    raise ValueError(
        "Field dataset row count does not match expectation."
    )

print("Field row count: PASS")

print()
print(f"Expected state rows: {expected_state_rows:,}")
print(f"Actual state rows:   {len(state_df):,}")

if len(state_df) != expected_state_rows:
    raise ValueError(
        "State dataset row count does not match expectation."
    )

print("State row count: PASS")


# ------------------------------------------------------------
# Check one state row per trajectory/time.
# ------------------------------------------------------------

state_duplicates = state_df.duplicated(
    subset=[
        "trajectory_id",
        "time_s",
    ]
).sum()

print()

print(
    f"Duplicate trajectory/time states: "
    f"{state_duplicates}"
)

if state_duplicates != 0:
    raise ValueError(
        "Duplicate trajectory/time states detected."
    )

print("State uniqueness: PASS")


# ------------------------------------------------------------
# Check no split leakage.
# ------------------------------------------------------------

split_per_trajectory = (
    field_df.groupby("trajectory_id")["split"]
    .nunique()
)

if split_per_trajectory.max() != 1:
    raise ValueError(
        "Trajectory appears in multiple dataset splits."
    )

print("Trajectory-level split isolation: PASS")


# ------------------------------------------------------------
# Check physical values.
# ------------------------------------------------------------

checks = {
    "h finite": np.isfinite(field_df["h"]).all(),
    "x finite": np.isfinite(field_df["x"]).all(),
    "y finite": np.isfinite(field_df["y"]).all(),
    "time finite": np.isfinite(field_df["time_s"]).all(),
    "rho finite": np.isfinite(field_df["rho"]).all(),
    "mu finite": np.isfinite(field_df["mu"]).all(),
    "h non-negative": (field_df["h"] >= 0).all(),
    "rho positive": (field_df["rho"] > 0).all(),
    "mu positive": (field_df["mu"] > 0).all(),
}

for name, passed in checks.items():

    print(
        f"{name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise ValueError(
            f"Physical-data check failed: {name}"
        )


# ============================================================
# RANGE REPORT
# ============================================================

print()
print("=" * 70)
print("DATA RANGES")
print("=" * 70)

print()
print("Field variables:")

for column in [
    "x",
    "y",
    "time_s",
    "h",
    "rho",
    "mu",
]:

    print(
        f"{column:12s}: "
        f"{field_df[column].min():.6e} "
        f"to "
        f"{field_df[column].max():.6e}"
    )


print()
print("Geometry/state variables:")

for column in [
    "xc",
    "yc",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "volume",
    "max_thickness",
]:

    print(
        f"{column:16s}: "
        f"{state_df[column].min():.6e} "
        f"to "
        f"{state_df[column].max():.6e}"
    )


# ============================================================
# FILE SIZES
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

for path in [
    FIELD_PATH,
    STATE_PATH,
    SPLIT_PATH,
]:

    size_mb = path.stat().st_size / (1024 ** 2)

    print(
        f"{path.name:35s} "
        f"{size_mb:,.2f} MB"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PREPARATION COMPLETE")
print("=" * 70)

print()
print("Field dataset:")
print(FIELD_PATH)

print()
print("Geometry/state dataset:")
print(STATE_PATH)

print()
print("Trajectory split:")
print(SPLIT_PATH)

print()
print("No row-level random splitting was performed.")
print("All points from each trajectory remain in one split.")

print("=" * 70)
