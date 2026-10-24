"""Build the matched hospital electricity-source ablation figure.

The current-system baseline uses corrected public-grid results for the 720
public hospitals and the independent solar model for the 660 solar-only
hospitals.  Public results take precedence for the 35 hospitals recorded with
both sources.  This produces the same 1,380-hospital cohort as the counterfactual
in which every hospital relies on the public grid.
"""

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hospital_grid_ablation import hospital_node_id
from synthetic_tracks import (
    STORM_FOLDER_TEMPLATE,
    TRACK_COUNTS,
    expected_track_columns,
)


IDENTITY_COLUMNS = ("node_id", "facility_i")
TRACK_COLUMN_PATTERN = re.compile(r"^\d+_\d+$")
EXPECTED_PUBLIC_HOSPITALS = 720
EXPECTED_ALL_GRID_HOSPITALS = 1380
EXPECTED_SOLAR_HOSPITALS = 695
EXPECTED_SOLAR_ONLY_HOSPITALS = 660


SYNTHETIC_TRACKS = sum(TRACK_COUNTS)
SYNTHETIC_CATALOGUE_YEARS = 10_000.0

def _facility_key(value):
    """Normalize DBF/CSV integer formatting without changing the output value."""

    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]
    return text


def _track_columns(frame):
    """Return recognized track columns in canonical order."""

    present = {
        str(column)
        for column in frame.columns
        if TRACK_COLUMN_PATTERN.fullmatch(str(column))
    }
    expected = expected_track_columns()
    unexpected = sorted(present.difference(expected))
    if unexpected:
        raise ValueError(
            "Unexpected synthetic-track columns: " + ", ".join(unexpected[:8])
        )
    return [column for column in expected if column in present]


def _normalize_identity(frame, label):
    missing = [column for column in IDENTITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing identity columns: {missing}")

    normalized = frame.copy()
    for column in IDENTITY_COLUMNS:
        if normalized[column].isna().any():
            raise ValueError(f"{label} contains missing {column} values")
        if column == "node_id":
            normalized[column] = normalized[column].astype(str).str.strip()
        else:
            normalized[column] = normalized[column].map(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )
        if normalized[column].map(
            lambda value: isinstance(value, str) and value == ""
        ).any():
            raise ValueError(f"{label} contains blank {column} values")

    duplicated_nodes = normalized["node_id"].duplicated(keep=False)
    if duplicated_nodes.any():
        examples = normalized.loc[duplicated_nodes, "node_id"].unique()[:5]
        raise ValueError(
            f"{label} contains duplicate node_id values: {examples.tolist()}"
        )
    facility_keys = normalized["facility_i"].map(_facility_key)
    duplicated_facilities = facility_keys.duplicated(keep=False)
    if duplicated_facilities.any():
        examples = normalized.loc[
            duplicated_facilities, "facility_i"
        ].unique()[:5]
        raise ValueError(
            f"{label} contains duplicate facility_i values: {examples.tolist()}"
        )
    return normalized


def _validate_probabilities(
    frame, track_columns, label, allow_unavailable=False
):
    if not track_columns:
        raise ValueError(f"{label} contains no synthetic-track columns")
    values = frame.loc[:, track_columns].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        bad_locations = np.argwhere(~np.isfinite(array))
        row, column = bad_locations[0]
        raise ValueError(
            f"{label} contains a missing/non-finite value at row {row}, "
            f"track {track_columns[column]}"
        )
    minimum = float(array.min())
    maximum = float(array.max())
    minimum_allowed = -1.0 if allow_unavailable else 0.0
    if minimum < minimum_allowed or maximum > 1.0:
        raise ValueError(
            f"{label} probabilities must be within "
            f"[{minimum_allowed:g}, 1]; "
            f"observed range is [{minimum}, {maximum}]"
        )
    if allow_unavailable:
        negative = array < 0.0
        if np.any(negative & ~np.isclose(array, -1.0)):
            raise ValueError(f"{label} unavailable values must equal -1")
    frame.loc[:, track_columns] = values


def load_power_summary(path, require_complete=True, allow_unavailable=False):
    """Load and validate a public or all-grid aggregate result CSV."""

    path = Path(path)
    frame = pd.read_csv(
        path,
        dtype={"node_id": "string"},
    )
    frame = _normalize_identity(frame, str(path))
    track_columns = _track_columns(frame)
    expected = expected_track_columns()
    missing = [column for column in expected if column not in track_columns]
    if require_complete and missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"{path} is not a complete 2703-track result; missing "
            f"{len(missing)} track(s), including {preview}"
        )
    _validate_probabilities(
        frame,
        track_columns,
        str(path),
        allow_unavailable=allow_unavailable,
    )
    return frame.loc[:, [*IDENTITY_COLUMNS, *track_columns]].copy()


def load_hospital_data(path):
    """Load the simulation-era SARA data and recreate graph node IDs."""

    path = Path(path)
    hospitals = gpd.read_file(path)
    if hospitals.crs is None:
        raise ValueError(f"Hospital data has no CRS: {path}")
    hospitals = hospitals.to_crs(epsg=4326)
    if not hospitals.index.is_unique:
        raise ValueError("SARA hospital row indices must be unique")
    if "facility_i" not in hospitals:
        raise ValueError("SARA hospital data is missing facility_i")
    hospitals = hospitals.copy()
    hospitals["node_id"] = [hospital_node_id(index) for index in hospitals.index]
    return hospitals


def _solar_result_path(root, storm_folder, kind):
    candidates = [
        root / f"{storm_folder}_var_track_solar_{kind}_func_2",
        root / f"{storm_folder}_var_track_solar_{kind}_func",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No solar {kind} aggregate found for {storm_folder}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _solar_identity(frame, hospitals, label):
    """Return solar identity columns, validating them against SARA."""

    if "facility_i" not in frame:
        raise ValueError(f"{label} is missing facility_i")
    identity = frame.loc[:, ["facility_i"]].copy()
    identity["facility_key"] = identity["facility_i"].map(_facility_key)

    hospital_identity = hospitals.loc[
        hospitals["facility_i"].notna(), ["node_id", "facility_i"]
    ].copy()
    hospital_identity["node_id"] = hospital_identity["node_id"].astype(str)
    hospital_identity["facility_key"] = hospital_identity["facility_i"].map(
        _facility_key
    )
    if hospital_identity["facility_key"].duplicated().any():
        raise ValueError("Non-null SARA facility_i values must be unique")

    if "node_id" in frame:
        identity.insert(0, "node_id", frame["node_id"].astype(str).str.strip())
        expected = identity.merge(
            hospital_identity,
            on=["node_id", "facility_key"],
            how="left",
            indicator=True,
            validate="one_to_one",
            suffixes=("", "_sara"),
        )
        if (expected["_merge"] != "both").any():
            bad = expected.loc[expected["_merge"] != "both", "node_id"].head()
            raise ValueError(
                f"{label} node/facility identity does not match SARA: "
                f"{bad.tolist()}"
            )
    else:
        identity = identity.merge(
            hospital_identity.loc[:, ["node_id", "facility_key"]],
            on="facility_key",
            how="left",
            validate="one_to_one",
        )
        if identity["node_id"].isna().any():
            bad = identity.loc[identity["node_id"].isna(), "facility_i"].head()
            raise ValueError(
                f"{label} facilities are missing from SARA: {bad.tolist()}"
            )
    identity = identity.loc[:, ["node_id", "facility_i"]]
    return _normalize_identity(identity, label)


def _solar_values(frame, count, label):
    wanted = [str(track_id) for track_id in range(count)]
    missing = [column for column in wanted if column not in frame]
    if missing:
        raise ValueError(
            f"{label} is missing {len(missing)} track column(s), including "
            + ", ".join(missing[:8])
        )
    values = frame.loc[:, wanted].apply(pd.to_numeric, errors="coerce")
    return values


def load_solar_summary(root, hospitals, zero_empty_events=True):
    """Load the independent-solar aggregate and construct 2,703 probabilities.

    A historical solar bug produced an entirely missing column when no
    interpolated storm center entered Mozambique.  When ``zero_empty_events``
    is true, only columns that are all missing in *both* irradiation and damage
    are converted to zero.  Partial or one-sided missing data always fail.
    Repaired track IDs are recorded in ``result.attrs['zeroed_empty_tracks']``.
    """

    root = Path(root)
    identity = None
    combined_columns = {}
    zeroed_tracks = []
    source_paths = []

    for storm_set, count in enumerate(TRACK_COUNTS):
        storm_folder = STORM_FOLDER_TEMPLATE.format(storm_set=storm_set)
        irr_path = _solar_result_path(root, storm_folder, "irr")
        damage_path = _solar_result_path(root, storm_folder, "damage")
        source_paths.extend([str(irr_path), str(damage_path)])
        irr = gpd.read_file(irr_path)
        damage = gpd.read_file(damage_path)

        irr_identity = _solar_identity(irr, hospitals, str(irr_path))
        damage_identity = _solar_identity(damage, hospitals, str(damage_path))
        irr_by_node = irr_identity.set_index("node_id")
        damage_by_node = damage_identity.set_index("node_id")
        if set(irr_by_node.index) != set(damage_by_node.index):
            raise ValueError(
                f"Solar irradiation/damage cohorts differ for {storm_folder}"
            )
        _check_facility_matches(irr_identity, damage_identity, "Solar damage")
        if identity is None:
            identity = irr_identity.sort_values("node_id").reset_index(drop=True)
        else:
            canonical_nodes = set(identity["node_id"])
            if set(irr_identity["node_id"]) != canonical_nodes:
                raise ValueError(f"Solar hospital cohort differs for {storm_folder}")
            _check_facility_matches(identity, irr_identity, "Solar")

        irr_values = _solar_values(irr, count, str(irr_path))
        damage_values = _solar_values(damage, count, str(damage_path))
        irr_values.index = irr_identity["node_id"]
        damage_values.index = damage_identity["node_id"]
        irr_values = irr_values.reindex(identity["node_id"]).reset_index(drop=True)
        damage_values = damage_values.reindex(
            identity["node_id"]
        ).reset_index(drop=True)

        for track_id in range(count):
            local_column = str(track_id)
            track_column = f"{storm_set}_{track_id}"
            irr_track = irr_values[local_column]
            damage_track = damage_values[local_column]
            irr_all_missing = bool(irr_track.isna().all())
            damage_all_missing = bool(damage_track.isna().all())
            if irr_all_missing or damage_all_missing:
                if (
                    zero_empty_events
                    and irr_all_missing
                    and damage_all_missing
                ):
                    irr_track = pd.Series(0.0, index=identity.index)
                    damage_track = pd.Series(0.0, index=identity.index)
                    zeroed_tracks.append(track_column)
                else:
                    raise ValueError(
                        f"Solar track {track_column} is entirely missing in "
                        "only one input or empty-event repair is disabled"
                    )
            elif irr_track.isna().any() or damage_track.isna().any():
                raise ValueError(
                    f"Solar track {track_column} contains partial missing values"
                )

            for component_label, component in (
                ("irradiation", irr_track),
                ("damage", damage_track),
            ):
                values = component.to_numpy(dtype=float)
                minimum = float(values.min())
                maximum = float(values.max())
                if (
                    not np.isfinite(values).all()
                    or minimum < 0.0
                    or maximum > 1.0
                ):
                    raise ValueError(
                        f"Solar {component_label} track {track_column} must be "
                        f"finite and within [0, 1]; observed [{minimum}, {maximum}]"
                    )

            combined = damage_track + (
                (1.0 - damage_track)
                * (irr_track > 0.6)
                * (irr_track - 0.6)
                / 0.4
            )
            combined_columns[track_column] = combined.to_numpy(dtype=float)

    output = pd.concat(
        [identity, pd.DataFrame(combined_columns, index=identity.index)],
        axis=1,
    )
    track_columns = _track_columns(output)
    if tuple(track_columns) != expected_track_columns():
        raise ValueError("Solar aggregate does not contain the complete catalogue")
    _validate_probabilities(output, track_columns, "independent-solar aggregate")
    output.attrs["zeroed_empty_tracks"] = tuple(zeroed_tracks)
    output.attrs["source_paths"] = tuple(source_paths)
    return output


def _matching_track_columns(*frames):
    tracks = [_track_columns(frame) for frame in frames]
    if not tracks[0]:
        raise ValueError("No track columns are available for comparison")
    if any(track_set != tracks[0] for track_set in tracks[1:]):
        descriptions = [len(track_set) for track_set in tracks]
        raise ValueError(
            "Public, all-grid, and solar track columns must match exactly; "
            f"observed counts are {descriptions}"
        )
    return tracks[0]


def _check_facility_matches(reference, candidate, label):
    reference_by_node = reference.set_index("node_id")["facility_i"]
    candidate_by_node = candidate.set_index("node_id")["facility_i"]
    shared = reference_by_node.index.intersection(candidate_by_node.index)
    reference_keys = reference_by_node.loc[shared].map(_facility_key)
    candidate_keys = candidate_by_node.loc[shared].map(_facility_key)
    mismatch = shared[reference_keys.to_numpy() != candidate_keys.to_numpy()]
    if len(mismatch):
        raise ValueError(
            f"{label} facility_i does not match all-grid identity for nodes: "
            f"{mismatch[:5].tolist()}"
        )


def combine_hybrid_baseline(public_df, all_grid_df, solar_df):
    """Construct the matched current-system baseline over the all-grid cohort."""

    public = _normalize_identity(public_df, "public result")
    all_grid = _normalize_identity(all_grid_df, "all-grid result")
    solar = _normalize_identity(solar_df, "solar result")
    track_columns = _matching_track_columns(public, all_grid, solar)
    _validate_probabilities(
        public,
        track_columns,
        "public result",
        allow_unavailable=True,
    )
    _validate_probabilities(
        all_grid,
        track_columns,
        "all-grid result",
        allow_unavailable=True,
    )
    _validate_probabilities(solar, track_columns, "solar result")

    public_nodes = set(public["node_id"])
    all_grid_nodes = set(all_grid["node_id"])
    solar_nodes = set(solar["node_id"])
    if not public_nodes.issubset(all_grid_nodes):
        extras = sorted(public_nodes.difference(all_grid_nodes))
        raise ValueError(
            "Public nodes must be a subset of all-grid nodes: " + str(extras[:5])
        )
    if not solar_nodes.issubset(all_grid_nodes):
        extras = sorted(solar_nodes.difference(all_grid_nodes))
        raise ValueError(
            "Solar nodes must be a subset of all-grid nodes: " + str(extras[:5])
        )
    missing = all_grid_nodes.difference(public_nodes.union(solar_nodes))
    if missing:
        raise ValueError(
            "All-grid hospitals missing from both current-system baselines: "
            + ", ".join(sorted(missing)[:8])
        )

    _check_facility_matches(all_grid, public, "Public")
    _check_facility_matches(all_grid, solar, "Solar")

    all_indexed = all_grid.set_index("node_id", drop=False)
    public_indexed = public.set_index("node_id", drop=False)
    solar_indexed = solar.set_index("node_id", drop=False)
    baseline_identity = all_indexed.loc[:, [*IDENTITY_COLUMNS]].copy()
    solar_only = all_indexed.index.difference(public_indexed.index, sort=False)
    baseline_values = pd.DataFrame(
        np.nan,
        index=all_indexed.index,
        columns=track_columns,
        dtype=float,
    )
    baseline_values.loc[solar_only, track_columns] = solar_indexed.loc[
        solar_only, track_columns
    ].to_numpy()
    # Public wins for the 35 public/solar overlaps, matching the original model.
    baseline_values.loc[public_indexed.index, track_columns] = public_indexed.loc[
        :, track_columns
    ].to_numpy()
    if baseline_values.isna().any().any():
        raise ValueError("Hybrid baseline construction left missing values")
    baseline = pd.concat([baseline_identity, baseline_values], axis=1)
    return baseline.reset_index(drop=True)


def build_comparison_table(public_df, all_grid_df, solar_df):
    """Return per-hospital paired means and ``all_grid - baseline`` change."""

    baseline = combine_hybrid_baseline(public_df, all_grid_df, solar_df)
    all_grid = _normalize_identity(all_grid_df, "all-grid result")
    track_columns = _matching_track_columns(baseline, all_grid)
    public_nodes = set(_normalize_identity(public_df, "public result")["node_id"])

    baseline_indexed = baseline.set_index("node_id")
    all_grid_indexed = all_grid.set_index("node_id", drop=False)
    comparison = all_grid_indexed.loc[:, [*IDENTITY_COLUMNS]].copy()
    comparison["baseline_source"] = np.where(
        comparison["node_id"].isin(public_nodes), "public", "solar_only"
    )
    baseline_tracks = baseline_indexed.loc[comparison.index, track_columns]
    comparison["baseline_mean"] = baseline_tracks.where(
        baseline_tracks >= 0.0
    ).mean(axis=1).fillna(-1.0)
    all_grid_tracks = all_grid_indexed.loc[comparison.index, track_columns]
    comparison["all_grid_mean"] = all_grid_tracks.where(
        all_grid_tracks >= 0.0
    ).mean(axis=1).fillna(-1.0)
    comparison["delta"] = np.where(
        (comparison["baseline_mean"] >= 0.0)
        & (comparison["all_grid_mean"] >= 0.0),
        comparison["all_grid_mean"] - comparison["baseline_mean"],
        np.nan,
    )

    if "geometry" in all_grid_df:
        geometry = all_grid_df.set_index("node_id").loc[
            comparison.index, "geometry"
        ]
        comparison["geometry"] = geometry
        if isinstance(all_grid_df, gpd.GeoDataFrame):
            comparison = gpd.GeoDataFrame(
                comparison,
                geometry="geometry",
                crs=all_grid_df.crs,
            )
    return comparison.reset_index(drop=True)


def attach_hospital_geometry(frame, hospitals):
    """Join SARA geometry using both stable node and facility identifiers."""

    normalized = _normalize_identity(frame, "comparison result")
    hospital_identity = hospitals.loc[
        hospitals["facility_i"].notna(), ["node_id", "facility_i", "geometry"]
    ].copy()
    hospital_identity["node_id"] = hospital_identity["node_id"].astype(str)
    _check_facility_matches(normalized, hospital_identity, "SARA")
    merged = normalized.merge(
        hospital_identity.loc[:, ["node_id", "geometry"]],
        on="node_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["geometry"].isna().any():
        missing = merged.loc[merged["geometry"].isna(), "node_id"].head()
        raise ValueError(
            "Comparison hospitals missing SARA geometry: " + str(missing.tolist())
        )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=hospitals.crs)


def validate_standard_cohorts(public_df, all_grid_df, solar_df):
    """Validate the expected Mozambique hospital-source cohorts."""

    public_nodes = set(public_df["node_id"].astype(str))
    all_grid_nodes = set(all_grid_df["node_id"].astype(str))
    solar_nodes = set(solar_df["node_id"].astype(str))
    counts = {
        "public": len(public_nodes),
        "all_grid": len(all_grid_nodes),
        "solar": len(solar_nodes),
        "solar_only": len(solar_nodes.difference(public_nodes)),
        "public_solar_overlap": len(solar_nodes.intersection(public_nodes)),
    }
    expected = {
        "public": EXPECTED_PUBLIC_HOSPITALS,
        "all_grid": EXPECTED_ALL_GRID_HOSPITALS,
        "solar": EXPECTED_SOLAR_HOSPITALS,
        "solar_only": EXPECTED_SOLAR_ONLY_HOSPITALS,
        "public_solar_overlap": 35,
    }
    if counts != expected:
        raise ValueError(f"Unexpected hospital cohorts: {counts}; expected {expected}")
    if public_nodes.union(solar_nodes) != all_grid_nodes:
        raise ValueError("Public/solar union does not equal the all-grid cohort")
    return counts


def summarize_comparison(comparison):
    """Summarize paired changes overall and by current electricity source."""

    records = []
    groups = [
        ("all_hospitals", comparison),
        ("public", comparison[comparison["baseline_source"] == "public"]),
        (
            "solar_only",
            comparison[comparison["baseline_source"] == "solar_only"],
        ),
    ]
    for label, group in groups:
        evaluated = group.loc[
            group["baseline_mean"].ge(0.0)
            & group["all_grid_mean"].ge(0.0)
            & group["delta"].notna()
        ].copy()
        delta = evaluated["delta"]
        records.append(
            {
                "cohort": label,
                "hospitals": len(group),
                "evaluated_hospitals": len(evaluated),
                "baseline_mean": evaluated["baseline_mean"].mean(),
                "all_grid_mean": evaluated["all_grid_mean"].mean(),
                "mean_delta_percentage_points": 100.0 * delta.mean(),
                "median_delta_percentage_points": 100.0 * delta.median(),
                "share_all_grid_higher": (delta > 0).mean(),
                "share_all_grid_lower": (delta < 0).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def load_boundary(path):
    """Load a Mozambique boundary from either a country layer or admin layer."""

    boundary = gpd.read_file(path)
    if boundary.crs is None:
        raise ValueError(f"Boundary data has no CRS: {path}")
    boundary = boundary.to_crs(epsg=4326)
    for country_column in ("COUNTRY", "ADMIN", "NAME_0", "ADM0_EN"):
        if country_column in boundary:
            selected = boundary[
                boundary[country_column].astype(str).str.casefold()
                == "mozambique"
            ]
            if len(selected):
                boundary = selected
                break
    return boundary


def _finite_max(values, minimum):
    value = float(np.nanmax(np.asarray(values, dtype=float)))
    return max(value, minimum)


def plot_ablation_figure(comparison_gdf, boundary_gdf=None):
    """Plot current hybrid, all-grid, and paired-change hospital maps."""

    required = {
        "baseline_source",
        "baseline_mean",
        "all_grid_mean",
        "delta",
        "geometry",
    }
    missing = required.difference(comparison_gdf.columns)
    if missing:
        raise ValueError(f"Comparison table is missing plotting fields: {missing}")
    if comparison_gdf.crs is None:
        raise ValueError("Comparison geometry must define a CRS")
    comparison = comparison_gdf.to_crs(epsg=4326).copy()
    if boundary_gdf is not None:
        boundary = boundary_gdf.to_crs(epsg=4326)
    else:
        boundary = None

    # Annualized disruption index (yr^-1): mean event-level value times the
    # annual frequency of synthetic cyclones affecting Mozambique.
    annual_frequency = SYNTHETIC_TRACKS / SYNTHETIC_CATALOGUE_YEARS
    comparison["baseline_index"] = annual_frequency * comparison["baseline_mean"]
    comparison["all_grid_index"] = annual_frequency * comparison["all_grid_mean"]
    comparison["delta_index"] = annual_frequency * comparison["delta"]
    risk_values = comparison[["baseline_index", "all_grid_index"]].where(
        comparison[["baseline_mean", "all_grid_mean"]].to_numpy() >= 0.0
    )
    risk_max = _finite_max(risk_values.to_numpy(), 1e-9)
    delta_limit = _finite_max(
        np.abs(comparison["delta_index"].to_numpy()),
        1e-9,
    )
    risk_norm = colors.Normalize(vmin=0.0, vmax=risk_max)
    delta_norm = colors.TwoSlopeNorm(
        vmin=-delta_limit,
        vcenter=0.0,
        vmax=delta_limit,
    )
    # Match the BuGn outage-probability palette used by the source notebook.
    # Keep the shared data-driven range so current values above the notebook's
    # historical 5% ceiling remain distinguishable rather than saturating.
    risk_cmap = plt.get_cmap("BuGn")
    delta_cmap = plt.get_cmap("RdBu_r")

    figure = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(0.055, 0.945),
        hspace=0.02,
    )
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.set_axis_off()
    axes = np.asarray(
        [figure.add_subplot(grid[1, column]) for column in range(3)]
    )
    for axis in axes:
        if boundary is not None:
            boundary.plot(
                ax=axis,
                color="#f1f1f1",
                edgecolor="#555555",
                linewidth=0.65,
                zorder=0,
            )

    source_styles = {
        "public": ("#0072B2", "Public-grid baseline"),
        "solar_only": ("#E69F00", "Independent-solar baseline"),
    }
    for source, (edge_color, _) in source_styles.items():
        source_data = comparison[comparison["baseline_source"] == source]
        baseline_available = source_data[source_data["baseline_mean"].ge(0.0)]
        baseline_unavailable = source_data[source_data["baseline_mean"].lt(0.0)]
        delta_available = source_data[source_data["delta"].notna()]
        delta_unavailable = source_data[source_data["delta"].isna()]
        if not baseline_available.empty:
            baseline_available.plot(
                ax=axes[0],
                column="baseline_index",
                cmap=risk_cmap,
                norm=risk_norm,
                markersize=11,
                edgecolor=edge_color,
                linewidth=0.28,
                zorder=2,
            )
        if not delta_available.empty:
            delta_available.plot(
                ax=axes[2],
                column="delta_index",
                cmap=delta_cmap,
                norm=delta_norm,
                markersize=11,
                edgecolor=edge_color,
                linewidth=0.28,
                zorder=2,
            )
        if not baseline_unavailable.empty:
            baseline_unavailable.plot(
                ax=axes[0],
                color="silver",
                markersize=11,
                edgecolor="#555555",
                linewidth=0.28,
                zorder=2,
            )
        if not delta_unavailable.empty:
            delta_unavailable.plot(
                ax=axes[2],
                color="silver",
                markersize=11,
                edgecolor="#555555",
                linewidth=0.28,
                zorder=2,
            )
    all_grid_available = comparison[comparison["all_grid_mean"].ge(0.0)]
    all_grid_unavailable = comparison[comparison["all_grid_mean"].lt(0.0)]
    if not all_grid_available.empty:
        all_grid_available.plot(
            ax=axes[1],
            column="all_grid_index",
            cmap=risk_cmap,
            norm=risk_norm,
            markersize=11,
            edgecolor="#333333",
            linewidth=0.18,
            zorder=2,
        )
    if not all_grid_unavailable.empty:
        all_grid_unavailable.plot(
            ax=axes[1],
            color="silver",
            markersize=11,
            edgecolor="#555555",
            linewidth=0.28,
            zorder=2,
        )

    titles = (
        "a  Current hybrid system",
        "b  All hospitals on public grid",
        "c  Paired change: all-grid − current",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        axis.set_axis_off()
        axis.set_aspect("equal")

    bounds = (
        boundary.total_bounds
        if boundary is not None and len(boundary)
        else comparison.total_bounds
    )
    min_x, min_y, max_x, max_y = bounds
    x_padding = max((max_x - min_x) * 0.04, 0.1)
    y_padding = max((max_y - min_y) * 0.04, 0.1)
    for axis in axes:
        axis.set_xlim(min_x - x_padding, max_x + x_padding)
        axis.set_ylim(min_y - y_padding, max_y + y_padding)

    risk_bar = figure.colorbar(
        cm.ScalarMappable(norm=risk_norm, cmap=risk_cmap),
        ax=axes[:2],
        orientation="horizontal",
        fraction=0.045,
        pad=0.025,
    )
    risk_bar.set_label("Annualized electricity disruption index (yr$^{-1}$)")
    risk_bar.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    delta_bar = figure.colorbar(
        cm.ScalarMappable(norm=delta_norm, cmap=delta_cmap),
        ax=axes[2],
        orientation="horizontal",
        fraction=0.045,
        pad=0.025,
    )
    delta_bar.set_label("Change in annualized electricity disruption index (yr$^{-1}$)")
    delta_bar.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=edge_color,
            markeredgewidth=1.2,
            markersize=6,
            label=label,
        )
        for edge_color, label in source_styles.values()
    ]
    # Give the source legend its own layout row. A figure-level legend above
    # the axes is not included in constrained-layout calculations and can
    # overlap the center-panel title when the figure is saved tightly.
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
    )
    return figure


def _write_comparison_csv(comparison, path):
    output = comparison.drop(columns="geometry").copy()
    output["longitude"] = comparison.geometry.x
    output["latitude"] = comparison.geometry.y
    output.to_csv(path, index=False)


def generate_outputs(
    public_summary,
    all_grid_summary,
    solar_root,
    hospital_data,
    boundary_path,
    output_prefix,
    validate_cohorts=True,
    zero_empty_solar_events=True,
):
    """Run the complete validated analysis and write figure/data artifacts."""

    # Final hybrid public-grid outputs retain the complete 720-row reporting
    # cohort and mark nine facilities outside the generator-backed active
    # component as -1.  Preserve those rows as unavailable in the figure.
    public = load_power_summary(
        public_summary, require_complete=True, allow_unavailable=True
    )
    all_grid = load_power_summary(
        all_grid_summary,
        require_complete=True,
        allow_unavailable=True,
    )
    hospitals = load_hospital_data(hospital_data)
    solar = load_solar_summary(
        solar_root,
        hospitals,
        zero_empty_events=zero_empty_solar_events,
    )
    cohort_counts = (
        validate_standard_cohorts(public, all_grid, solar)
        if validate_cohorts
        else None
    )
    all_grid_geo = attach_hospital_geometry(all_grid, hospitals)
    comparison = build_comparison_table(public, all_grid_geo, solar)
    summary = summarize_comparison(comparison)
    boundary = load_boundary(boundary_path) if boundary_path else None
    figure = plot_ablation_figure(comparison, boundary)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    _write_comparison_csv(
        comparison,
        prefix.with_name(prefix.name + "_per_hospital.csv"),
    )
    summary.to_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        index=False,
    )
    metadata = {
        "track_count": len(expected_track_columns()),
        "track_weighting": "equal weight per synthetic cyclone",
        "delta_definition": "all_grid_mean - baseline_mean",
        "solar_empty_tracks_set_to_zero": list(
            solar.attrs.get("zeroed_empty_tracks", ())
        ),
        "cohort_counts": cohort_counts,
        "inputs": {
            "public_summary": str(public_summary),
            "all_grid_summary": str(all_grid_summary),
            "solar_root": str(solar_root),
            "hospital_data": str(hospital_data),
            "boundary": str(boundary_path) if boundary_path else None,
        },
    }
    with prefix.with_name(prefix.name + "_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return comparison, summary, figure, metadata


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a matched current-hybrid versus all-grid hospital "
            "electricity ablation figure."
        )
    )
    parser.add_argument("--public-summary", required=True)
    parser.add_argument("--all-grid-summary", required=True)
    parser.add_argument("--solar-root", required=True)
    parser.add_argument("--hospital-data", required=True)
    parser.add_argument("--boundary")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--allow-nonstandard-cohorts",
        action="store_true",
        help="Skip project-specific 720/1380/695 cohort count checks.",
    )
    parser.add_argument(
        "--error-on-empty-solar-events",
        action="store_true",
        help="Reject rather than zero legacy all-NaN no-impact solar events.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _, summary, figure, metadata = generate_outputs(
        public_summary=args.public_summary,
        all_grid_summary=args.all_grid_summary,
        solar_root=args.solar_root,
        hospital_data=args.hospital_data,
        boundary_path=args.boundary,
        output_prefix=args.output_prefix,
        validate_cohorts=not args.allow_nonstandard_cohorts,
        zero_empty_solar_events=not args.error_on_empty_solar_events,
    )
    plt.close(figure)
    print(summary.to_string(index=False))
    print(
        "Solar empty-event tracks set to zero:",
        metadata["solar_empty_tracks_set_to_zero"],
    )
    print(f"Wrote figure bundle with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
