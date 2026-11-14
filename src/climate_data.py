"""NetCDF loading, coordinate discovery, subsetting, and anomaly calculations."""

from __future__ import annotations


import io
import fnmatch
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union
import re

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import xarray as xr

_COMMON_FUNCTION_ROOT_ENV = os.environ.get("EARTH_DATA_COMMON_FUNCTION_ROOT", "").strip()
_COMMON_FUNCTION_ROOT_CANDIDATES = [
    Path(_COMMON_FUNCTION_ROOT_ENV) if _COMMON_FUNCTION_ROOT_ENV else None,
    Path(__file__).resolve().parents[2] / "common-files" / "function",
]
_ORIGINAL_FUNCTION_ROOT = next(
    (path for path in _COMMON_FUNCTION_ROOT_CANDIDATES if path is not None and path.is_dir()),
    None,
)
if _ORIGINAL_FUNCTION_ROOT is None:
    raise ImportError(
        "The shared helper directory is unavailable. Set "
        "EARTH_DATA_COMMON_FUNCTION_ROOT to the common-files/function directory "
        "or place common-files beside the project directory."
    )
if str(_ORIGINAL_FUNCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORIGINAL_FUNCTION_ROOT))

from shared_helpers.detrend import detrend_data as _shared_detrend_data
from shared_helpers.ensemble import (
    ensemble_agreement_fraction as _ensemble_agreement_fraction,
    ensemble_consensus as _ensemble_consensus,
)
from shared_helpers.annual_cycle import (
    AnnualCycleResult,
    TimeResolution,
    build_annual_cycles,
    continuous_running_mean,
    detect_time_resolution,
    polygon_grid_mask,
    spatial_mean_timeseries,
)


class DatasetError(ValueError):
    """Raised when a dataset cannot satisfy the viewer's data contract."""


ALIASES = {
    "time": ("time", "datetime", "date", "valid_time"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
    "level": (
        "level",
        "lev",
        "plev",
        "pressure",
        "pressure_level",
        "isobaricInhPa",
        "isobaricInPa",
    ),
}


@dataclass(frozen=True)
class PeriodInfo:
    years: list[int]
    year_min: int
    year_max: int


MAX_PLOT_CELLS = 5_000_000
MAX_EXPORT_BYTES = 512 * 1024 * 1024
MAX_ANNUAL_SOURCE_BYTES = 768 * 1024 * 1024


def finite_data_range(da: xr.DataArray) -> tuple[float, float]:
    """Return the finite minimum and maximum without eagerly loading all values."""
    try:
        numeric = da.astype(float)
        minimum = float(numeric.min(skipna=True).values)
        maximum = float(numeric.max(skipna=True).values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError("The selected variable does not contain numeric values.") from exc
    if not np.all(np.isfinite([minimum, maximum])):
        raise DatasetError("The selected variable contains no finite numeric values.")
    return minimum, maximum


def discover_netcdf_files(roots: Iterable[Path], limit: int | None = 80, pattern: str = "*") -> list[Path]:
    """Find a bounded set of local NetCDF files without reading their contents."""
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.nc", "*.nc4", "*.cdf", "*.h5", "*.hdf5", "*.he5"):
            try:
                # Combining a folder is intentionally non-recursive. Nested
                # folders are separate data collections and must be selected
                # explicitly by the user.
                paths = root.glob(suffix)
                for path in paths:
                    if path not in seen and fnmatch.fnmatch(path.name, pattern):
                        seen.add(path)
                        files.append(path)
                    if limit is not None and len(files) >= limit:
                        return sorted(files)
            except OSError:
                continue
    return sorted(files)


def load_dataset(
    source: str | tuple[str, ...] | tuple[tuple[str, ...], ...],
    payload: bytes | None = None,
) -> xr.Dataset:
    """Open NetCDF/HDF5 data lazily, with filename/metadata time recovery."""
    try:
        if source == "upload":
            if not payload:
                raise DatasetError("No uploaded file was supplied.")
            return _standardize(_open_one(io.BytesIO(payload), name="uploaded-data"))
        if isinstance(source, tuple):
            if not source:
                raise DatasetError("No NetCDF files were selected for combination.")
            if isinstance(source[0], tuple):
                groups = [tuple(group) for group in source]
                if len(groups) < 2 or any(not group for group in groups):
                    raise DatasetError("An ensemble requires at least two non-empty member folders.")
                members = [_standardize(load_dataset(group)) for group in groups]
                variables = [set(member.data_vars) for member in members]
                if any(names != variables[0] for names in variables[1:]):
                    raise DatasetError("Ensemble members must contain the same data variables.")
                try:
                    return xr.concat(
                        members,
                        dim=xr.IndexVariable("ensemble", np.arange(len(members))),
                        coords="minimal",
                        compat="equals",
                        join="exact",
                    )
                except (ValueError, OSError) as exc:
                    raise DatasetError(
                        "Ensemble members must have matching latitude, longitude, and time coordinates."
                    ) from exc
            try:
                dataset = xr.open_mfdataset(
                    list(source), combine="by_coords", chunks="auto",
                    parallel=False, decode_times=True, mask_and_scale=True,
                )
            except (ValueError, OSError) as exc:
                # Some HDF5 files are not NetCDF-compatible or have no
                # decodable time coordinate. Open and standardize each file,
                # then combine only after spatial/time subsetting by xarray.
                datasets = [_open_one(path) for path in source]
                try:
                    dataset = xr.combine_by_coords(datasets, combine_attrs="override")
                except Exception:
                    raise exc
            return _standardize(dataset)
        return _standardize(_open_one(source))
    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(str(exc)) from exc


def ensemble_member_count(ds: xr.Dataset) -> int:
    """Return the number of loaded ensemble members, or one for a single dataset."""
    return int(ds.sizes.get("ensemble", 1))


def ensemble_mean_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Return the arithmetic mean across ensemble members without loading all values."""
    if "ensemble" not in ds.dims:
        return ds
    result = ds.mean("ensemble", skipna=True, keep_attrs=True)
    result.attrs = ds.attrs.copy()
    result.attrs["ensemble_mean_members"] = ensemble_member_count(ds)
    return result


def ensemble_consensus(da: xr.DataArray, threshold: float = 0.8) -> xr.DataArray | None:
    """Return a robust-significance mask for an ensemble field."""
    if "ensemble" not in da.dims:
        return None
    try:
        return _ensemble_consensus(da, threshold)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc


def ensemble_agreement_fraction(da: xr.DataArray) -> xr.DataArray | None:
    """Return the fraction of ensemble members sharing the ensemble-mean sign."""
    if "ensemble" not in da.dims:
        return None
    try:
        return _ensemble_agreement_fraction(da)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc


def dataset_to_netcdf_bytes(ds: xr.Dataset) -> bytes:
    """Serialize a dataset in memory for a Streamlit download."""
    try:
        estimated_bytes = int(ds.nbytes)
        if estimated_bytes > MAX_EXPORT_BYTES:
            raise DatasetError(
                f"The NetCDF export is estimated at {estimated_bytes / 1024**3:.2f} GiB, "
                f"above the safe in-memory limit of {MAX_EXPORT_BYTES / 1024**3:.2f} GiB."
            )
        return bytes(ds.to_netcdf(engine="scipy"))
    except DatasetError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetError(f"Unable to serialize the NetCDF download: {exc}") from exc


def validate_plot_size(da: xr.DataArray, max_cells: int = MAX_PLOT_CELLS) -> None:
    """Reject spatial fields that would require an unsafe in-memory plot."""
    spatial_dims = set(da.coords.get("lat", xr.DataArray()).dims)
    spatial_dims.update(da.coords.get("lon", xr.DataArray()).dims)
    cell_count = 1
    for dimension in spatial_dims:
        cell_count *= int(da.sizes.get(dimension, 1))
    if cell_count > max_cells:
        raise DatasetError(
            f"The selected grid contains {cell_count:,} cells, above the safe plotting limit "
            f"of {max_cells:,}. Select a smaller region or lower-resolution dataset."
        )


def _open_one(source: str | Path | io.BytesIO, name: str | None = None) -> xr.Dataset:
    """Try CF/NetCDF and HDF5 engines without eagerly loading data."""
    attempts: list[Exception] = []
    for engine in (None, "h5netcdf", "netcdf4"):
        try:
            kwargs = {"decode_times": True, "mask_and_scale": True}
            if engine is not None:
                kwargs["engine"] = engine
            try:
                return xr.open_dataset(source, chunks="auto", **kwargs)
            except ValueError as exc:
                # dask is optional; retain lazy decoding when available but
                # still make small/non-dask installations usable.
                if "chunk manager" in str(exc).lower() or "dask" in str(exc).lower():
                    return xr.open_dataset(source, **kwargs)
                raise
        except Exception as exc:
            attempts.append(exc)
    # A valid HDF5 container can have numeric time plus a CF reference string
    # that xarray cannot decode. Retry without decoding and recover it below.
    try:
        raw = xr.open_dataset(source, decode_times=False, mask_and_scale=True)
    except Exception as exc:
        detail = "; ".join(str(error) for error in attempts[-3:])
        raise DatasetError(f"Unable to open {name or source}: {detail}") from exc
    return _recover_time_coordinate(raw, str(name or source))


def _recover_time_coordinate(ds: xr.Dataset, source_name: str) -> xr.Dataset:
    """Recover a missing/undecodable time coordinate from CF attrs or filename."""
    names = choose_coordinate_names(ds)
    time_name = names["time"]
    if time_name is None:
        match = re.search(r"(?<!\d)((?:19|20)\d{2})(?:[-_]?([01]\d))?(?:[-_]?([0-3]\d))?", source_name)
        if match is None:
            metadata = " ".join(
                str(value)
                for key, value in ds.attrs.items()
                if any(token in key.lower() for token in ("date", "time", "period", "reference", "title"))
            )
            match = re.search(
                r"(?<!\d)((?:19|20)\d{2})(?:[-_/]?([01]\d))?(?:[-_/]?([0-3]\d))?",
                metadata,
            )
        data_dims = next((variable.dims for variable in ds.data_vars.values() if variable.dims), ())
        if match and data_dims:
            year, month, day = int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1)
            try:
                timestamp = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}")
            except ValueError as exc:
                raise DatasetError(f"Invalid date recovered from metadata: {year}-{month:02d}-{day:02d}") from exc
            candidate = next(
                (dimension for dimension in data_dims if dimension not in {"lat", "latitude", "lon", "longitude", "x", "y", "level"}),
                None,
            )
            if candidate is None:
                return ds
            size = ds.sizes[candidate]
            ds = ds.rename_dims({candidate: "time"}).assign_coords(
                time=np.repeat(timestamp, size)
            )
        return ds
    coord = ds[time_name]
    units = str(coord.attrs.get("units", ""))
    calendar = coord.attrs.get("calendar", "standard")
    if np.issubdtype(coord.dtype, np.number) and "since" in units:
        try:
            decoded = xr.coding.times.decode_cf_datetime(coord, units, calendar)
            return ds.assign_coords({time_name: decoded})
        except (ValueError, OverflowError, TypeError):
            pass
    return ds


def choose_coordinate_names(ds: xr.Dataset) -> dict[str, str | None]:
    """Report the coordinate names selected by the alias resolver."""
    return {key: _find_name(ds, aliases) for key, aliases in ALIASES.items()}


def _find_name(ds: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
    names = set(ds.coords) | set(ds.dims)
    for alias in aliases:
        if alias in names:
            return alias
    return None


def _standardize(ds: xr.Dataset) -> xr.Dataset:
    ds = smart_mask_fill_values(ds)
    names = choose_coordinate_names(ds)
    missing = [key for key in ("lat", "lon") if names[key] is None]
    if missing:
        raise DatasetError(f"Missing required coordinate(s): {', '.join(missing)}.")
    rename = {value: key for key, value in names.items() if value and value != key}
    ds = ds.rename(rename)
    ds = _normalize_spatial_coordinates(ds)
    lon = ds.lon
    if float(lon.max()) > 180:
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
    ds = ds.sortby("lon").sortby("lat")
    return ds


def _normalize_spatial_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Normalize rectilinear coordinates or regrid a curvilinear grid."""
    if ds.lat.ndim == 2 or ds.lon.ndim == 2:
        return _regrid_curvilinear_dataset(ds)
    return _require_rectilinear_coordinates(ds)


def _require_rectilinear_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Validate and normalize the 1-D spatial coordinates used by the viewer."""
    dimension_renames: dict[str, str] = {}
    for name in ("lat", "lon"):
        coordinate = ds[name]
        if coordinate.ndim != 1:
            raise DatasetError(
                f"{name} coordinate is {coordinate.ndim}-D "
                f"with dims={coordinate.dims} and shape={coordinate.shape}. "
                "This viewer requires 1-D latitude/longitude coordinates; "
                "regrid curvilinear CMIP6 data before loading it."
            )
        dimension = coordinate.dims[0]
        if dimension != name:
            if name in ds.dims and name != dimension:
                raise DatasetError(
                    f"Cannot normalize {name} coordinate dimension {dimension!r}: "
                    f"dimension {name!r} already exists."
                )
            dimension_renames[dimension] = name
    if dimension_renames:
        ds = ds.swap_dims(dimension_renames)
    return ds


def _regrid_curvilinear_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Regrid a 2-D latitude/longitude grid to a same-sized regular grid."""
    latitude = ds.lat
    longitude = ds.lon
    if latitude.ndim != 2 or longitude.ndim != 2:
        raise DatasetError(
            "Latitude and longitude must both be either 1-D or 2-D; "
            f"got lat dims={latitude.dims}, lon dims={longitude.dims}."
        )
    if latitude.dims != longitude.dims:
        raise DatasetError(
            "Latitude and longitude must use the same 2-D grid dimensions; "
            f"got lat dims={latitude.dims}, lon dims={longitude.dims}."
        )
    grid_dims = latitude.dims
    grid_shape = latitude.shape
    if any(size < 2 for size in grid_shape):
        raise DatasetError(
            f"Curvilinear grid must have at least two points per dimension; "
            f"got dims={grid_dims}, shape={grid_shape}."
        )
    latitude_values = np.asarray(latitude.values, dtype=float)
    longitude_values = np.asarray(longitude.values, dtype=float)
    valid = np.isfinite(latitude_values) & np.isfinite(longitude_values)
    if not np.any(valid):
        raise DatasetError("The curvilinear grid contains no finite latitude/longitude points.")

    if _is_separable_curvilinear_grid(latitude_values, longitude_values):
        return _convert_separable_grid(ds, latitude_values, longitude_values, grid_dims)

    target_latitude = np.linspace(
        float(np.nanmin(latitude_values)),
        float(np.nanmax(latitude_values)),
        grid_shape[0],
    )
    longitude_start, longitude_span = _minimal_longitude_interval(longitude_values[valid])
    target_longitude = longitude_start + np.linspace(
        0.0,
        longitude_span,
        grid_shape[1],
        endpoint=longitude_span < 360.0 - 1e-8,
    )
    source_longitude = ((longitude_values - longitude_start) % 360.0) + longitude_start
    source_points = np.column_stack(
        (latitude_values[valid], source_longitude[valid])
    )
    target_latitude_grid, target_longitude_grid = np.meshgrid(
        target_latitude, target_longitude, indexing="ij"
    )
    tree = cKDTree(source_points)
    _, nearest = tree.query(
        np.column_stack(
            (target_latitude_grid.ravel(), target_longitude_grid.ravel())
        )
    )
    source_indices = np.flatnonzero(valid)[nearest]
    return _apply_curvilinear_regrid(
        ds,
        grid_dims,
        source_indices,
        target_latitude,
        target_longitude,
        grid_shape,
    )


def _is_separable_curvilinear_grid(latitude: np.ndarray, longitude: np.ndarray) -> bool:
    """Return whether 2-D coordinates are repetitions of 1-D axes."""
    return bool(
        np.allclose(latitude, latitude[:, :1], equal_nan=True)
        and np.allclose(longitude, longitude[:1, :], equal_nan=True)
    )


def _convert_separable_grid(
    ds: xr.Dataset,
    latitude: np.ndarray,
    longitude: np.ndarray,
    grid_dims: tuple[str, str],
) -> xr.Dataset:
    """Convert a separable 2-D coordinate pair without changing data values."""
    lat_dim, lon_dim = grid_dims
    result = ds.drop_vars(["lat", "lon"])
    result = result.assign_coords(
        lat=(lat_dim, latitude[:, 0]),
        lon=(lon_dim, longitude[0, :]),
    )
    return result.swap_dims({lat_dim: "lat", lon_dim: "lon"})


def _minimal_longitude_interval(longitudes: np.ndarray) -> tuple[float, float]:
    """Find the shortest continuous longitude interval containing the grid."""
    normalized = np.sort(np.asarray(longitudes, dtype=float) % 360.0)
    if normalized.size == 1:
        return float(normalized[0]), 0.0
    gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
    largest_gap = int(np.argmax(gaps))
    start = normalized[(largest_gap + 1) % normalized.size]
    span = 360.0 - gaps[largest_gap]
    return float(start), float(span)


def _apply_curvilinear_regrid(
    ds: xr.Dataset,
    grid_dims: tuple[str, str],
    source_indices: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    grid_shape: tuple[int, int],
) -> xr.Dataset:
    """Apply nearest-neighbour source-cell mapping to dataset variables."""
    lat_dim, lon_dim = grid_dims
    source_size = int(np.prod(grid_shape))
    target_shape = (target_latitude.size, target_longitude.size)
    variables: dict[str, xr.DataArray] = {}
    for name, variable in ds.data_vars.items():
        if lat_dim not in variable.dims or lon_dim not in variable.dims:
            variables[name] = variable
            continue
        other_dims = [dim for dim in variable.dims if dim not in grid_dims]
        ordered = variable.transpose(*other_dims, lat_dim, lon_dim)
        # Index through xarray's data object so dask-backed CMIP6 variables
        # remain lazy.  Calling ``.values`` here would load every time,
        # level, and ensemble member before the user selects a period.
        flattened = ordered.stack(_source_grid=grid_dims)
        selected = flattened.isel(
            _source_grid=xr.DataArray(source_indices, dims="_target_grid")
        )
        regridded = selected.data.reshape(
            tuple(ordered.sizes[dimension] for dimension in other_dims)
            + target_shape
        )
        variables[name] = xr.DataArray(
            regridded,
            dims=tuple(other_dims) + ("lat", "lon"),
            coords={
                dim: ordered.coords[dim]
                for dim in other_dims
                if dim in ordered.coords
            },
            name=variable.name,
            attrs=variable.attrs,
        )
    coordinates = {
        name: coordinate
        for name, coordinate in ds.coords.items()
        if name not in {"lat", "lon"} and not set(grid_dims).intersection(coordinate.dims)
    }
    coordinates.update(
        {
            "lat": ("lat", target_latitude),
            "lon": ("lon", target_longitude),
        }
    )
    result = xr.Dataset(variables, coords=coordinates, attrs=ds.attrs)
    result.attrs["curvilinear_regrid"] = "nearest-neighbour"
    result.attrs["curvilinear_source_grid"] = f"{grid_dims[0]}={grid_shape[0]}, {grid_dims[1]}={grid_shape[1]}"
    return result


def available_periods(da: xr.DataArray) -> PeriodInfo:
    if "time" not in da.coords:
        return PeriodInfo([], 0, 0)
    years = np.asarray(da.time.dt.year.values, dtype=int)
    unique = sorted(set(years.tolist()))
    return PeriodInfo(unique, unique[0], unique[-1]) if unique else PeriodInfo([], 0, 0)


def temporal_resolution(da: xr.DataArray) -> TimeResolution:
    """Detect the temporal sampling of a data array."""
    if "time" not in da.coords:
        raise DatasetError("The selected variable has no time coordinate.")
    return detect_time_resolution(da.time)


def polygon_mean_timeseries(
    da: xr.DataArray,
    vertices: Sequence[tuple[float, float]],
    weighting: str = "coslat",
    level: float | None = None,
) -> xr.DataArray:
    """Return a polygon-averaged time series using a safe spatial mask."""
    selected = da
    if level is not None and "level" in selected.coords:
        selected = selected.sel(level=level, method="nearest")
    if "ensemble" in selected.dims:
        selected = selected.mean("ensemble", skipna=True)
    extra_dims = [dim for dim in selected.dims if dim not in {"time", "lat", "lon"}]
    if extra_dims:
        selected = selected.mean(extra_dims, skipna=True)
    vertices_array = np.asarray(vertices, dtype=float)
    reference = vertices_array[0, 0]
    unwrapped_vertices = reference + (
        (vertices_array[:, 0] - reference + 180.0) % 360.0
    ) - 180.0
    longitude_values = np.asarray(selected.lon.values, dtype=float)
    longitude_unwrapped = reference + (
        (longitude_values - reference + 180.0) % 360.0
    ) - 180.0
    longitude_count = int(
        np.count_nonzero(
            (longitude_unwrapped >= np.min(unwrapped_vertices))
            & (longitude_unwrapped <= np.max(unwrapped_vertices))
        )
    )
    latitude_values = np.asarray(selected.lat.values, dtype=float)
    latitude_count = int(
        np.count_nonzero(
            (latitude_values >= np.min(vertices_array[:, 1]))
            & (latitude_values <= np.max(vertices_array[:, 1]))
        )
    )
    time_count = int(selected.sizes.get("time", 1))
    other_count = int(np.prod([
        selected.sizes[dimension]
        for dimension in selected.dims
        if dimension not in {"time", "lat", "lon"}
    ], dtype=np.int64))
    itemsize = int(np.dtype(selected.dtype).itemsize)
    estimated_bytes = time_count * max(1, latitude_count) * max(1, longitude_count) * other_count * itemsize
    if estimated_bytes > MAX_ANNUAL_SOURCE_BYTES:
        raise DatasetError(
            "The highlight-region annual-cycle source is estimated at "
            f"{estimated_bytes / 1024**3:.2f} GiB. Narrow the map region, "
            "select a lower-resolution field, or disable the annual-cycle plot "
            f"to stay below the {MAX_ANNUAL_SOURCE_BYTES / 1024**3:.2f} GiB safety limit."
        )
    try:
        return spatial_mean_timeseries(selected, vertices, weighting)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc

def normalize_years(year: int | Sequence[int] | None) -> list[int] | None:
    """Accept None, one year, or any collection of years; return a sorted unique list."""
    if year is None:
        return None
    if isinstance(year, (int, np.integer)):
        return [int(year)]
    years = sorted({int(value) for value in year})
    if not years:
        raise DatasetError("No target year is selected.")
    return years


def format_years(year: int | Sequence[int] | None) -> str:
    """Compact label such as '1991-2020, 2023 (n=31)'."""
    years = normalize_years(year)
    if years is None:
        return "all"
    runs, start = [], years[0]
    for previous, current in zip(years, years[1:] + [None]):
        if current != previous + 1:
            runs.append(str(start) if start == previous else f"{start}-{previous}")
            start = current
    label = ", ".join(runs)
    return f"{label} (n={len(years)})" if len(years) > 1 else label


def parse_year_spec(text: str, available: Sequence[int]) -> tuple[list[int], list[int]]:
    """Parse '1991-2020, 2023' into (years present in the data, single years not in the data)."""
    available_years = sorted({int(value) for value in available})
    available_set = set(available_years)
    chosen: set[int] = set()
    unknown: set[int] = set()
    for match in re.finditer(r"(\d+)(?:\s*[-\u2013\u2014~:]\s*(\d+))?", text):
        first = int(match.group(1))
        if match.group(2) is None:
            (chosen if first in available_set else unknown).add(first)
        else:
            low, high = sorted((first, int(match.group(2))))
            chosen.update(value for value in available_years if low <= value <= high)
    return sorted(chosen), sorted(unknown)

def _time_mask(da: xr.DataArray, month: int | None, season: str | tuple[int, ...] | None):
    if "time" not in da.coords:
        raise DatasetError("The selected variable has no time coordinate.")
    mask = xr.ones_like(da.time, dtype=bool)
    if month is not None:
        mask &= da.time.dt.month == month
    if season:
        if isinstance(season, tuple):
            months = season
        else:
            months = (12, 1, 2) if season == "DJF" else (11, 12, 1) if season == "NDJ" else {"MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}[season]
        mask &= da.time.dt.month.isin(months)
    return mask


def _period_mask(
    da: xr.DataArray,
    year: int | Sequence[int] | None,
    month: int | None,
    season: str | tuple[int, ...] | None,
    year_range: tuple[int, int] | None = None,
):
    """Build a time mask, treating NDJ as November-December-January across years."""
    mask = _time_mask(da, month, season)
    years = normalize_years(year)
    if isinstance(season, tuple) or season in {"DJF", "NDJ"}:
        month_number = da.time.dt.month
        calendar_year = da.time.dt.year
        ordered_months = season if isinstance(season, tuple) else ((12, 1, 2) if season == "DJF" else (11, 12, 1))
        start_month = ordered_months[0]
        first_months = tuple(month for month in ordered_months if month >= start_month)
        following_months = tuple(month for month in ordered_months if month < start_month)
        if years is not None:
            return mask & ((month_number.isin(first_months) & calendar_year.isin(years)) | (month_number.isin(following_months) & calendar_year.isin([value + 1 for value in years])))
        if year_range is not None:
            start, end = year_range
            return mask & ((month_number.isin(first_months) & calendar_year.isin(range(start, end + 1))) | (month_number.isin(following_months) & calendar_year.isin(range(start + 1, end + 2))))
    if years is not None:
        mask &= da.time.dt.year.isin(years)
    if year_range is not None:
        mask &= (da.time.dt.year >= year_range[0]) & (da.time.dt.year <= year_range[1])
    return mask


def select_period(
    da: xr.DataArray,
    year: int | Sequence[int] | None,
    month: int | None = None,
    season: str | tuple[int, ...] | None = None,
    level: float | None = None,
    collapse_ensemble: bool = True,
) -> xr.DataArray:
    """Select one year/month or season and average remaining time samples."""
    result = da
    if level is not None and "level" in result.coords:
        result = result.sel(level=level, method="nearest")
    mask = _period_mask(result, year, month, season)
    result = result.where(mask, drop=True)
    if "time" in result.dims:
        if result.sizes["time"] == 0:
            raise DatasetError("No observations match the selected time period.")
        # CMIP6 files can contain isolated missing samples. Excluding those
        # samples preserves valid grid cells instead of turning their complete
        # period mean into NaN.
        result = result.mean("time", skipna=True)
    return _collapse_to_spatial_grid(result, collapse_ensemble=collapse_ensemble)


def subset_analysis_time(
    da: xr.DataArray,
    years: int | Sequence[int] | None = None,
    year_range: tuple[int, int] | None = None,
    include_years: int | Sequence[int] | None = None,
    month: int | None = None,
    season: str | tuple[int, ...] | None = None,
) -> xr.DataArray:
    """Lazily retain only samples needed by a calculation.

    Keeping this operation before detrending, vector derivation, or plotting is
    important for large remote and dask-backed datasets.
    """
    if "time" not in da.dims:
        return da
    mask = _period_mask(da, years, month, season, year_range)
    if include_years is not None:
        mask = mask | _period_mask(da, include_years, month, season)
    result = da.where(mask, drop=True)
    if result.sizes.get("time", 0) == 0:
        raise DatasetError("No observations match the selected analysis period.")
    return result


def _collapse_to_spatial_grid(da: xr.DataArray, collapse_ensemble: bool = True) -> xr.DataArray:
    extra_dims = [
        dimension for dimension in da.dims
        if dimension not in {"lat", "lon"} and (collapse_ensemble or dimension != "ensemble")
    ]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return da.squeeze(drop=True)


def calculate_anomaly(
    da: xr.DataArray,
    year: int | Sequence[int] | None,
    month: int | None = None,
    season: str | tuple[int, ...] | None = None,
    level: float | None = None,
    baseline_years: tuple[int, int] | None = None,
    collapse_ensemble: bool = True,
) -> xr.DataArray:
    """Return target period minus the matching all-year climatology."""
    target = select_period(
        da,
        year,
        month,
        season,
        level,
        collapse_ensemble=collapse_ensemble,
    )
    baseline = da
    if level is not None and "level" in baseline.coords:
        baseline = baseline.sel(level=level, method="nearest")
    mask = _period_mask(baseline, None, month, season, baseline_years)
    baseline = baseline.where(mask, drop=True)
    if baseline.sizes.get("time", 0) == 0:
        raise DatasetError("The climatology baseline contains no matching observations.")
    return _collapse_to_spatial_grid(
        target - baseline.mean("time", skipna=True),
        collapse_ensemble=collapse_ensemble,
    )


def convert_display_units(da: xr.DataArray, anomaly: bool = False) -> xr.DataArray:
    """Convert common climate units to readable plotting units.

    Conversion is deliberately limited to unambiguous CF-style unit strings;
    unknown units are preserved rather than guessed from variable names.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    standard_name = str(da.attrs.get("standard_name", "")).lower()
    long_name = str(da.attrs.get("long_name", "")).lower()
    name = str(da.name or "").lower()
    result = da.copy()
    label = standard_name or long_name or name
    if units in {"k", "degk", "kelvin"}:
        result = da if anomaly else da - 273.15
        result.attrs = da.attrs.copy()
        result.attrs["units"] = "degC"
        result.attrs["conversion_note"] = "Kelvin anomaly expressed as Celsius difference" if anomaly else "Converted from Kelvin"
    elif units in {"pa", "pascal", "pascals"}:
        result = da / 100.0
        result.attrs = da.attrs.copy()
        result.attrs["units"] = "hPa"
        result.attrs["conversion_note"] = "Converted from Pa"
    elif units in {"kg m-2 s-1", "kg m**-2 s**-1", "kg/m2/s", "mm s-1"} and any(token in label for token in ("precip", "rain", "snow")):
        result = da * 86400.0
        result.attrs = da.attrs.copy()
        result.attrs["units"] = "mm day-1"
        result.attrs["conversion_note"] = "Converted to daily water equivalent"
    return result


def select_region(da: xr.DataArray, lon_bounds: tuple[float, float], lat_bounds: tuple[float, float], convention: str) -> xr.DataArray:
    """Select regions, including ranges that wrap around the international date line."""
    if "lon" not in da.coords or "lat" not in da.coords:
        raise DatasetError("The selected variable must have latitude and longitude coordinates.")
    if convention not in {"-180 to 180", "0 to 360"}:
        raise DatasetError(f"Unsupported longitude convention: {convention}")
    lon_min, lon_max = (float(value) for value in lon_bounds)
    if not np.all(np.isfinite([lon_min, lon_max])):
        raise DatasetError("Longitude bounds must be finite.")
    if convention == "-180 to 180":
        source_longitudes = ((np.asarray(da.lon.values, dtype=float) + 180.0) % 360.0) - 180.0
        requested_min, requested_max = lon_min, lon_max
        is_global = lon_min == -180.0 and lon_max == 180.0
    else:
        source_longitudes = np.asarray(da.lon.values, dtype=float) % 360.0
        requested_min, requested_max = lon_min % 360.0, lon_max % 360.0
        is_global = lon_min == 0.0 and lon_max == 360.0
    da = da.assign_coords(lon=source_longitudes).sortby("lon")
    if is_global:
        result = da
    elif requested_min <= requested_max:
        result = da.sel(lon=slice(requested_min, requested_max))
    else:
        west = da.where(da.lon >= requested_min, drop=True)
        east = da.where(da.lon <= requested_max, drop=True)
        result = xr.concat([west, east], dim="lon")
        lon_values = np.asarray(result.lon.values, dtype=float)
        split_index = west.sizes.get("lon", 0)
        lon_values[split_index:] += 360.0
        result = result.assign_coords(lon=lon_values)
    if result.sizes.get("lon", 0) == 0 or result.sizes.get("lat", 0) == 0:
        raise DatasetError("The selected region does not overlap the dataset.")
    result = result.sel(lat=slice(min(lat_bounds), max(lat_bounds)))
    if convention == "0 to 360" and is_global:
        result = result.assign_coords(lon=(result.lon % 360)).sortby("lon")
    return result


def transform_values(
    da: xr.DataArray,
    transformation: str,
    operand: float = 9.8,
    output_units: str = "",
) -> xr.DataArray:
    """Apply an explicit value transformation selected by the user.

    The geopotential conversion follows ``Z / g = geopotential height`` with
    standard gravity ``g = 9.8 m s-2`` by default.
    """
    if transformation == "No conversion":
        if output_units.strip():
            result = da.copy()
            result.attrs["units"] = output_units.strip()
            result.attrs["units_latex"] = output_units.strip()
            return result
        return da
    if transformation == "Automatic CF unit conversion":
        result = convert_display_units(da)
        if output_units.strip():
            result.attrs = result.attrs.copy()
            result.attrs["units"] = output_units.strip()
            result.attrs["units_latex"] = output_units.strip()
        return result
    if not np.isfinite(operand) or (operand == 0 and transformation in {"Divide by a value", "Geopotential (m^2/s^2) / 9.8 -> height (m)"}):
        raise DatasetError("The arithmetic operand must be finite and non-zero for division.")
    if transformation == "Geopotential (m^2/s^2) / 9.8 -> height (m)":
        result = da / operand
        unit = output_units.strip() or "m"
        note = f"Geopotential divided by {operand:g} m s-2"
    elif transformation == "Air-sea CO2 flux (mmolC m^-2 day^-1) * 0.36525 -> Air-sea CO2 flux (molC m^-2 year^-1)":
        result = da * 0.36525
        unit = output_units.strip() or "molC m^-2 year^-1"
        note = f"Air-sea CO2 flux multiplied by {operand:g}"
    elif transformation == "Multiply by a value":
        result = da * operand
        unit = output_units.strip()
        note = f"Multiplied by {operand:g}"
    elif transformation == "Divide by a value":
        result = da / operand
        unit = output_units.strip()
        note = f"Divided by {operand:g}"
    elif transformation == "Add a value":
        result = da + operand
        unit = output_units.strip()
        note = f"Added {operand:g}"
    elif transformation == "Subtract a value":
        result = da - operand
        unit = output_units.strip()
        note = f"Subtracted {operand:g}"
    else:
        raise DatasetError(f"Unknown value transformation: {transformation}")
    result.attrs = da.attrs.copy()
    if unit:
        result.attrs["units"] = unit
        result.attrs["units_latex"] = unit
    result.attrs["conversion_note"] = note
    return result


def detrend_data(
    da: xr.DataArray,
    detrend_type: str = "normalize",
    reference_year: float | None = None,
) -> xr.DataArray:
    """Remove a grid-cell linear trend before temporal aggregation.

    ``normalize`` preserves the value at a reference year, whereas ``anomaly``
    returns residuals from the fitted trend line. A grid cell is processed only
    when every time sample is finite, matching the supplied MATLAB workflow.
    """
    try:
        return _shared_detrend_data(da, detrend_type, reference_year)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc


def _wind_stress_components(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface wind stress using the Large-Pond drag formulation."""
    speed = np.hypot(u, v)
    drag_coefficient = np.where(
        speed <= 1.0,
        0.00218,
        np.where(
            speed <= 3.0,
            (0.62 + 1.56 / np.maximum(speed, 1e-12)) * 0.001,
            np.where(speed < 10.0, 0.00114, (0.49 + 0.065 * speed) * 0.001),
        ),
    )
    air_density = 1.2
    return air_density * drag_coefficient * speed * u, air_density * drag_coefficient * speed * v


def wind_stress_curl(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Calculate wind-stress curl in N m^-3 on a latitude-longitude grid."""
    u, v = xr.align(u, v, join="exact")
    if not {"lat", "lon"}.issubset(u.dims) or u.dims != v.dims:
        raise DatasetError("Wind-stress curl requires aligned U and V latitude-longitude fields.")
    spatial_order = [dimension for dimension in u.dims if dimension not in {"lat", "lon"}]
    ordered_u = u.transpose(*spatial_order, "lat", "lon")
    ordered_v = v.transpose(*spatial_order, "lat", "lon")
    latitudes = np.asarray(ordered_u.lat.values, dtype=float)
    longitudes = np.asarray(ordered_u.lon.values, dtype=float)
    if latitudes.size < 2 or longitudes.size < 2:
        raise DatasetError("Wind-stress curl requires at least two latitude and longitude points.")
    stress_u, stress_v = _wind_stress_components(np.asarray(ordered_u.values, dtype=float), np.asarray(ordered_v.values, dtype=float))
    earth_radius = 6_378_137.0
    lat_radians = np.deg2rad(latitudes)
    x_spacing = earth_radius * np.cos(lat_radians)[:, None] * np.deg2rad(np.gradient(longitudes))[None, :]
    y_spacing = earth_radius * np.deg2rad(np.gradient(latitudes))
    derivative_v_x = np.gradient(stress_v, axis=-1) / x_spacing
    derivative_u_y = np.gradient(stress_u, axis=-2) / y_spacing.reshape((1,) * len(spatial_order) + (latitudes.size, 1))
    curl_values = derivative_v_x - derivative_u_y
    result = xr.DataArray(curl_values, dims=ordered_u.dims, coords=ordered_u.coords, name="wind_stress_curl", attrs={"long_name": "Wind stress curl", "units": "N m-3"})
    return result.transpose(*u.dims)


def vector_magnitude(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Return sqrt(U² + V²) while preserving xarray metadata and laziness."""
    u, v = xr.align(u, v, join="exact")
    if not {"lat", "lon"}.issubset(u.dims) or u.dims != v.dims:
        raise DatasetError("Vector magnitude requires aligned U and V latitude-longitude fields.")
    result = xr.apply_ufunc(np.hypot, u, v, dask="allowed")
    result.name = "vector_magnitude"
    result.attrs = {
        "long_name": f"Magnitude of {u.name or 'U'} and {v.name or 'V'}",
        "units": u.attrs.get("units", ""),
    }
    return result


def _selected_years(da: xr.DataArray, year_range: tuple[int, int] | None) -> list[int]:
    """Return the requested analysis years, constrained to available timestamps."""
    periods = available_periods(da)
    if not periods.years:
        raise DatasetError("Trend and Hovmoller analyses require a time coordinate.")
    if year_range is None:
        return periods.years
    start, end = year_range
    if start > end:
        raise DatasetError("The analysis start year must not exceed the end year.")
    years = [year for year in periods.years if start <= year <= end]
    if len(years) < 2:
        raise DatasetError("At least two valid years are required for a linear trend.")
    return years


def _period_stack(
    da: xr.DataArray,
    year_range: tuple[int, int] | None,
    month: int | None = None,
    season: str | None = None,
    level: float | None = None,
) -> xr.DataArray:
    """Build one spatial field per analysis year using the selected period."""
    years = _selected_years(da, year_range)
    fields: list[xr.DataArray] = []
    valid_years: list[int] = []
    for year in years:
        try:
            fields.append(select_period(da, year, month=month, season=season, level=level))
            valid_years.append(year)
        except DatasetError:
            continue
    if len(fields) < 2:
        raise DatasetError("At least two selected periods contain observations.")
    return xr.concat(fields, dim=xr.IndexVariable("analysis_year", valid_years))


def linear_trend_field(
    series: xr.DataArray,
    confidence_level: float = 0.95,
    significance_method: str = "Student's t",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute grid-cell linear slopes, p-values, and significance masks.

    The first dimension is treated as the time-like regression dimension. The
    slope uses ordinary least squares, while significance is assessed with a
    two-sided Student's t test or Kendall's rank test.
    """
    if not 0.0 < confidence_level < 1.0:
        raise DatasetError("Confidence level must lie strictly between zero and one.")
    if series.ndim < 2:
        raise DatasetError("Trend analysis requires a time dimension and at least one spatial dimension.")
    time_dim = "analysis_year" if "analysis_year" in series.dims else series.dims[0]
    series = series.transpose(time_dim, *[dimension for dimension in series.dims if dimension != time_dim])
    values = np.asarray(series.values, dtype=float)
    sample_count = values.shape[0]
    coordinate_values = np.asarray(series.coords.get(time_dim, np.arange(sample_count)).values, dtype=float)
    if coordinate_values.shape != (sample_count,) or not np.all(np.isfinite(coordinate_values)):
        x = np.arange(sample_count, dtype=float)
    else:
        x = coordinate_values
    flat = values.reshape(sample_count, -1)
    slopes = np.full(flat.shape[1], np.nan)
    p_values = np.full(flat.shape[1], np.nan)
    alpha = 1.0 - confidence_level
    for index in range(flat.shape[1]):
        y = flat[:, index]
        valid = np.isfinite(y)
        if valid.sum() < 3:
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        slopes[index] = slope
        if significance_method == "Mann-Kendall":
            p_values[index] = stats.kendalltau(x_valid, y_valid, nan_policy="omit").pvalue
        else:
            residuals = y_valid - (slope * x_valid + intercept)
            degrees_of_freedom = len(y_valid) - 2
            sum_squared_x = np.sum((x_valid - x_valid.mean()) ** 2)
            standard_error = np.sqrt(np.sum(residuals**2) / degrees_of_freedom / sum_squared_x)
            if standard_error == 0.0:
                p_values[index] = 0.0 if slope != 0.0 else 1.0
            else:
                t_value = slope / standard_error
                p_values[index] = 2.0 * stats.t.sf(abs(t_value), degrees_of_freedom)
    spatial_dims = series.dims[1:]
    spatial_coords = {dimension: series.coords[dimension] for dimension in spatial_dims if dimension in series.coords}
    slope_field = xr.DataArray(slopes.reshape(values.shape[1:]), dims=spatial_dims, coords=spatial_coords, name="linear_trend")
    p_value_field = xr.DataArray(p_values.reshape(values.shape[1:]), dims=spatial_dims, coords=spatial_coords, name="trend_p_value")
    significant_field = (p_value_field < alpha).where(np.isfinite(p_value_field), False).astype(bool).rename("trend_significant")
    return slope_field, p_value_field, significant_field


def calculate_linear_trend(
    da: xr.DataArray,
    year_range: tuple[int, int] | None,
    month: int | None = None,
    season: str | None = None,
    level: float | None = None,
    confidence_level: float = 0.95,
    significance_method: str = "Student's t",
    time_unit: str = "Year",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculate a trend map with slope units per year, month, day, or decade."""
    if time_unit in {"Year", "Decade"}:
        series = _period_stack(da, year_range, month=month, season=season, level=level)
        if time_unit == "Decade":
            coordinate = np.asarray(series.analysis_year.values, dtype=float)
            series = series.assign_coords(
                analysis_year=("analysis_year", (coordinate - coordinate[0]) / 10.0)
            )
    elif time_unit in {"Month", "Day"}:
        if "time" not in da.coords:
            raise DatasetError("Monthly and daily trends require a time coordinate.")
        series = da
        if level is not None and "level" in series.coords:
            series = series.sel(level=level, method="nearest")
        mask = _period_mask(series, None, month, season, year_range)
        series = series.where(mask, drop=True)
        if series.sizes.get("time", 0) < 3:
            raise DatasetError(f"At least three observations are required for a trend per {time_unit.lower()}.")
        extra_dims = [dimension for dimension in series.dims if dimension not in {"time", "lat", "lon"}]
        if extra_dims:
            series = series.mean(extra_dims, skipna=False)
        if time_unit == "Month":
            month_index = series.time.dt.year * 12 + series.time.dt.month
            elapsed_time = (month_index - month_index.isel(time=0)).values.astype(float)
        else:
            elapsed_time = ((series.time - series.time.isel(time=0)) / np.timedelta64(1, "D")).values.astype(float)
        series = series.rename({"time": "analysis_time"}).assign_coords(analysis_time=elapsed_time)
    else:
        raise DatasetError("Trend time basis must be Year, Month, Day, or Decade.")
    slope, p_value, significant = linear_trend_field(series, confidence_level, significance_method)
    source_units = str(da.attrs.get("units", "")).strip()
    slope.attrs = da.attrs.copy()
    slope.attrs["long_name"] = f"Linear trend of {da.attrs.get('long_name', da.name or 'field')}"
    if source_units:
        slope.attrs["units"] = f"{source_units} per {time_unit.lower()}"
        slope.attrs["units_latex"] = f"{source_units}\\,\\mathrm{{per\\ {time_unit.lower()}}}"
    else:
        slope.attrs["units"] = f"value per {time_unit.lower()}"
    p_value.attrs["long_name"] = "Two-sided trend significance p-value"
    significant.attrs["long_name"] = f"Trend significance at {confidence_level:.0%} confidence"
    return slope, p_value, significant


def build_hovmoller(
    da: xr.DataArray,
    year_range: tuple[int, int] | None,
    axis: str,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    convention: str,
    month: int | None = None,
    season: str | None = None,
    level: float | None = None,
) -> xr.DataArray:
    """Build a year-by-space Hovmoller field for the selected period."""
    if axis not in {"lon", "lat"}:
        raise DatasetError("The Hovmoller axis must be 'lon' or 'lat'.")
    years = _selected_years(da, year_range)
    fields: list[xr.DataArray] = []
    valid_years: list[int] = []
    perpendicular = "lat" if axis == "lon" else "lon"
    for year in years:
        try:
            field = select_region(select_period(da, year, month=month, season=season, level=level), lon_bounds, lat_bounds, convention)
        except DatasetError:
            continue
        fields.append(field.mean(perpendicular, skipna=True))
        valid_years.append(year)
    if not fields:
        raise DatasetError("No observations are available for the Hovmoller analysis.")
    return xr.concat(fields, dim=xr.IndexVariable("analysis_year", valid_years)).transpose("analysis_year", axis)


def select_vertical_section(
    da: xr.DataArray,
    year: int | None,
    axis: str,
    coordinate: float | None = None,
    start: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
    month: int | None = None,
    season: str | None = None,
) -> xr.DataArray:
    """Extract a depth-resolved zonal, meridional, or oblique section.

    The returned field has dimensions ``level`` and a horizontal section
    coordinate. Oblique sections use linear interpolation between endpoints.
    """
    if "level" not in da.dims and "level" not in da.coords:
        raise DatasetError("A vertical section requires a level coordinate.")
    selected = da
    mask = _period_mask(selected, year, month, season)
    selected = selected.where(mask, drop=True)
    if "time" in selected.dims:
        if selected.sizes["time"] == 0:
            raise DatasetError("No observations match the selected time period.")
        selected = selected.mean("time", skipna=True)
    extra_dims = [dimension for dimension in selected.dims if dimension not in {"level", "lat", "lon"}]
    if extra_dims:
        selected = selected.mean(extra_dims, skipna=True)
    if axis == "Longitude":
        if coordinate is None:
            raise DatasetError("A latitude coordinate is required for a zonal section.")
        section = selected.sel(lat=coordinate, method="nearest")
        section = section.rename({"lon": "section_coordinate"}).assign_coords(section_coordinate=selected.lon)
    elif axis == "Latitude":
        if coordinate is None:
            raise DatasetError("A longitude coordinate is required for a meridional section.")
        section = selected.sel(lon=coordinate, method="nearest")
        section = section.rename({"lat": "section_coordinate"}).assign_coords(section_coordinate=selected.lat)
    elif axis == "Oblique":
        if start is None or end is None:
            raise DatasetError("Oblique sections require start and end coordinates.")
        point_count = max(selected.sizes.get("lat", 1), selected.sizes.get("lon", 1))
        path = np.linspace(0.0, 1.0, point_count)
        lat_path = start[1] + path * (end[1] - start[1])
        lon_path = start[0] + path * (end[0] - start[0])
        section = selected.interp(lat=xr.DataArray(lat_path, dims="section_coordinate"), lon=xr.DataArray(lon_path, dims="section_coordinate"))
        section = section.assign_coords(section_coordinate=np.arange(point_count, dtype=float))
    else:
        raise DatasetError("Section orientation must be Longitude, Latitude, or Oblique.")
    return section.transpose("level", "section_coordinate")

def smart_mask_fill_values(obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Automatically detects and replaces undocumented fill/missing values (e.g., -9999, 32767) 
    with NaN across an xarray Dataset or DataArray.
    
    This function safe-guards against memory explosions by leveraging Dask lazy evaluation 
    and only triggers masks when specific fill values are actively present in the data.
    """
    if isinstance(obj, xr.Dataset):
        return obj.map(smart_mask_fill_values, keep_attrs=True)
    # These sentinels are applied lazily; unlike a global min/max scan this does
    # not force a large dask-backed dataset into memory.
    fills = set((-99999, -9999, -999, -32767, 9999, 99999, 65535))
    for key in ("_FillValue", "missing_value"):
        value = obj.attrs.get(key)
        if value is not None:
            values = np.asarray(value).ravel()
            fills.update(float(item) for item in values if np.isfinite(item))
    result = obj
    for value in fills:
        result = result.where(result != value)
    valid_min = obj.attrs.get("valid_min")
    valid_max = obj.attrs.get("valid_max")
    if valid_min is not None:
        result = result.where(result >= float(valid_min))
    if valid_max is not None:
        result = result.where(result <= float(valid_max))
    return result


def mask_invalid_values(
    obj: Union[xr.Dataset, xr.DataArray],
    lower: float | None = None,
    upper: float | None = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Replace metadata/sentinel or user-threshold values with NaN lazily."""
    result = smart_mask_fill_values(obj)
    if lower is not None:
        result = result.where(result >= lower)
    if upper is not None:
        result = result.where(result <= upper)
    return result
