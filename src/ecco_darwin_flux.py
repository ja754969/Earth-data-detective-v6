"""Secure, metadata-first streaming access to ECCO-Darwin MDS flux output."""

from __future__ import annotations

import argparse
from functools import lru_cache
import logging
import os
import re
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import urljoin, urlparse

import certifi
import dask.array as dask_array
from dask import delayed
import fsspec
import numpy as np
import xarray as xr
import aiohttp
from aiohttp import TCPConnector
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)
SECONDS_PER_YEAR = 365.25 * 86400.0
CARBON_MOLAR_MASS_G_PER_MOL = 12.011
PICO_PETAGRAMS_PER_GRAM = 1e-15
ALLOWED_HOST = "data.nas.nasa.gov"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
# A multi-year annual-cycle calculation can schedule hundreds of delayed MDS
# reads.  Keep requests deliberately modest for the public NAS endpoint.
REMOTE_READ_SEMAPHORE = threading.BoundedSemaphore(value=2)


class ConfigurationError(ValueError):
    """Raised when a user-supplied ECCO configuration is unsafe or invalid."""


class RemoteDataError(OSError):
    """Raised when a public ECCO object remains unavailable after retries."""


class _LinkParser(HTMLParser):
    """Extract links from a simple Apache-style directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class EccoConfig(BaseModel):
    """Validated parameters for one remote ECCO-Darwin extraction."""

    model_config = ConfigDict(extra="forbid")

    flux_url: str = Field(min_length=1)
    grid_url: str = Field(min_length=1)
    ref_datetime: datetime
    dt_seconds: float = Field(gt=0)
    start_date: datetime
    end_date: datetime
    min_lon: float = Field(ge=-180, le=360)
    max_lon: float = Field(ge=-180, le=360)
    min_lat: float = Field(ge=-90, le=90)
    max_lat: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def validate_ranges(self) -> "EccoConfig":
        """Validate URLs, date ordering, and latitude ordering."""
        for value in (self.flux_url, self.grid_url):
            _validate_remote_url(value)
        if self.ref_datetime.tzinfo is None:
            self.ref_datetime = self.ref_datetime.replace(tzinfo=timezone.utc)
        if self.start_date.tzinfo is None:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date.tzinfo is None:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date.")
        if self.min_lat > self.max_lat:
            raise ValueError("min_lat must not be greater than max_lat.")
        return self


@dataclass(frozen=True)
class MdsMetadata:
    """Parsed dimensions, precision, and optional model time interval."""

    shape: tuple[int, ...]
    dtype: np.dtype
    time_interval: tuple[float, float] | None


def _validate_remote_url(url: str) -> str:
    """Allow only HTTPS URLs on the NASA data host."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or parsed.hostname != ALLOWED_HOST:
        raise ConfigurationError(
            f"Remote URL must use https://{ALLOWED_HOST}; received {url!r}."
        )
    return url


def _ssl_context() -> ssl.SSLContext:
    """Build a certificate-verifying TLS context that requires TLS 1.2 or newer."""
    context = ssl.create_default_context(cafile=certifi.where())
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


async def _secure_http_client(loop=None, **kwargs):
    """Create an aiohttp client bound to fsspec's event loop and TLS context."""
    connector = TCPConnector(ssl=_ssl_context(), loop=loop)
    return aiohttp.ClientSession(connector=connector, loop=loop, **kwargs)


@lru_cache(maxsize=1)
def _filesystem() -> fsspec.AbstractFileSystem:
    """Create the HTTPS filesystem used for all remote reads."""
    return fsspec.filesystem("https", get_client=_secure_http_client)


def _read_remote(url: str) -> bytes:
    """Read one validated remote object into memory and never create a local file."""
    _validate_remote_url(url)
    filesystem = _filesystem()
    # The public NAS directory is rate-limited.  A full monthly extraction makes
    # hundreds of small metadata requests, so a transient 429 must not be
    # represented as a missing object (and silently turn into a truncated series).
    retryable_statuses = {429, 500, 502, 503, 504}
    with REMOTE_READ_SEMAPHORE:
        for attempt in range(7):
            try:
                with filesystem.open(url, mode="rb", block_size=0) as remote_file:
                    return remote_file.read()
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    raise FileNotFoundError(url) from exc
                if exc.status not in retryable_statuses:
                    raise
                if attempt == 6:
                    raise RemoteDataError(
                        f"NASA ECCO returned HTTP {exc.status} after seven attempts: {url}"
                    ) from exc
            except FileNotFoundError as exc:
                # fsspec translates an HTTP failure during its metadata (HEAD/GET)
                # request into FileNotFoundError.  Preserve a real 404, but retry
                # a 429/5xx response retained anywhere in its exception chain.
                status = _http_status_from_exception(exc)
                if status not in retryable_statuses:
                    raise
                if attempt == 6:
                    raise RemoteDataError(
                        f"NASA ECCO returned HTTP {status} after seven attempts: {url}"
                    ) from exc
            time.sleep(min(30.0, 1.5 * (2 ** attempt)))
    raise RuntimeError("Unreachable remote-read retry state.")


def _http_status_from_exception(exc: BaseException) -> int | None:
    """Return an aiohttp status embedded by fsspec in an exception chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, aiohttp.ClientResponseError):
            return current.status
        current = current.__cause__ or current.__context__
    return None


def _directory_links(url: str) -> list[str]:
    """List links from a validated remote directory without downloading data files."""
    directory = url.rstrip("/") + "/"
    html = _read_remote(directory).decode("utf-8", errors="strict")
    parser = _LinkParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        candidate = urljoin(directory, href)
        if urlparse(candidate).hostname == ALLOWED_HOST:
            links.append(candidate)
    return sorted(set(links))


def parse_mds_metadata(text: str) -> MdsMetadata:
    """Parse MITgcm ``.meta`` dimensions, precision, and time interval."""
    dimensions: list[int] = []
    dim_section = re.search(r"dimList\s*=\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if dim_section is None:
        raise ValueError("MDS metadata does not contain dimList.")
    dimension_values = [
        int(value) for value in re.findall(r"[-+]?\d+", dim_section.group(1))
    ]
    if len(dimension_values) % 3:
        raise ValueError("MDS dimList must contain groups of three integers.")
    for size, start, end in zip(
        dimension_values[0::3], dimension_values[1::3], dimension_values[2::3]
    ):
        if size <= 0 or end < start or end - start + 1 != size:
            raise ValueError("MDS metadata contains an invalid dimension triple.")
        dimensions.append(size)
    if not dimensions:
        raise ValueError("MDS metadata contains no usable dimensions.")
    precision_match = re.search(
        r"dataprec\s*=\s*\[\s*['\"]([^'\"]+)['\"]",
        text,
        re.IGNORECASE,
    )
    if precision_match is None:
        raise ValueError("MDS metadata does not contain dataprec.")
    precision = precision_match.group(1).lower()
    if precision in {"float32", "real*4"}:
        dtype = np.dtype(">f4")
    elif precision in {"float64", "real*8"}:
        dtype = np.dtype(">f8")
    else:
        raise ValueError(f"Unsupported MDS dataprec {precision!r}.")
    interval_match = re.search(
        rf"timeInterval\s*=\s*\[\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\]",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    interval = None
    if interval_match:
        interval = tuple(
            float(value.replace("D", "E").replace("d", "e"))
            for value in interval_match.groups()
        )
    return MdsMetadata(tuple(dimensions), dtype, interval)


def _read_mds(url: str) -> tuple[np.ndarray, MdsMetadata]:
    """Read and decode one MDS object using its adjacent metadata."""
    data_url = url[:-5] + ".data" if url.endswith(".meta") else url
    meta_url = data_url[:-5] + ".meta"
    metadata = parse_mds_metadata(_read_remote(meta_url).decode("ascii"))
    raw = _read_remote(data_url)
    values = np.frombuffer(raw, dtype=metadata.dtype)
    expected = int(np.prod(metadata.shape))
    if values.size != expected:
        raise ValueError(f"MDS payload has {values.size} values; expected {expected}.")
    return values.reshape(metadata.shape, order="F"), metadata


def _time_from_metadata_or_name(
    data_url: str, metadata: MdsMetadata, config: EccoConfig
) -> datetime:
    """Resolve a file timestamp from timeInterval or an iteration in its name."""
    if metadata.time_interval is not None:
        seconds = metadata.time_interval[1]
    else:
        match = re.search(r"(\d+)(?=\.data$)", PurePosixPath(urlparse(data_url).path).name)
        if match is None:
            raise ValueError(f"Unable to derive a timestamp from {data_url!r}.")
        seconds = int(match.group(1)) * config.dt_seconds
    return config.ref_datetime + timedelta(seconds=seconds)


def _normalise_longitudes(values: np.ndarray, convention: str) -> np.ndarray:
    """Convert longitudes to the requested 0-360 or -180-180 convention."""
    if convention == "0 to 360":
        return np.mod(values, 360.0)
    return (values + 180.0) % 360.0 - 180.0


def _spatial_mask(
    longitude: np.ndarray,
    latitude: np.ndarray,
    config: EccoConfig,
) -> np.ndarray:
    """Create a date-line-aware mask for either longitude convention."""
    convention = "0 to 360" if config.min_lon >= 0 and config.max_lon >= 0 else "-180 to 180"
    lon = _normalise_longitudes(longitude, convention)
    lower, upper = config.min_lon, config.max_lon
    if convention == "0 to 360":
        lower, upper = lower % 360.0, upper % 360.0
    longitude_selected = (
        np.ones(lon.shape, dtype=bool)
        if (lower == 0 and upper in {360, 0})
        else (lon >= lower) & (lon <= upper)
        if lower <= upper
        else (lon >= lower) | (lon <= upper)
    )
    return longitude_selected & (latitude >= config.min_lat) & (latitude <= config.max_lat)


def _flux_records(config: EccoConfig) -> list[tuple[str, datetime]]:
    """Return available monthly MDS records in the configured time range.

    The small adjacent ``.meta`` records are read before any binary payload.
    Their ``timeInterval`` values are authoritative; filename iterations are
    used only as a compatibility fallback for legacy metadata without a time
    interval.  This prevents treating a model time-step index as a month count.
    """
    records: list[tuple[str, datetime]] = []
    for data_url in _directory_links(config.flux_url):
        if not data_url.endswith(".data"):
            continue
        meta_url = data_url[:-5] + ".meta"
        try:
            metadata = parse_mds_metadata(_read_remote(meta_url).decode("ascii"))
        except (FileNotFoundError, RemoteDataError) as exc:
            # Preserve a usable, lazy viewer during a transient metadata outage.
            # ``dt_seconds`` is the ECCO model step (1200 s), not a month length.
            LOGGER.warning(
                "ECCO metadata unavailable; using filename iteration fallback: %s (%s)",
                meta_url,
                exc,
            )
            metadata = MdsMetadata((), np.dtype(">f4"), None)
        try:
            timestamp = _time_from_metadata_or_name(data_url, metadata, config)
        except OverflowError as exc:
            raise ValueError(
                f"ECCO timestamp overflow for {data_url}; check ref_datetime and dt_seconds."
            ) from exc
        if config.start_date <= timestamp <= config.end_date:
            records.append((data_url, timestamp))
    return sorted(records, key=lambda item: item[1])


def _regular_grid_indices(
    longitude: np.ndarray,
    latitude: np.ndarray,
    resolution_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a curvilinear LLC grid to a compact regular lat/lon grid.

    Longitude copies either side of the date line make nearest-neighbour lookup
    continuous for source grids stored with the date line at an array edge.
    """
    if not 0 < resolution_degrees <= 5:
        raise ValueError("resolution_degrees must be in the interval (0, 5].")
    valid = np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(valid):
        raise ValueError("ECCO grid has no finite longitude/latitude coordinates.")
    source_lon = np.mod(longitude[valid], 360.0)
    source_lat = latitude[valid]
    source_flat = np.flatnonzero(valid)
    points = np.column_stack((
        np.tile(source_lat, 3),
        np.concatenate((source_lon - 360.0, source_lon, source_lon + 360.0)),
    ))
    source_lookup = np.tile(source_flat, 3)
    target_lat = np.arange(-90.0 + resolution_degrees / 2, 90.0, resolution_degrees)
    target_lon = np.arange(-180.0 + resolution_degrees / 2, 180.0, resolution_degrees)
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    _, nearest = cKDTree(points).query(
        np.column_stack((lat_grid.ravel(), np.mod(lon_grid.ravel(), 360.0)))
    )
    return source_lookup[nearest], target_lat, target_lon


def _read_flux_cells(data_url: str, source_indices: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Read one MDS field, retaining only cells needed by the display grid.

    A public-server outage for a single month becomes an explicit all-missing
    slice, allowing a long annual-cycle request to retain its other months.
    The warning records the omitted timestamp without manufacturing data.
    """
    try:
        values, _ = _read_mds(data_url)
    except (FileNotFoundError, RemoteDataError) as exc:
        LOGGER.warning("ECCO field unavailable; using an all-missing slice: %s (%s)", data_url, exc)
        return np.full(target_shape, np.nan, dtype=np.float32)
    return np.asarray(values).reshape(-1, order="C")[source_indices].reshape(target_shape)


def open_flux_grid(
    config: EccoConfig,
    resolution_degrees: float = 1.0,
) -> xr.Dataset:
    """Expose ECCO-Darwin CO2 flux as a lazy regular-grid xarray dataset.

    Each time step is a delayed remote MDS read.  Consequently, selecting one
    month loads one binary field rather than all available monthly files.
    """
    grid_files = _directory_links(config.grid_url)
    grid_urls = {
        PurePosixPath(urlparse(url).path).name.lower(): url
        for url in grid_files
        if url.endswith(".data")
    }
    try:
        longitude, _ = _read_mds(grid_urls["xc.data"])
        latitude, _ = _read_mds(grid_urls["yc.data"])
    except KeyError as exc:
        raise ValueError("Grid directory must expose XC.data and YC.data.") from exc
    source_indices, target_lat, target_lon = _regular_grid_indices(
        longitude, latitude, resolution_degrees
    )
    records = _flux_records(config)
    if not records:
        raise ValueError("No flux records overlap the requested date range.")
    target_shape = (target_lat.size, target_lon.size)
    chunks = [
        dask_array.from_delayed(
            delayed(_read_flux_cells)(url, source_indices, target_shape),
            shape=target_shape,
            dtype=np.dtype(">f4"),
        )
        for url, _ in records
    ]
    times = np.asarray(
        [stamp.astimezone(timezone.utc).replace(tzinfo=None) for _, stamp in records],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        data_vars={
            "co2_flux": xr.DataArray(
                dask_array.stack(chunks, axis=0),
                dims=("time", "lat", "lon"),
                attrs={
                    "long_name": "ECCO-Darwin air-sea CO2 flux",
                    "units": "mmol C m-2 s-1",
                    "source_grid": "ECCO LLC270",
                    "resampling": "nearest-neighbour to regular latitude-longitude grid",
                },
            )
        },
        coords={"time": times, "lat": target_lat, "lon": target_lon},
        attrs={
            "source": config.flux_url,
            "grid_source": config.grid_url,
            "processing": "Lazy MDS streaming; each selected month is fetched on demand.",
            "display_resolution_degrees": resolution_degrees,
        },
    )


def extract_flux_timeseries(config: EccoConfig) -> xr.Dataset:
    """Stream selected ECCO-Darwin flux records and return an area-weighted series."""
    grid_files = _directory_links(config.grid_url)
    grid_urls = {
        PurePosixPath(urlparse(url).path).name.lower(): url
        for url in grid_files
        if url.endswith(".data")
    }
    try:
        longitude, _ = _read_mds(grid_urls["xc.data"])
        latitude, _ = _read_mds(grid_urls["yc.data"])
        area, _ = _read_mds(grid_urls["rac.data"])
    except KeyError as exc:
        raise ValueError("Grid directory must expose XC.data, YC.data, and RAC.data.") from exc
    mask = _spatial_mask(longitude, latitude, config) & np.isfinite(area) & (area > 0)
    selected_area = area[mask]
    if selected_area.size == 0:
        raise ValueError("The requested region does not overlap the ECCO grid.")
    flux_urls = [
        url for url in _directory_links(config.flux_url)
        if url.endswith(".data")
    ]
    times: list[datetime] = []
    mean_fluxes: list[float] = []
    integrated_fluxes: list[float] = []
    for data_url in flux_urls:
        meta_url = data_url[:-5] + ".meta"
        try:
            metadata = parse_mds_metadata(_read_remote(meta_url).decode("ascii"))
        except FileNotFoundError:
            LOGGER.warning("Skipping %s because its metadata object is unavailable.", data_url)
            continue
        timestamp = _time_from_metadata_or_name(data_url, metadata, config)
        if not (config.start_date <= timestamp <= config.end_date):
            LOGGER.debug("Skipping %s after metadata-only date filtering.", data_url)
            continue
        try:
            flux, _ = _read_mds(data_url)
        except FileNotFoundError:
            LOGGER.warning("Skipping %s because its data object is unavailable.", data_url)
            continue
        values = flux[mask]
        valid = np.isfinite(values)
        if not np.any(valid):
            LOGGER.warning("Skipping %s because the selected cells are all missing.", data_url)
            continue
        weights = selected_area[valid]
        raw_flux = values[valid]
        mean_mol_m2_yr = float(np.sum(raw_flux * weights) / np.sum(weights) / 1000.0 * SECONDS_PER_YEAR)
        integrated_pg_yr = float(
            np.sum(raw_flux * weights) / 1000.0 * SECONDS_PER_YEAR
            * CARBON_MOLAR_MASS_G_PER_MOL * PICO_PETAGRAMS_PER_GRAM
        )
        times.append(timestamp)
        mean_fluxes.append(mean_mol_m2_yr)
        integrated_fluxes.append(integrated_pg_yr)
    if not times:
        raise ValueError("No flux records overlap the requested date range.")
    time_values = np.asarray(
        [value.astimezone(timezone.utc).replace(tzinfo=None) for value in times],
        dtype="datetime64[ns]",
    )
    order = np.argsort(time_values)
    result = xr.Dataset(
        data_vars={
            "area_weighted_flux": xr.DataArray(
                np.asarray(mean_fluxes)[order],
                dims=("time",),
                attrs={
                    "long_name": "Area-weighted air-sea CO2 flux density",
                    "units": "mol C m-2 yr-1",
                },
            ),
            "regional_carbon_flux": xr.DataArray(
                np.asarray(integrated_fluxes)[order],
                dims=("time",),
                attrs={
                    "long_name": "Integrated regional air-sea carbon flux",
                    "units": "Pg C yr-1",
                },
            ),
        },
        coords={"time": time_values[order]},
        attrs={
            "source": config.flux_url,
            "grid_source": config.grid_url,
            "Conventions": "CF-1.10",
            "processing": "MITgcm MDS metadata-first HTTPS streaming; no data files written locally",
        },
    )
    return result


def load_environment_config(overrides: dict[str, object] | None = None) -> EccoConfig:
    """Load ECCO configuration from the project ``.env`` and optional overrides."""
    project_env = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
    load_dotenv(project_env)
    names = (
        "flux_url", "grid_url", "ref_datetime", "dt_seconds", "start_date",
        "end_date", "min_lon", "max_lon", "min_lat", "max_lat",
    )
    values = {name: os.environ.get(f"ECCO_{name.upper()}") for name in names}
    if overrides:
        values.update(overrides)
    for name in tuple(values):
        if isinstance(values[name], str):
            values[name] = values[name].strip()
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ConfigurationError(
            "Missing ECCO configuration values: " + ", ".join(missing)
        )
    return EccoConfig(**values)


def _load_config() -> EccoConfig:
    """Load environment defaults, then apply command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "flux_url", "grid_url", "ref_datetime", "dt_seconds", "start_date",
        "end_date", "min_lon", "max_lon", "min_lat", "max_lat",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", default=None)
    overrides = {
        name: value
        for name, value in vars(parser.parse_args()).items()
        if value is not None
    }
    try:
        return load_environment_config(overrides)
    except ConfigurationError as exc:
        parser.error(str(exc))


def main() -> int:
    """Run the command-line extraction and print an xarray-compatible summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = extract_flux_timeseries(_load_config())
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
