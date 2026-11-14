"""Secure, extensible remote geospatial data ingestion.

This module deliberately stops at data ingestion.  It returns spatially
resolved ``xarray.DataArray`` objects and derives area-weighted time series;
statistical analysis belongs to the application's analysis layer.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import ssl
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

import certifi
import numpy as np
import xarray as xr
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ecco_darwin_flux import EccoConfig, extract_flux_timeseries, open_flux_grid

LOGGER = logging.getLogger(__name__)
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


class RemoteIngestionError(RuntimeError):
    """Raised when a configured remote source cannot be opened safely."""


class SecurityPolicyError(RemoteIngestionError):
    """Raised when a request conflicts with the local security policy."""


class DatasetSpec(BaseModel):
    """Non-secret metadata for a registered dataset."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    description: str = ""
    protocol: str
    endpoint: str | None = None
    collection: str | None = None
    variable: str | None = None
    longitude_name: str | None = None
    latitude_name: str | None = None
    time_name: str | None = None
    chunks: dict[str, int] | Literal["auto"] = "auto"
    reference_datetime: str | None = None
    dt_seconds: float | None = None
    unit_scale: float = 1.0
    unit_offset: float = 0.0
    output_units: str | None = None
    available_start: date | None = None
    available_end: date | None = None
    asset_key: str | None = None
    ee_scale: float | None = Field(default=None, gt=0)
    ee_crs: str | None = None

    @field_validator("endpoint")
    @classmethod
    def endpoint_is_https_or_cloud_uri(cls, value: str | None) -> str | None:
        """Reject unsupported endpoint schemes before a provider uses them."""
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme.lower() not in {"https", "gs", "az"}:
            raise ValueError("endpoint must use https, gs, or az scheme")
        return value

    @model_validator(mode="after")
    def validate_available_dates(self) -> "DatasetSpec":
        """Reject inverted optional availability bounds in registry metadata."""
        if self.available_start and self.available_end and self.available_start > self.available_end:
            raise ValueError("available_start must not be after available_end")
        return self


class DatasetRegistry(BaseModel):
    """Validated mapping loaded from the non-secret dataset registry."""

    model_config = ConfigDict(extra="forbid")
    version: int = 1
    datasets: dict[str, DatasetSpec]


class SecurityPolicy(BaseModel):
    """Environment-level transport and identity policy, never dataset secrets."""

    model_config = ConfigDict(extra="forbid")
    allowed_domains: frozenset[str]
    earthdata_auth_strategy: Literal["netrc"] = "netrc"
    azure_auth_mode: Literal["default_credential", "planetary_computer"] = "default_credential"
    gcp_project_id: str | None = None
    gee_project: str | None = None
    ee_opt_url: str | None = None

    @field_validator("allowed_domains")
    @classmethod
    def domains_are_hostnames(cls, values: frozenset[str]) -> frozenset[str]:
        """Require explicit DNS hostnames rather than permissive wildcards."""
        if not values:
            raise ValueError("ALLOWED_DOMAINS must contain at least one hostname")
        invalid = [item for item in values if not item or "/" in item or "*" in item]
        if invalid:
            raise ValueError("ALLOWED_DOMAINS must contain exact hostnames only")
        return frozenset(item.lower() for item in values)


class DataRequest(BaseModel):
    """Validated user request shared by every provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    bbox: tuple[float, float, float, float]
    date_range: tuple[date, date]
    longitude_convention: Literal["-180_180", "0_360"] = "-180_180"

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Validate longitude and latitude bounds while allowing date-line wrapping."""
        west, east, south, north = value
        if not all(np.isfinite(item) for item in value):
            raise ValueError("bbox values must be finite")
        if not (-360 <= west <= 360 and -360 <= east <= 360):
            raise ValueError("longitude bounds must be within [-360, 360]")
        if not (-90 <= south < north <= 90):
            raise ValueError("latitude bounds must satisfy -90 <= south < north <= 90")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> "DataRequest":
        """Reject inverted temporal ranges."""
        if self.date_range[0] > self.date_range[1]:
            raise ValueError("start date must not be after end date")
        return self


def load_security_policy(env_path: str | Path | None = None) -> SecurityPolicy:
    """Load only policy and identity mode values from an ignored ``.env`` file."""
    if env_path is not None:
        load_dotenv(env_path, override=False)
    raw_domains = os.environ.get("ALLOWED_DOMAINS", "")
    if not raw_domains.strip():
        raise SecurityPolicyError(
            "ALLOWED_DOMAINS is not configured. Add exact HTTPS hostnames to the "
            "project .env (copy .env.example); remote access remains blocked."
        )
    return SecurityPolicy(
        allowed_domains=frozenset(item.strip() for item in raw_domains.split(",") if item.strip()),
        earthdata_auth_strategy=os.environ.get("EARTHDATA_AUTH_STRATEGY", "netrc"),
        azure_auth_mode=os.environ.get("AZURE_AUTH_MODE", "default_credential"),
        gcp_project_id=os.environ.get("GCP_PROJECT_ID") or None,
        gee_project=os.environ.get("GEE_PROJECT") or None,
        ee_opt_url=os.environ.get("EE_OPT_URL") or None,
    )


def load_dataset_registry(config_path: str | Path) -> DatasetRegistry:
    """Load the portable, non-secret YAML dataset registry."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return DatasetRegistry.model_validate(raw)


def validate_remote_url(url: str, policy: SecurityPolicy) -> str:
    """Apply an exact domain allowlist before any HTTP(S) network I/O."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or not hostname or hostname not in policy.allowed_domains:
        raise SecurityPolicyError("Remote endpoint is not an allowed HTTPS domain.")
    if parsed.username or parsed.password or parsed.fragment:
        raise SecurityPolicyError("Remote endpoint must not contain credentials or fragments.")
    return url


def tls_context() -> ssl.SSLContext:
    """Build a TLS 1.2+ context using the certifi root certificate bundle."""
    context = ssl.create_default_context(cafile=certifi.where())
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def _require(module_name: str, package_name: str) -> Any:
    """Load a provider dependency only when that provider is selected."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RemoteIngestionError(
            f"Provider dependency '{module_name}' is unavailable. Install '{package_name}' in the OC environment."
        ) from exc


def _coordinate_name(data: xr.DataArray, preferred: str | None, aliases: tuple[str, ...]) -> str:
    """Resolve one coordinate name without imposing a provider-specific schema."""
    candidates = (preferred,) if preferred else ()
    for name in (*candidates, *aliases):
        if name and (name in data.coords or name in data.dims):
            return name
    raise RemoteIngestionError(f"Could not identify any of {aliases} coordinates.")


def _configured_variable(dataset: xr.Dataset, variable_name: str) -> xr.DataArray:
    """Return one registered variable with an actionable schema-mismatch error."""
    if variable_name in dataset.data_vars:
        return dataset[variable_name]
    available = ", ".join(sorted(dataset.data_vars)) or "none"
    raise RemoteIngestionError(
        f"Configured variable '{variable_name}' is unavailable in this granule. "
        f"Available data variables: {available}. Update datasets.yaml."
    )


def _normalise_grid(data: xr.DataArray, spec: DatasetSpec, request: DataRequest) -> xr.DataArray:
    """Standardise geographic axes, date range, longitude convention, and units."""
    latitude = _coordinate_name(data, spec.latitude_name, ("lat", "latitude", "y"))
    longitude = _coordinate_name(data, spec.longitude_name, ("lon", "longitude", "x"))
    time = _coordinate_name(data, spec.time_name, ("time", "date", "datetime"))
    renamed = data.rename({latitude: "lat", longitude: "lon", time: "time"})
    if renamed["lat"].ndim != 1 or renamed["lon"].ndim != 1:
        raise RemoteIngestionError("This provider requires a rectilinear one-dimensional latitude/longitude grid.")
    result = renamed.sortby("lat")
    lon = result["lon"]
    if request.longitude_convention == "0_360":
        result = result.assign_coords(lon=np.mod(lon, 360)).sortby("lon")
        west, east = np.mod(request.bbox[0], 360), np.mod(request.bbox[1], 360)
        limit = 360.0
    else:
        result = result.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
        west = ((request.bbox[0] + 180) % 360) - 180
        east = ((request.bbox[1] + 180) % 360) - 180
        limit = 180.0
    south, north = request.bbox[2:]
    result = result.sel(lat=slice(south, north), time=slice(str(request.date_range[0]), str(request.date_range[1])))
    if west <= east:
        result = result.sel(lon=slice(west, east))
    else:
        # A wrapped bbox is two safe, compact slices—not a near-global selection.
        left = result.sel(lon=slice(west, limit))
        right = result.sel(lon=slice(-limit if limit == 180 else 0, east))
        result = xr.concat((left, right), dim="lon")
    if result.sizes.get("lat", 0) == 0 or result.sizes.get("lon", 0) == 0:
        raise RemoteIngestionError("The requested bbox does not overlap the dataset grid.")
    result = result * spec.unit_scale + spec.unit_offset
    result.attrs = dict(data.attrs)
    if spec.output_units:
        result.attrs["units"] = spec.output_units
    result.attrs["longitude_convention"] = request.longitude_convention
    return result


def _monthly_start(data: xr.DataArray) -> xr.DataArray:
    """Aggregate duplicate sub-monthly observations and label each month at its start."""
    return data.resample(time="MS").mean(keep_attrs=True)


def _cmr_bounding_boxes(request: DataRequest) -> tuple[tuple[float, float, float, float], ...]:
    """Translate the internal bbox contract to CMR's ``west,south,east,north`` order.

    CMR uses the conventional ``[-180, 180]`` longitude domain and cannot
    represent a wrapped rectangle in one request.  A date-line selection is
    therefore queried as two non-wrapping rectangles.
    """
    west, east, south, north = request.bbox
    if abs(east - west) >= 360:
        return ((-180.0, south, 180.0, north),)
    cmr_west = ((west + 180.0) % 360.0) - 180.0
    cmr_east = ((east + 180.0) % 360.0) - 180.0
    if cmr_west <= cmr_east:
        return ((cmr_west, south, cmr_east, north),)
    return (
        (cmr_west, south, 180.0, north),
        (-180.0, south, cmr_east, north),
    )


class BaseDataSourceProvider(ABC):
    """Provider adapter contract for secure spatial grid retrieval."""

    provider_name: ClassVar[str]

    @abstractmethod
    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Return a lazy, spatially resolved standardized data array."""

    def fetch_timeseries(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Area-weight the standardized grid and return Month-Start observations."""
        grid = self.fetch_grid(spec, request, policy)
        weights = np.cos(np.deg2rad(grid["lat"])).clip(min=0)
        series = grid.weighted(weights).mean(("lat", "lon"), skipna=True)
        return _monthly_start(series)


class ProviderRegistry:
    """Factory registry that keeps the pipeline closed to provider-specific imports."""

    _providers: dict[str, Callable[[], BaseDataSourceProvider]] = {}

    @classmethod
    def register(cls, provider_type: str) -> Callable[[type[BaseDataSourceProvider]], type[BaseDataSourceProvider]]:
        """Register an adapter class under its YAML provider identifier."""
        def decorator(provider_class: type[BaseDataSourceProvider]) -> type[BaseDataSourceProvider]:
            cls._providers[provider_type] = provider_class
            return provider_class
        return decorator

    @classmethod
    def create(cls, provider_type: str) -> BaseDataSourceProvider:
        """Instantiate a registered provider without exposing registry internals."""
        try:
            return cls._providers[provider_type]()
        except KeyError as exc:
            raise RemoteIngestionError(f"No provider registered for '{provider_type}'.") from exc


@ProviderRegistry.register("netcdf_https")
class HTTPSNetCDFProvider(BaseDataSourceProvider):
    """Lazy HTTPS/OPeNDAP NetCDF adapter for configured public services."""

    provider_name = "netcdf_https"

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Open only an allowlisted HTTPS NetCDF endpoint and subset it lazily."""
        if not spec.endpoint or not spec.variable:
            raise RemoteIngestionError("netcdf_https requires endpoint and variable metadata.")
        endpoint = validate_remote_url(spec.endpoint, policy)
        dataset = xr.open_dataset(endpoint, chunks=spec.chunks)
        return _normalise_grid(dataset[spec.variable], spec, request)


@ProviderRegistry.register("mitgcm_mds")
class MITgcmMDSProvider(BaseDataSourceProvider):
    """ECCO-Darwin MDS adapter using the existing metadata-first streaming reader."""

    provider_name = "mitgcm_mds"

    def _config(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> EccoConfig:
        if not spec.endpoint or not spec.collection or not spec.reference_datetime or spec.dt_seconds is None:
            raise RemoteIngestionError("mitgcm_mds requires endpoint, collection, reference_datetime, and dt_seconds.")
        validate_remote_url(spec.endpoint, policy)
        validate_remote_url(spec.collection, policy)
        return EccoConfig(
            flux_url=spec.endpoint,
            grid_url=spec.collection,
            ref_datetime=spec.reference_datetime,
            dt_seconds=spec.dt_seconds,
            start_date=request.date_range[0].isoformat(), end_date=request.date_range[1].isoformat(),
            min_lon=request.bbox[0], max_lon=request.bbox[1], min_lat=request.bbox[2], max_lat=request.bbox[3],
        )

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Return the lazy spatial ECCO display grid; chunks load only when plotted."""
        grid = open_flux_grid(self._config(spec, request, policy))
        return _normalise_grid(grid[spec.variable or "co2_flux"], spec, request)

    def fetch_timeseries(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Use RAC weights in the native ECCO reader before returning a monthly series."""
        dataset = extract_flux_timeseries(self._config(spec, request, policy))
        return _monthly_start(dataset["area_weighted_flux"])


@ProviderRegistry.register("earthdata_cmr")
class EarthdataCMRProvider(BaseDataSourceProvider):
    """NASA Earthdata CMR adapter using netrc authentication and stream opening."""

    provider_name = "earthdata_cmr"

    @staticmethod
    def _configure_windows_netrc() -> None:
        """Point Earthaccess at ``.netrc`` when Windows defaults to ``_netrc``.

        Earthaccess on Windows defaults to ``%USERPROFILE%\\_netrc``.  The
        documented POSIX-style ``.netrc`` is also common, so honor it without
        copying credentials or overriding an administrator-supplied ``NETRC``.
        """
        if os.name != "nt" or os.environ.get("NETRC"):
            return
        posix_style_netrc = Path.home() / ".netrc"
        if posix_style_netrc.is_file():
            os.environ["NETRC"] = str(posix_style_netrc)

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Find CMR granules, open remotely, and retain only the requested MERRA-style field."""
        if not spec.collection or not spec.variable:
            raise RemoteIngestionError("earthdata_cmr requires collection and variable metadata.")
        earthaccess = _require("earthaccess", "earthaccess")
        self._configure_windows_netrc()
        try:
            authentication = earthaccess.login(strategy=policy.earthdata_auth_strategy)
        except Exception as exc:
            raise RemoteIngestionError(
                "Earthdata netrc authentication failed. Configure an owner-only ~/.netrc "
                "for urs.earthdata.nasa.gov; do not store credentials in this project."
            ) from exc
        if not getattr(authentication, "authenticated", False) or getattr(earthaccess, "__store__", None) is None:
            raise RemoteIngestionError(
                "Earthdata netrc authentication is unavailable. Configure an owner-only ~/.netrc "
                "for urs.earthdata.nasa.gov, then restart Streamlit."
            )
        granules = []
        seen_granules: set[str] = set()
        for bounding_box in _cmr_bounding_boxes(request):
            matches = earthaccess.search_data(
                short_name=spec.collection,
                temporal=(request.date_range[0].isoformat(), request.date_range[1].isoformat()),
                bounding_box=bounding_box,
            )
            for granule in matches:
                identifier = str(granule)
                if identifier not in seen_granules:
                    seen_granules.add(identifier)
                    granules.append(granule)
        if not granules:
            raise RemoteIngestionError("Earthdata CMR found no granules for the requested selection.")
        handles = earthaccess.open(granules)
        datasets = [xr.open_dataset(handle, chunks=spec.chunks) for handle in handles]
        data = xr.concat(
            [_configured_variable(item, spec.variable) for item in datasets],
            dim=spec.time_name or "time",
        )
        return _normalise_grid(data, spec, request)


@ProviderRegistry.register("azure_stac")
class AzureSTACProvider(BaseDataSourceProvider):
    """STAC COG/Zarr adapter for Planetary Computer or private Azure identity flows."""

    provider_name = "azure_stac"

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Search STAC and lazily read the selected COG asset through range requests."""
        if not spec.endpoint or not spec.collection or not spec.asset_key:
            raise RemoteIngestionError("azure_stac requires endpoint, collection, and asset_key metadata.")
        endpoint = validate_remote_url(spec.endpoint, policy)
        pystac_client = _require("pystac_client", "pystac-client")
        client = pystac_client.Client.open(endpoint)
        search = client.search(collections=[spec.collection], bbox=request.bbox, datetime=f"{request.date_range[0]}/{request.date_range[1]}")
        items = list(search.items())
        if not items:
            raise RemoteIngestionError("Azure STAC found no items for the requested selection.")
        href = items[0].assets[spec.asset_key].href
        validate_remote_url(href, policy)
        if policy.azure_auth_mode == "planetary_computer":
            planetary_computer = _require("planetary_computer", "planetary-computer")
            href = planetary_computer.sign(href)
        else:
            # Instantiate only the Azure identity chain (managed identity, workload
            # identity, Azure CLI, and developer login); no account key is accepted.
            _require("azure.identity", "azure-identity").DefaultAzureCredential()
        rioxarray = _require("rioxarray", "rioxarray")
        data = rioxarray.open_rasterio(href, chunks=spec.chunks).squeeze("band", drop=True)
        # COGs commonly use projected x/y; users should register an already-geographic COG for this adapter.
        return _normalise_grid(data, spec, request)


@ProviderRegistry.register("gcp_gcs")
class GCSZarrProvider(BaseDataSourceProvider):
    """GCS Zarr adapter using Application Default Credentials rather than key files."""

    provider_name = "gcp_gcs"

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Open a consolidated Zarr store through ADC and slice lazily in cloud storage."""
        if not spec.endpoint or not spec.variable:
            raise RemoteIngestionError("gcp_gcs requires endpoint and variable metadata.")
        if not spec.endpoint.startswith("gs://"):
            raise RemoteIngestionError("gcp_gcs endpoint must use a gs:// URI.")
        google_auth = _require("google.auth", "google-auth")
        gcsfs = _require("gcsfs", "gcsfs")
        _, project = google_auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
        # gcsfs delegates refreshable Application Default Credentials to google-auth.
        filesystem = gcsfs.GCSFileSystem(project=policy.gcp_project_id or project, token="google_default")
        store = filesystem.get_mapper(spec.endpoint)
        dataset = xr.open_zarr(store, consolidated=True, chunks=spec.chunks)
        return _normalise_grid(dataset[spec.variable], spec, request)


@ProviderRegistry.register("google_earth_engine")
class GoogleEarthEngineProvider(BaseDataSourceProvider):
    """Earth Engine adapter that transfers a compact server-side reduced series."""

    provider_name = "google_earth_engine"

    def fetch_grid(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Reject grid requests because this high-security adapter only exports reduced values."""
        raise RemoteIngestionError("google_earth_engine supports server-side time series only; use a STAC/Zarr dataset for maps.")

    def fetch_timeseries(self, spec: DatasetSpec, request: DataRequest, policy: SecurityPolicy) -> xr.DataArray:
        """Reduce each image server-side, then transfer only dates and regional means."""
        if not spec.collection or not spec.variable:
            raise RemoteIngestionError("google_earth_engine requires collection and variable metadata.")
        ee = _require("ee", "earthengine-api")
        ee.Initialize(project=policy.gee_project or policy.gcp_project_id, opt_url=policy.ee_opt_url)
        geometry = ee.Geometry.Rectangle(list(request.bbox), geodesic=False)
        collection = ee.ImageCollection(spec.collection).filterDate(str(request.date_range[0]), str(request.date_range[1])).filterBounds(geometry)
        scale = spec.ee_scale or 10000
        def reduce_image(image: Any) -> Any:
            value = image.select(spec.variable).reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=1_000_000_000).get(spec.variable)
            return ee.Feature(None, {"time": image.date().format("YYYY-MM-dd"), "value": value})
        features = collection.map(reduce_image).getInfo().get("features", [])
        times = [item["properties"]["time"] for item in features if item["properties"].get("value") is not None]
        values = [item["properties"]["value"] for item in features if item["properties"].get("value") is not None]
        if not times:
            raise RemoteIngestionError("Earth Engine returned no finite regional values.")
        result = xr.DataArray(values, dims=("time",), coords={"time": np.asarray(times, dtype="datetime64[ns]")}, name=spec.variable)
        result.attrs["units"] = spec.output_units or "unknown"
        return _monthly_start(result * spec.unit_scale)


def _resolve(dataset_name: str, config_path: str | Path) -> tuple[DatasetSpec, SecurityPolicy]:
    """Load the selected metadata and project-local security policy."""
    registry = load_dataset_registry(config_path)
    try:
        spec = registry.datasets[dataset_name]
    except KeyError as exc:
        raise RemoteIngestionError(f"Dataset '{dataset_name}' is not registered.") from exc
    policy = load_security_policy(Path(config_path).with_name(".env"))
    return spec, policy


def fetch_standardized_grid(dataset_name: str, bbox: tuple[float, float, float, float], date_range: tuple[str | date, str | date], config_path: str, longitude_convention: Literal["-180_180", "0_360"] = "-180_180") -> xr.DataArray:
    """Return a secure, spatially resolved remote grid for maps and downstream processing."""
    request = DataRequest(bbox=bbox, date_range=date_range, longitude_convention=longitude_convention)
    spec, policy = _resolve(dataset_name, config_path)
    return ProviderRegistry.create(spec.provider).fetch_grid(spec, request, policy)


def fetch_standardized_timeseries(dataset_name: str, bbox: tuple[float, float, float, float], date_range: tuple[str | date, str | date], config_path: str, longitude_convention: Literal["-180_180", "0_360"] = "-180_180") -> xr.DataArray:
    """Return a Month-Start regional time series derived from a secure remote source."""
    request = DataRequest(bbox=bbox, date_range=date_range, longitude_convention=longitude_convention)
    spec, policy = _resolve(dataset_name, config_path)
    return ProviderRegistry.create(spec.provider).fetch_timeseries(spec, request, policy)


def main() -> int:
    """Provide a small validated CLI suitable for scripted regional extraction."""
    parser = argparse.ArgumentParser(description="Secure remote geospatial time-series extraction")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--config-path", default=str(Path(__file__).resolve().parents[1] / "datasets.yaml"))
    arguments = parser.parse_args()
    series = fetch_standardized_timeseries(arguments.dataset, (arguments.min_lon, arguments.max_lon, arguments.min_lat, arguments.max_lat), (arguments.start_date, arguments.end_date), arguments.config_path)
    print(series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
