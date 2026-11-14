"""Tests for validation, date-line handling, and registry extensibility."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src import remote_ingestion as ingestion
from src.remote_ingestion import (
    BaseDataSourceProvider, DataRequest, DatasetSpec, EarthdataCMRProvider, ProviderRegistry,
    SecurityPolicy, SecurityPolicyError, _cmr_bounding_boxes, _configured_variable, _normalise_grid, load_dataset_registry,
    load_security_policy,
    validate_remote_url,
)


def _policy():
    return SecurityPolicy(allowed_domains=frozenset({"allowed.example"}))


def test_rejects_disallowed_endpoint_and_embedded_credentials():
    with pytest.raises(SecurityPolicyError):
        validate_remote_url("https://evil.example/file.nc", _policy())
    with pytest.raises(SecurityPolicyError):
        validate_remote_url("https://name:secret@allowed.example/file.nc", _policy())


def test_missing_allowlist_has_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOWED_DOMAINS", raising=False)
    with pytest.raises(SecurityPolicyError, match="copy .env.example"):
        load_security_policy(tmp_path / "missing.env")


def test_date_line_subset_stitches_both_sides_in_requested_convention():
    source = xr.DataArray(
        np.arange(12, dtype=float).reshape(1, 2, 6),
        dims=("time", "latitude", "longitude"),
        coords={"time": ["2024-01-15"], "latitude": [-1, 1], "longitude": [0, 10, 170, 180, 190, 350]},
    )
    request = DataRequest(bbox=(170, -170, -2, 2), date_range=(date(2024, 1, 1), date(2024, 1, 31)))
    result = _normalise_grid(source, DatasetSpec(provider="test", protocol="test"), request)
    assert result.lon.values.tolist() == [170, -180, -170]
    assert result.sizes == {"time": 1, "lat": 2, "lon": 3}


def test_cmr_bbox_uses_documented_axis_order_and_splits_the_date_line():
    global_request = DataRequest(bbox=(0, 360, -90, 90), date_range=(date(2024, 1, 1), date(2024, 1, 2)), longitude_convention="0_360")
    wrapped_request = DataRequest(bbox=(170, -170, -20, 20), date_range=(date(2024, 1, 1), date(2024, 1, 2)))
    assert _cmr_bounding_boxes(global_request) == ((-180.0, -90.0, 180.0, 90.0),)
    assert _cmr_bounding_boxes(wrapped_request) == ((170.0, -20.0, 180.0, 20.0), (-180.0, -20.0, -170.0, 20.0))


def test_cmr_provider_sends_cmr_ordered_global_bbox(monkeypatch):
    calls = []

    class FakeEarthaccess:
        __store__ = object()

        @staticmethod
        def login(**kwargs):
            assert kwargs == {"strategy": "netrc"}
            return type("Authentication", (), {"authenticated": True})()

        @staticmethod
        def search_data(**kwargs):
            calls.append(kwargs["bounding_box"])
            return []

    monkeypatch.setattr(ingestion, "_require", lambda *_: FakeEarthaccess)
    request = DataRequest(bbox=(0, 360, -90, 90), date_range=(date(2024, 1, 1), date(2024, 1, 2)), longitude_convention="0_360")
    spec = DatasetSpec(provider="earthdata_cmr", protocol="earthdata_cmr", collection="M2TMNXOCN", variable="TSKINW")
    with pytest.raises(ingestion.RemoteIngestionError, match="no granules"):
        EarthdataCMRProvider().fetch_grid(spec, request, _policy())
    assert calls == [(-180.0, -90.0, 180.0, 90.0)]


def test_cmr_provider_stops_before_open_when_netrc_login_is_not_authenticated(monkeypatch):
    class FakeEarthaccess:
        __store__ = None

        @staticmethod
        def login(**kwargs):
            return type("Authentication", (), {"authenticated": False})()

        @staticmethod
        def search_data(**kwargs):
            raise AssertionError("CMR search must not run without authenticated Earthdata access")

    monkeypatch.setattr(ingestion, "_require", lambda *_: FakeEarthaccess)
    request = DataRequest(bbox=(-1, 1, -1, 1), date_range=(date(2024, 1, 1), date(2024, 1, 2)))
    spec = DatasetSpec(provider="earthdata_cmr", protocol="earthdata_cmr", collection="M2TMNXOCN", variable="TSKINW")
    with pytest.raises(ingestion.RemoteIngestionError, match="netrc authentication is unavailable"):
        EarthdataCMRProvider().fetch_grid(spec, request, _policy())


def test_windows_netrc_uses_existing_dot_netrc_without_copying_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.os, "name", "nt")
    monkeypatch.delenv("NETRC", raising=False)
    monkeypatch.setattr(ingestion.Path, "home", lambda: tmp_path)
    (tmp_path / ".netrc").write_text("machine example login placeholder password placeholder\n", encoding="utf-8")
    EarthdataCMRProvider._configure_windows_netrc()
    assert Path(ingestion.os.environ["NETRC"]) == tmp_path / ".netrc"


def test_ecco_registry_uses_the_shared_llc270_grid_directory():
    registry = load_dataset_registry("datasets.yaml")
    assert registry.datasets["ecco_darwin_v5_co2_flux"].collection == "https://data.nas.nasa.gov/ecco/llc_270/grid/"


def test_merra_registry_uses_confirmed_water_skin_temperature_variable():
    registry = load_dataset_registry("datasets.yaml")
    assert registry.datasets["merra2_m2tmnxocn_sst"].variable == "TSKINWTR"


def test_registry_exposes_dataset_specific_remote_date_bounds():
    registry = load_dataset_registry("datasets.yaml")
    ecco = registry.datasets["ecco_darwin_v5_co2_flux"]
    merra = registry.datasets["merra2_m2tmnxocn_sst"]
    assert (ecco.available_start, ecco.available_end) == (date(1992, 1, 1), date(2011, 12, 31))
    assert merra.available_start == date(1980, 1, 1)
    assert merra.available_end is None


def test_registry_rejects_inverted_dataset_availability_dates():
    with pytest.raises(ValueError, match="available_start"):
        DatasetSpec(
            provider="test",
            protocol="test",
            available_start=date(2025, 1, 1),
            available_end=date(2024, 1, 1),
        )


def test_configured_variable_explains_a_dataset_schema_mismatch():
    dataset = xr.Dataset({"TSKINWTR": xr.DataArray([273.15], dims=("time",))})
    with pytest.raises(ingestion.RemoteIngestionError, match="Available data variables: TSKINWTR"):
        _configured_variable(dataset, "TSKINW")


def test_request_rejects_invalid_latitude_order():
    with pytest.raises(ValueError, match="latitude"):
        DataRequest(bbox=(0, 1, 10, 10), date_range=(date(2024, 1, 1), date(2024, 1, 2)))


def test_registry_accepts_new_provider_without_pipeline_change():
    @ProviderRegistry.register("test_provider")
    class TestProvider(BaseDataSourceProvider):
        provider_name = "test_provider"

        def fetch_grid(self, spec, request, policy):
            return xr.DataArray(np.ones((1, 1, 1)), dims=("time", "lat", "lon"), coords={"time": ["2024-01-01"], "lat": [0], "lon": [0]})

    assert isinstance(ProviderRegistry.create("test_provider"), TestProvider)


def test_registry_metadata_parses_without_credentials():
    registry = load_dataset_registry("datasets.yaml")
    assert registry.datasets["merra2_m2tmnxocn_sst"].unit_offset == -273.15
    assert "password" not in registry.model_dump_json().lower()


def test_grid_normalisation_applies_registered_kelvin_to_celsius_conversion():
    source = xr.DataArray(
        np.array([[[273.15]]]), dims=("time", "lat", "lon"),
        coords={"time": ["2024-01-01"], "lat": [0], "lon": [0]},
    )
    request = DataRequest(bbox=(-1, 1, -1, 1), date_range=(date(2024, 1, 1), date(2024, 1, 1)))
    result = _normalise_grid(source, DatasetSpec(provider="test", protocol="test", unit_offset=-273.15, output_units="degC"), request)
    assert result.item() == pytest.approx(0.0)
    assert result.attrs["units"] == "degC"
