"""Focused tests for the secure ECCO-Darwin MDS reader."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src import ecco_darwin_flux as flux


def _config(**overrides):
    values = {
        "flux_url": "https://data.nas.nasa.gov/ecco/flux/",
        "grid_url": "https://data.nas.nasa.gov/ecco/grid/",
        "ref_datetime": "1992-01-01T00:00:00+00:00",
        "dt_seconds": 86400,
        "start_date": "1992-01-01T00:00:00+00:00",
        "end_date": "1992-01-03T00:00:00+00:00",
        "min_lon": 350,
        "max_lon": 20,
        "min_lat": -10,
        "max_lat": 10,
    }
    values.update(overrides)
    return flux.EccoConfig(**values)


def test_parse_mds_metadata_without_nested_dimension_brackets():
    metadata = flux.parse_mds_metadata(
        """dimList = [ 2, 1, 2, 3, 1, 3, 1, 1, 1 ]\n"""
        """dataprec = [ 'float32' ]\ntimeInterval = [ 0.0, 8.64D+04 ]"""
    )
    assert metadata.shape == (2, 3, 1)
    assert metadata.dtype == np.dtype(">f4")
    assert metadata.time_interval == (0.0, 86400.0)


def test_spatial_mask_wraps_at_date_line():
    config = _config()
    longitude = np.array([[349.0, 355.0, 5.0, 21.0]])
    latitude = np.zeros_like(longitude)
    assert _selected(flux._spatial_mask(longitude, latitude, config)) == [False, True, True, False]


def test_invalid_remote_host_is_rejected():
    with pytest.raises(ValueError):
        _config(flux_url="https://example.com/flux/")


def test_remote_read_retries_rate_limit_instead_of_marking_object_missing(monkeypatch):
    attempts = []
    pauses = []

    class RemoteFile:
        def __enter__(self):
            attempts.append("open")
            if len(attempts) == 1:
                raise flux.aiohttp.ClientResponseError(
                    request_info=None, history=(), status=429, message="Too Many Requests"
                )
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"metadata"

    class Filesystem:
        def open(self, *args, **kwargs):
            return RemoteFile()

    monkeypatch.setattr(flux, "_filesystem", lambda: Filesystem())
    monkeypatch.setattr(flux.time, "sleep", pauses.append)

    assert flux._read_remote("https://data.nas.nasa.gov/ecco/flux.meta") == b"metadata"
    assert len(attempts) == 2
    assert pauses == [1.5]


def test_remote_read_retries_fsspec_wrapped_bad_gateway(monkeypatch):
    attempts = []
    pauses = []

    class RemoteFile:
        def __enter__(self):
            attempts.append("open")
            if len(attempts) == 1:
                transport_error = flux.aiohttp.ClientResponseError(
                    request_info=None, history=(), status=502, message="Bad Gateway"
                )
                wrapped = FileNotFoundError("fsspec wrapped a temporary HTTP error")
                raise wrapped from transport_error
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"metadata"

    class Filesystem:
        def open(self, *args, **kwargs):
            return RemoteFile()

    monkeypatch.setattr(flux, "_filesystem", lambda: Filesystem())
    monkeypatch.setattr(flux.time, "sleep", pauses.append)

    assert flux._read_remote("https://data.nas.nasa.gov/ecco/flux.meta") == b"metadata"
    assert len(attempts) == 2
    assert pauses == [1.5]


def test_unavailable_month_becomes_all_missing_slice(monkeypatch):
    monkeypatch.setattr(
        flux,
        "_read_mds",
        lambda _: (_ for _ in ()).throw(flux.RemoteDataError("NASA unavailable")),
    )

    result = flux._read_flux_cells(
        "https://data.nas.nasa.gov/ecco/flux/missing.data",
        np.array([0, 1]),
        (1, 2),
    )

    assert result.shape == (1, 2)
    assert np.isnan(result).all()


def test_regular_grid_supports_nominal_ecco_third_degree_resolution():
    longitude = np.array([[0.0, 1.0], [0.0, 1.0]])
    latitude = np.array([[0.0, 0.0], [1.0, 1.0]])
    _, target_latitude, target_longitude = flux._regular_grid_indices(
        longitude, latitude, 1.0 / 3.0
    )

    assert target_latitude.size == 540
    assert target_longitude.size == 1080


def test_open_flux_grid_defers_each_month_until_selected(monkeypatch):
    config = _config()
    longitude = np.array([[179.0, -179.0], [1.0, -1.0]])
    latitude = np.array([[1.0, 1.0], [-1.0, -1.0]])
    calls = []

    monkeypatch.setattr(
        flux,
        "_directory_links",
        lambda url: [f"{url.rstrip('/')}/XC.data", f"{url.rstrip('/')}/YC.data"]
        if url == config.grid_url
        else [],
    )
    monkeypatch.setattr(
        flux,
        "_read_mds",
        lambda url: (longitude if url.endswith("XC.data") else latitude, None),
    )
    monkeypatch.setattr(
        flux,
        "_flux_records",
        lambda _: [
            ("https://data.nas.nasa.gov/ecco/flux/first.data", datetime(1992, 1, 1, tzinfo=timezone.utc)),
            ("https://data.nas.nasa.gov/ecco/flux/second.data", datetime(1992, 2, 1, tzinfo=timezone.utc)),
        ],
    )

    def fake_read(url, source_indices, target_shape):
        calls.append(url)
        return np.full(target_shape, 1.0 if url.endswith("first.data") else 2.0, dtype=np.float32)

    monkeypatch.setattr(flux, "_read_flux_cells", fake_read)
    dataset = flux.open_flux_grid(config, resolution_degrees=5.0)

    assert dataset["co2_flux"].dims == ("time", "lat", "lon")
    assert dataset.sizes == {"time": 2, "lat": 36, "lon": 72}
    assert calls == []
    assert np.all(dataset["co2_flux"].isel(time=1).compute().values == 2.0)
    assert calls == ["https://data.nas.nasa.gov/ecco/flux/second.data"]


def test_flux_records_reads_small_metadata_before_filtering_binary_payloads(monkeypatch):
    config = _config(
        start_date="1992-01-01T00:00:00+00:00",
        end_date="1992-01-01T00:00:00+00:00",
        dt_seconds=1200,
    )
    calls = []
    metadata = (
        b"dimList = [ 1, 1, 1, 1, 1, 1 ]\n"
        b"dataprec = [ 'float32' ]\n"
        b"timeInterval = [ 0.0, 0.0 ]\n"
    )

    monkeypatch.setattr(
        flux,
        "_directory_links",
        lambda _: [
            "https://data.nas.nasa.gov/ecco/flux/CO2_flux.0000000000.data",
            "https://data.nas.nasa.gov/ecco/flux/CO2_flux.9999999999.data",
        ],
    )

    def read_metadata(url):
        calls.append(url)
        assert url.endswith(".meta")
        return metadata

    monkeypatch.setattr(flux, "_read_remote", read_metadata)
    records = flux._flux_records(config)
    assert [url for url, _ in records] == ["https://data.nas.nasa.gov/ecco/flux/CO2_flux.0000000000.data", "https://data.nas.nasa.gov/ecco/flux/CO2_flux.9999999999.data"]
    assert all(url.endswith(".meta") for url in calls)


def test_flux_records_uses_model_timestep_when_metadata_is_temporarily_unavailable(monkeypatch):
    config = _config(
        start_date="1992-02-01T00:00:00+00:00",
        end_date="1992-02-01T00:00:00+00:00",
        dt_seconds=1200,
    )
    data_url = "https://data.nas.nasa.gov/ecco/flux/CO2_flux.0000002232.data"
    monkeypatch.setattr(flux, "_directory_links", lambda _: [data_url])
    monkeypatch.setattr(
        flux,
        "_read_remote",
        lambda _: (_ for _ in ()).throw(flux.RemoteDataError("temporary metadata outage")),
    )
    records = flux._flux_records(config)
    assert records == [(data_url, datetime(1992, 2, 1, tzinfo=timezone.utc))]


def test_extraction_reads_data_only_after_date_filter(monkeypatch):
    config = _config(
        min_lon=0,
        max_lon=360,
        min_lat=-90,
        max_lat=90,
        end_date="1992-01-01T00:00:00+00:00",
    )
    calls = []
    meta = b"dimList = [ 2, 1, 2, 1, 1, 1 ]\ndataprec = [ 'float32' ]\n"

    def fake_read(url):
        calls.append(url)
        if url.endswith(".meta"):
            return meta
        return np.array([1.0, 1.0], dtype=">f4").tobytes()

    monkeypatch.setattr(flux, "_read_remote", fake_read)
    monkeypatch.setattr(
        flux,
        "_directory_links",
        lambda url: (
            [
                url.rstrip("/") + "/XC.data",
                url.rstrip("/") + "/YC.data",
                url.rstrip("/") + "/RAC.data",
            ]
            if url == config.grid_url
            else [
                url.rstrip("/") + "/flux_0001.data",
                url.rstrip("/") + "/flux_0002.data",
            ]
        ),
    )
    monkeypatch.setattr(
        flux,
        "_time_from_metadata_or_name",
        lambda url, metadata, config: config.ref_datetime
        if url.endswith("0001.data")
        else datetime(1992, 1, 2, tzinfo=timezone.utc),
    )
    result = flux.extract_flux_timeseries(config)
    assert result.sizes["time"] == 1
    assert not any(url.endswith("flux_0002.data") for url in calls)


def _selected(mask):
    return mask.ravel().tolist()
