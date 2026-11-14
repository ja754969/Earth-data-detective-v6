"""Interactive climate-grid viewer.

Run with::

	streamlit run data_view.py

The computational and plotting code lives in ``src`` so it can also be used
from notebooks or automated workflows without starting Streamlit.
"""

from __future__ import annotations

import hashlib
import io
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from src.climate_data import (
	DatasetError,
	available_periods,
	calculate_anomaly,
	choose_coordinate_names,
	convert_display_units,
	build_hovmoller,
	calculate_linear_trend,
	detrend_data,
	discover_netcdf_files,
	format_years,
	load_dataset,
	mask_invalid_values,
	parse_year_spec,
	select_region,
	select_period,
	select_vertical_section,
	transform_values,
	vector_magnitude,
	wind_stress_curl,
	ensemble_mean_dataset,
	ensemble_agreement_fraction,
	dataset_to_netcdf_bytes,
	validate_plot_size,
	temporal_resolution,
	polygon_mean_timeseries,
	build_annual_cycles,
	finite_data_range,
)
from src.climate_plot import PlotOptions, draw_annual_cycles, draw_colorbar, draw_hovmoller, draw_map, draw_vertical_section
from src.ecco_darwin_flux import ConfigurationError, EccoConfig, open_flux_grid, load_environment_config
from src.remote_ingestion import (
	RemoteIngestionError,
	SecurityPolicyError,
	fetch_standardized_grid,
	load_dataset_registry,
	load_security_policy,
	validate_remote_url,
)

try:
	from shared_helpers.earth_data_loader import get_remote_dataset_from_config, load_manifest
except ImportError:
	get_remote_dataset_from_config = None
	load_manifest = None


APP_DIR = Path(__file__).resolve().parent
# Keep discovery scoped to this release; users can add another folder explicitly.
DEFAULT_DATA_ROOTS = [APP_DIR]


@st.cache_data(show_spinner=False)
def cached_dataset(source: str | tuple[str, ...], payload: bytes | None = None):
	"""Cache metadata/data handles by source and uploaded file content."""
	if isinstance(source, str) and source.startswith("config://"):
		if get_remote_dataset_from_config is None:
			raise DatasetError("Configured remote datasets require the optional remote-loader dependencies.")
		return get_remote_dataset_from_config(
			source.removeprefix("config://"),
			config_path=str(APP_DIR / "data_config.json"),
		)
	if isinstance(source, str) and source.startswith(("http://", "https://")):
		# Direct remote URLs remain available, but network access is now subject to
		# the same exact-domain policy as registered cloud providers.
		validate_remote_url(source, load_security_policy(APP_DIR / ".env"))
	return load_dataset(source, payload)


def _configured_remote_keys() -> list[str]:
	"""Return manifest keys without opening any remote resource."""
	if load_manifest is None:
		return []
	try:
		manifest = load_manifest(str(APP_DIR / "data_config.json"))
	except (OSError, ValueError, TypeError):
		return []
	datasets = manifest.get("datasets", {})
	return sorted(str(key) for key in datasets) if isinstance(datasets, dict) else []


@st.cache_data(show_spinner=False)
def cached_data_range_by_key(cache_key: str, _data):
	"""Cache finite extrema used to guide manual plotting controls."""
	return finite_data_range(_data)


def _preview_data_range(data):
	"""Bound initial extrema discovery for a lazy remote time series."""
	if "time" not in data.dims or data.sizes["time"] <= 3:
		return finite_data_range(data)
	indices = np.unique(np.linspace(0, data.sizes["time"] - 1, num=3, dtype=int))
	return finite_data_range(data.isel(time=indices))


def _data_range_cache_key(source, variable: str, payload: bytes | None) -> str:
	"""Build a stable cache key without asking Streamlit to hash xarray objects."""
	source_identity = repr(source)
	if payload is not None:
		source_identity += f":{hashlib.sha256(payload).hexdigest()}"
	return f"{source_identity}:{variable}"


@st.cache_resource(show_spinner=False)
def cached_ecco_grid(config_json: str, resolution_degrees: float):
	"""Keep the lazy ECCO task graph without materialising all monthly fields."""
	return open_flux_grid(EccoConfig.model_validate_json(config_json), resolution_degrees)


@st.cache_resource(show_spinner=False)
def cached_registered_remote_grid(
	dataset_name: str,
	start_date: str,
	end_date: str,
	min_lon: float,
	max_lon: float,
	min_lat: float,
	max_lat: float,
	longitude_convention: str,
):
	"""Open one registered provider grid lazily for the existing map workflow."""
	return fetch_standardized_grid(
		dataset_name,
		(min_lon, max_lon, min_lat, max_lat),
		(start_date, end_date),
		str(APP_DIR / "datasets.yaml"),
		longitude_convention,
	)


def _registered_remote_keys() -> list[str]:
	"""Return secure registry names without opening a provider connection."""
	try:
		return sorted(load_dataset_registry(APP_DIR / "datasets.yaml").datasets)
	except (OSError, ValueError):
		return []


def _registered_remote_date_bounds(dataset_name: str) -> tuple[date, date]:
	"""Return registry-specific UI bounds, with safe defaults for new providers."""
	try:
		spec = load_dataset_registry(APP_DIR / "datasets.yaml").datasets[dataset_name]
		return spec.available_start or date(1900, 1, 1), spec.available_end or date.today()
	except (KeyError, OSError, ValueError):
		return date(1900, 1, 1), date.today()


def _apply_display_units(data, units: str):
	"""Apply a user-supplied LaTeX unit label without changing the values."""
	if not units.strip():
		return data
	result = data.copy()
	result.attrs["units"] = units.strip()
	result.attrs["units_latex"] = units.strip()
	return result


def _units_label_for_export(units: str) -> str:
	"""Format the optional display-unit expression as Matplotlib math text."""
	value = units.strip()
	if not value:
		return ""
	if value.startswith("$") and value.endswith("$"):
		value = value[1:-1].strip()
	if r"\mathrm" not in value and r"\text" not in value:
		value = rf"\mathrm{{{value}}}"
	return f"${value}$"


_TRANSFORMATION_OPTIONS = [
	"No conversion",
	"Automatic CF unit conversion",
	"Multiply by a value",
	"Divide by a value",
	"Add a value",
	"Subtract a value",
	"Geopotential (m^2/s^2) / 9.8 -> height (m)",
	"Air-sea CO2 flux (mmolC m^-2 day^-1) * 0.36525 -> Air-sea CO2 flux (molC m^-2 year^-1)",
]


def _value_conversion_controls(label: str, key_prefix: str):
	"""Create an independent conversion control group for one data source."""
	st.markdown(f"**{label} value conversion**")
	transformation = st.selectbox(
		"Transformation",
		_TRANSFORMATION_OPTIONS,
		key=f"{key_prefix}-transformation",
	)
	operand = 9.8
	if transformation == "Geopotential (m^2/s^2) / 9.8 -> height (m)":
		operand = st.number_input(
			"Gravity constant (m/s^2)",
			min_value=0.000001,
			value=9.8,
			format="%.6f",
			key=f"{key_prefix}-gravity",
		)
	elif transformation == "Air-sea CO2 flux (mmolC m^-2 day^-1) * 0.36525 -> Air-sea CO2 flux (molC m^-2 year^-1)":
		operand = st.number_input(
			"mmolC/day to molC/year factor",
			min_value=0.000001,
			value=0.36525,
			format="%.6f",
			key=f"{key_prefix}-flux-factor",
		)
	elif transformation in {"Multiply by a value", "Divide by a value", "Add a value", "Subtract a value"}:
		operand = st.number_input("Arithmetic value", value=1.0, key=f"{key_prefix}-operand")
	output_units = st.text_input(
		"Display unit label (LaTeX, optional)",
		value="",
		key=f"{key_prefix}-units",
	)
	return transformation, float(operand), output_units


def _prepare_analysis_source(
	source,
	lon_bounds,
	lat_bounds,
	lon_convention,
	missing_lower,
	missing_upper,
	use_missing_lower,
	use_missing_upper,
	detrend_enabled,
	detrend_type,
	detrend_reference_year,
	year,
	baseline_years,
	month,
	season,
):
	"""Apply spatial/time limits before any operation that may materialize data."""
	result = mask_invalid_values(
		source,
		missing_lower if use_missing_lower else None,
		missing_upper if use_missing_upper else None,
	)
	result = select_region(result, lon_bounds, lat_bounds, lon_convention)
	if detrend_enabled:
		result = detrend_data(result, detrend_type, detrend_reference_year)
	return result


def _number_range(label: str, minimum: float, maximum: float, default: tuple[float, float]):
	"""Read a numeric range without forcing users to drag a coarse slider."""
	st.sidebar.caption(label)
	columns = st.sidebar.columns(2)
	lower = columns[0].number_input(
		"Minimum",
		min_value=float(minimum),
		max_value=float(maximum),
		value=float(np.clip(default[0], minimum, maximum)),
		step=0.1,
		format="%.6f",
		key=f"{label}-minimum",
	)
	upper = columns[1].number_input(
		"Maximum",
		min_value=float(minimum),
		max_value=float(maximum),
		value=float(np.clip(default[1], minimum, maximum)),
		step=0.1,
		format="%.6f",
		key=f"{label}-maximum",
	)
	return float(lower), float(upper)


def _parse_polygon_vertices(text: str) -> list[tuple[float, float]]:
	"""Parse ``lon,lat; lon,lat`` vertices entered in the sidebar."""
	vertices: list[tuple[float, float]] = []
	for item in text.replace("\n", ";").split(";"):
		if not item.strip():
			continue
		parts = [part.strip() for part in item.split(",")]
		if len(parts) != 2:
			raise ValueError("Each polygon vertex must be written as longitude,latitude.")
		vertices.append((float(parts[0]), float(parts[1])))
	if len(vertices) < 3:
		raise ValueError("A polygon requires at least three vertices.")
	return vertices


def _analysis_annotation(
	display_mode: str,
	temporal_mode: str,
	year: int | tuple[int, ...] | None,
	year_range: tuple[int, int] | None,
	month: int | None,
	season: tuple[int, ...] | None,
	season_start_month: int | None,
	lon_bounds: tuple[float, float],
	lat_bounds: tuple[float, float],
	lon_convention: str,
	detrend_enabled: bool,
	detrend_type: str,
	transformation: str,
	trend_time_unit: str,
) -> str:
	"""Create a compact provenance summary for figure annotation."""
	if month is not None:
		period_text = f"month={month:02d}"
	else:
		period_text = f"months={','.join(f'{value:02d}' for value in season or ())}; start={season_start_month:02d}"
	if display_mode == "Linear trend":
		time_text = f"analysis years={year_range}; slope basis={trend_time_unit.lower()}"
	else:
		time_text = f"target years={format_years(year)}; baseline years={year_range}"
	detrend_text = f"detrend={detrend_type}" if detrend_enabled else "detrend=off"
	return (
		f"{display_mode} | {temporal_mode.lower()} | {period_text} | {time_text}\n"
		f"region=lon {lon_bounds[0]:g} to {lon_bounds[1]:g} ({lon_convention}), "
		f"lat {lat_bounds[0]:g} to {lat_bounds[1]:g} | {detrend_text} | conversion={transformation}"
	)


def main() -> None:
	st.set_page_config(page_title="Earth Grid Explorer", layout="wide")
	st.title("Earth Grid Explorer")
	st.caption("Interactive NetCDF maps with lazy loading, anomalies, and vector overlays")

	with st.sidebar:
		st.header("Data source")
		data_source_mode = st.radio(
			"Data source type",
			["Existing NetCDF / remote sources", "ECCO-Darwin CO2 flux"],
			help="ECCO-Darwin reads MITgcm MDS metadata first and uses the settings in the project .env file.",
		)
		ecco_config_json = None
		ecco_resolution_degrees = 1.0
		if data_source_mode == "ECCO-Darwin CO2 flux":
			st.caption("Configuration: project .env file")
			if st.button("Read ECCO-Darwin CO2 flux", type="primary"):
				try:
					st.session_state["ecco-config-json"] = load_environment_config().model_dump_json()
				except (OSError, ValueError, ConfigurationError) as exc:
					st.error(f"Unable to read ECCO-Darwin CO2 flux: {exc}")
					st.stop()
			ecco_config_json = st.session_state.get("ecco-config-json")
			if not ecco_config_json:
				st.info("Configure .env, then click Read ECCO-Darwin CO2 flux.")
				st.stop()
			resolution_options = {
				"1° (recommended: fastest)": 1.0,
				"0.5°": 0.5,
				"0.25°": 0.25,
				"ECCO LLC270 nominal resolution (1/3°)": 1.0 / 3.0,
			}
			resolution_label = st.selectbox(
				"ECCO display resolution",
				list(resolution_options),
				index=0,
				help=(
					"ECCO LLC270 has a nominal horizontal resolution of about 1/3°. "
					"The viewer maps its curvilinear grid onto this regular display grid; "
					"finer choices increase memory use and remote processing."
				),
			)
			ecco_resolution_degrees = resolution_options[resolution_label]
			st.caption(
				f"ECCO LLC270 nominal horizontal resolution: ~1/3°. Display grid: {resolution_label}. "
				"Selected months are downloaded on demand. "
				"Vertical sections are unavailable because CO2_flux is a surface field."
			)
		auto_update = st.checkbox("Update plot automatically", value=True, help="When disabled, controls are applied only after clicking Update plot.")
		uploads = st.file_uploader("Upload NetCDF", type=["nc", "nc4", "cdf"])
		additional_root_text = st.text_input(
			"Additional local data folder",
			value="",
			help="Enter a specific folder, for example D:\\climate_data. The folder is scanned locally; files are not uploaded.",
		)
		file_pattern = st.text_input(
			"NetCDF filename pattern",
			value="*",
			help="Used only when combining files from the additional folder; for example, model_*_2020.nc.",
		)
		ensemble_mode = st.checkbox(
			"Calculate ensemble mean from folders",
			disabled=uploads is not None,
			help="Each listed folder is one ensemble member. Files inside one folder are combined along time.",
		)
		ensemble_folder_text = st.text_area(
			"Ensemble member folders",
			value="",
			disabled=not ensemble_mode,
			help="Enter one local folder per line or separate folders with semicolons.",
		)
		data_roots = list(DEFAULT_DATA_ROOTS)
		additional_root = None
		additional_files = []
		if additional_root_text.strip():
			additional_root = Path(additional_root_text.strip()).expanduser()
			if additional_root.is_dir():
				data_roots.append(additional_root)
				additional_files = discover_netcdf_files([additional_root], limit=None, pattern=file_pattern or "*")
			else:
				st.warning(f"Local data folder does not exist: {additional_root}")
		local_files = discover_netcdf_files(data_roots)
		choices = ["Upload a file"] + [str(path) for path in local_files]
		combine_files = st.checkbox(
			"Combine all NetCDF files in this folder",
			disabled=uploads is not None or not additional_files or ensemble_mode,
			help="Use this for folders containing separate time chunks. Files are opened lazily and combined by coordinates.",
		)
		if additional_root is not None and additional_root.is_dir() and file_pattern != "*":
			st.caption(f"Pattern match: {file_pattern} ({len(additional_files)} files)")
		source = st.selectbox("Local NetCDF file", choices, disabled=uploads is not None or combine_files)
		remote_url = st.text_input("Remote NetCDF / OPeNDAP URL", help="The URL is opened lazily by xarray.")
		configured_keys = _configured_remote_keys()
		configured_remote = st.selectbox(
			"Configured remote dataset",
			["None"] + configured_keys,
			help="Select a manifest entry for NASA Earthdata/COG, NOAA THREDDS/OPeNDAP, or Google Earth Engine. Credentials are read from environment variables, never from the manifest.",
		)
		registered_remote = st.selectbox(
			"Secure registered remote dataset",
			["None"] + _registered_remote_keys(),
			help="Uses datasets.yaml plus .env policy, passwordless cloud identity, and lazy remote subsetting.",
		)
		registered_request = None
		if registered_remote != "None":
			st.caption("Registered remote request (used before data are opened)")
			registered_min_date, registered_max_date = _registered_remote_date_bounds(registered_remote)
			registered_default_end = min(registered_max_date, date.today())
			registered_default_start = max(registered_min_date, date(registered_default_end.year - 1, 1, 1))
			registered_start = st.date_input(
				"Remote start date", value=registered_default_start, min_value=registered_min_date,
				max_value=registered_max_date, key=f"registered-remote-start-{registered_remote}",
			)
			registered_end = st.date_input(
				"Remote end date", value=registered_default_end, min_value=registered_min_date,
				max_value=registered_max_date, key=f"registered-remote-end-{registered_remote}",
			)
			registered_min_lon = st.number_input("Remote minimum longitude", value=-180.0, key="registered-remote-min-lon")
			registered_max_lon = st.number_input("Remote maximum longitude", value=180.0, key="registered-remote-max-lon")
			registered_min_lat = st.number_input("Remote minimum latitude", value=-90.0, key="registered-remote-min-lat")
			registered_max_lat = st.number_input("Remote maximum latitude", value=90.0, key="registered-remote-max-lat")
			registered_convention = st.selectbox(
				"Remote longitude convention", ["-180_180", "0_360"], key="registered-remote-convention"
			)
			if registered_start > registered_end:
				st.warning("Remote start date must not be after remote end date.")
			else:
				registered_request = (
					str(registered_start), str(registered_end), float(registered_min_lon), float(registered_max_lon),
					float(registered_min_lat), float(registered_max_lat), registered_convention,
				)
		comparison_enabled = st.checkbox(
			"Use a second folder for comparison overlays",
			disabled=uploads is not None,
			help="Select independent scalar and vector sources. The second folder may use a different grid resolution and spatial extent.",
		)
		comparison_root_text = st.text_input("Comparison data folder", value="", disabled=not comparison_enabled)
		comparison_pattern = st.text_input(
			"Comparison NetCDF filename pattern",
			value="*",
			disabled=not comparison_enabled,
			help="Only files directly inside the comparison folder are considered.",
		)
		comparison_files = []
		comparison_root = Path(comparison_root_text.strip()).expanduser() if comparison_root_text.strip() else None
		if comparison_enabled and comparison_root is not None:
			if comparison_root.is_dir():
				comparison_files = discover_netcdf_files([comparison_root], limit=None, pattern=comparison_pattern or "*")
				st.caption(f"Comparison pattern match: {comparison_pattern} ({len(comparison_files)} files)")
			else:
				st.warning(f"Comparison data folder does not exist: {comparison_root}")

	ensemble_groups = []
	if ensemble_mode:
		for folder_text in ensemble_folder_text.replace(";", "\n").splitlines():
			folder = Path(folder_text.strip()).expanduser()
			if not folder_text.strip():
				continue
			if not folder.is_dir():
				st.warning(f"Ensemble member folder does not exist: {folder}")
				continue
			files = discover_netcdf_files([folder], limit=None, pattern=file_pattern or "*")
			if not files:
				st.warning(f"No NetCDF files found in ensemble member folder: {folder}")
				continue
			ensemble_groups.append(tuple(str(path) for path in files))
		if len(ensemble_groups) < 2:
			st.info("Add at least two valid ensemble member folders.")
			st.stop()
	local_primary_selected = (
		uploads is not None
		or ensemble_mode
		or combine_files
		or source != "Upload a file"
	)
	if data_source_mode == "ECCO-Darwin CO2 flux":
		source_key = f"ecco://{hashlib.sha256(ecco_config_json.encode()).hexdigest()}:{ecco_resolution_degrees:g}"
	elif registered_remote != "None":
		source_key = f"registered://{registered_remote}"
	elif configured_remote != "None" and not (
		local_primary_selected and comparison_enabled and bool(comparison_files)
	):
		source_key = f"config://{configured_remote}"
	elif configured_remote != "None":
		st.info(
			"Both local primary and comparison folders are specified, so the local "
			"primary folder remains active and the configured remote dataset is not used."
		)
	elif uploads:
		source_key = "upload"
	elif remote_url.strip():
		source_key = remote_url.strip()
	elif ensemble_mode:
		source_key = tuple(ensemble_groups)
	elif combine_files:
		source_key = tuple(str(path) for path in additional_files)
	else:
		source_key = source
	if source_key == "upload" and uploads is None:
		st.info("Upload a NetCDF file, select a local file, or provide a remote URL to begin.")
		st.stop()
	payload = uploads.getvalue() if uploads else None

	try:
		if data_source_mode == "ECCO-Darwin CO2 flux":
			ds = cached_ecco_grid(ecco_config_json, ecco_resolution_degrees)
		elif registered_remote != "None" and registered_request is not None:
			registered_grid = cached_registered_remote_grid(registered_remote, *registered_request)
			ds = registered_grid.to_dataset(name=registered_grid.name or registered_remote)
		else:
			ds = cached_dataset(source_key, payload)
		coord_names = choose_coordinate_names(ds)
	except (DatasetError, OSError, ValueError, RemoteIngestionError, SecurityPolicyError) as exc:
		st.error(f"Unable to open this dataset: {exc}")
		st.stop()

	comparison_ds = None
	if comparison_enabled:
		if not comparison_files:
			st.error("Comparison overlays require at least one matching NetCDF file.")
			st.stop()
		try:
			comparison_ds = cached_dataset(tuple(str(path) for path in comparison_files))
		except (DatasetError, OSError, ValueError) as exc:
			st.error(f"Unable to open the comparison dataset: {exc}")
			st.stop()

	variables = list(ds.data_vars)
	if not variables:
		st.error("The dataset contains no data variables.")
		st.stop()
	ensemble_member_count = int(ds.sizes.get("ensemble", 1))

	with st.sidebar:
		st.header("Field and time")
		scalar_ds = ds
		contour_ds = ds
		vector_ds = ds
		if comparison_ds is not None:
			scalar_source = st.radio("Pcolor data source", ["Primary folder", "Comparison folder"], horizontal=True)
			contour_source = st.radio("Contour data source", ["Primary folder", "Comparison folder"], horizontal=True)
			vector_source = st.radio("Quiver/streamline data source", ["Primary folder", "Comparison folder"], horizontal=True)
			if scalar_source == "Comparison folder":
				scalar_ds = comparison_ds
			if contour_source == "Comparison folder":
				contour_ds = comparison_ds
			if vector_source == "Comparison folder":
				vector_ds = comparison_ds
		scalar_variables = list(scalar_ds.data_vars)
		if not scalar_variables:
			st.error("The selected scalar source contains no data variables.")
			st.stop()
		variable = st.selectbox("Scalar variable", scalar_variables)
		selected = scalar_ds[variable]
		contour_variables = list(contour_ds.data_vars)
		if not contour_variables:
			st.error("The selected contour source contains no data variables.")
			st.stop()
		contour_variable = st.selectbox(
			"Contour variable",
			contour_variables,
			index=contour_variables.index(variable) if variable in contour_variables else 0,
			help="Select the field used for isolines. It may come from a different folder and grid than the pcolor field.",
		)
		contour_selected = contour_ds[contour_variable]
		try:
			validate_plot_size(selected)
			selected_range = (
				_preview_data_range(selected)
				if data_source_mode == "ECCO-Darwin CO2 flux"
				else cached_data_range_by_key(
					_data_range_cache_key(source_key, variable, payload), selected
				)
			)
			contour_source_key = source_key
			if comparison_ds is not None and contour_source == "Comparison folder":
				contour_source_key = tuple(str(path) for path in comparison_files)
			contour_range = (
				_preview_data_range(contour_selected)
				if data_source_mode == "ECCO-Darwin CO2 flux"
				else cached_data_range_by_key(
					_data_range_cache_key(contour_source_key, contour_variable, payload),
					contour_selected,
				)
			)
		except DatasetError as exc:
			st.error(str(exc))
			st.stop()
		st.caption(
			f"Selected scalar data range: {selected_range[0]:.12g} to {selected_range[1]:.12g}. "
			"Manual color limits may be adjusted beyond this observed range."
		)
		if contour_variable == variable and contour_ds is scalar_ds:
			st.caption("Contour data range matches the selected scalar data range.")
		else:
			st.caption(
				f"Selected contour data range: {contour_range[0]:.12g} to {contour_range[1]:.12g}. "
				"Use it as the starting range for contour levels."
			)
		st.subheader("Missing-value handling")
		st.caption("CF _FillValue, missing_value, valid_min, and valid_max metadata are applied automatically.")
		missing_lower = st.number_input("Set values below this to NaN (optional)", value=-999.0, format="%.6g")
		use_missing_lower = st.checkbox("Enable lower threshold", value=True)
		missing_upper = st.number_input("Set values above this to NaN (optional)", value=99999.0, format="%.6g")
		use_missing_upper = st.checkbox("Enable upper threshold", value=False)
		levels = selected.coords.get("level")
		level_value = None
		if levels is not None:
			level_values = [float(value) for value in levels.values]
			level_value = st.selectbox("Pressure level", level_values, index=0)
		periods = available_periods(selected)
		try:
			detected_resolution = temporal_resolution(selected)
			st.caption(
				f"Detected temporal resolution: {detected_resolution.name} "
				f"(median interval {detected_resolution.median_interval_days:.3g} days)."
			)
		except DatasetError:
			detected_resolution = None
		if periods.years:
			year_mode = st.radio("Target year selection", ["Single year", "Year range", "Custom list"], horizontal=True)
			if year_mode == "Single year":
				year = (st.selectbox("Target year", periods.years),)
			elif year_mode == "Year range":
				first_year, last_year = (
					st.select_slider("Target year range", options=periods.years, value=(periods.years[0], periods.years[-1]))
					if len(periods.years) > 1 else (periods.years[0], periods.years[0])
				)
				year = tuple(value for value in periods.years if first_year <= value <= last_year)
			else:
				year_text = st.text_input("Target years", value=str(periods.years[-1]), help="Such as 1991-2020, 2023, 2025. Ranges and single years can be mixed.")
				chosen_years, unknown_years = parse_year_spec(year_text, periods.years)
				if unknown_years:
					st.warning(f"Not in the dataset, ignored: {', '.join(map(str, unknown_years))}")
				if not chosen_years:
					st.warning("Enter at least one valid target year.")
					st.stop()
				year = tuple(chosen_years)
			if len(year) > 1:
				st.caption(f"Averaging {len(year)} years: {format_years(year)}")
		else:
			year = None
		temporal_mode = st.radio(
			"Time aggregation",
			["Monthly", "Seasonal"],
			horizontal=True,
			help="Choose whether the selected months are averaged separately by month or combined into a seasonal mean.",
		)
		month = None
		season = None
		if temporal_mode == "Monthly":
			month = st.selectbox("Month", list(range(1, 13)), index=0)
		else:
			selected_months = st.multiselect("Seasonal months", list(range(1, 13)), default=[12, 1, 2])
			if not selected_months:
				st.warning("Select at least one month for seasonal aggregation.")
				st.stop()
			season_start_month = st.selectbox("Season start month", selected_months, index=0)
			ordered_months = tuple(
				month_number
				for offset in range(12)
				for month_number in [((season_start_month - 1 + offset) % 12) + 1]
				if month_number in selected_months
			)
			season = ordered_months
		display_mode = st.radio(
			"Display",
			["Absolute value", "Anomaly", "Linear trend"],
			horizontal=True,
			help="Absolute value shows the field itself; Anomaly subtracts the baseline climatology; Linear trend shows the fitted change over time.",
		)
		if len(periods.years) > 1:
			baseline_years = st.slider("Climatology years", periods.year_min, periods.year_max, (periods.year_min, periods.year_max))
		elif periods.years:
			baseline_years = (periods.year_min, periods.year_min)
			st.caption(f"Single-year dataset: climatology baseline fixed at {periods.year_min}.")
		else:
			baseline_years = None
		detrend_enabled = st.checkbox(
			"Apply grid-cell detrending before analysis",
			help="Remove a grid-cell time-dependent component before calculating anomalies, trends, or period means.",
		)
		detrend_type = st.selectbox(
			"Detrend mode",
			["normalize", "anomaly"],
			disabled=not detrend_enabled,
			help="Normalize removes a fitted reference value; anomaly removes the fitted temporal component while preserving the original scale.",
		)
		detrend_reference_year = None
		if detrend_enabled and detrend_type == "normalize":
			detrend_reference_year = st.number_input("Detrend reference year", value=float(np.median(periods.years)) if periods.years else 2000.0, step=1.0)
		trend_method = "Student's t"
		confidence_level = 0.95
		trend_marker_size = 12.0
		trend_time_unit = "Year"
		ensemble_threshold = 0.80
		if display_mode == "Linear trend":
			trend_time_unit = st.selectbox("Trend time basis", ["Year", "Month", "Day", "Decade"])
			trend_method = st.selectbox("Trend significance test", ["Student's t", "Mann-Kendall"])
			confidence_level = st.slider("Trend confidence level", 0.80, 0.99, 0.95, step=0.01)
			trend_marker_size = st.slider("Significance marker size", 2.0, 40.0, 12.0, step=1.0)
		if ensemble_member_count > 1:
			ensemble_threshold_percent = st.slider(
				"MMM agreement threshold",
				80,
				99,
				80,
				step=1,
				format="%d%%",
				help="Robust signal is stippled when at least this fraction of valid models shares the MMM sign. Other valid cells are hatched.",
			)
			ensemble_threshold = ensemble_threshold_percent / 100.0

		st.header("Region")
		lon_convention = st.radio("Longitude convention", ["-180 to 180", "0 to 360"], horizontal=True)
		lon_minimum, lon_maximum = (-180.0, 180.0) if lon_convention == "-180 to 180" else (0.0, 360.0)
		region = st.selectbox("Preset", ["Global", "Pacific Basin", "East Asia / Philippine Sea", "South China Sea", "cross-equatorial SE Asia", "Custom"])
		presets = {
			"Global": (lon_minimum, lon_maximum, -90.0, 90.0),
			"Pacific Basin": (100.0, 300.0 if lon_convention == "0 to 360" else -60.0, -20, 60.0),
			"East Asia / Philippine Sea": (110.0, 160.0, 0.0, 55.0),
			"South China Sea": (100.0, 125.0, 0.0, 25),
			"cross-equatorial SE Asia": (90, 160, -25, 25),
		}
		if region == "Custom":
			lon_bounds = _number_range("Longitude range (min, max)", lon_minimum, lon_maximum, (lon_minimum, lon_maximum))
			lat_bounds = _number_range("Latitude range (min, max)", -90.0, 90.0, (-60.0, 60.0))
		else:
			preset = presets[region]
			lon_bounds, lat_bounds = (preset[0], preset[1]), (preset[2], preset[3])
		highlight_region = st.checkbox(
			"Highlight an additional spatial range",
			value=False,
			help="Enable a second annual-cycle figure and a geographic highlight boundary.",
		)
		highlight_bounds = None
		highlight_vertices = None
		highlight_color = "#111111"
		highlight_linewidth = 1.5
		if highlight_region:
			highlight_shape = st.radio("Highlight geometry", ["Rectangle", "Polygon"], horizontal=True)
			if highlight_shape == "Rectangle":
				highlight_lon_bounds = _number_range(
					"Highlight longitude range",
					lon_minimum,
					lon_maximum,
					lon_bounds,
				)
				highlight_lat_bounds = _number_range(
					"Highlight latitude range",
					-90.0,
					90.0,
					lat_bounds,
				)
				highlight_vertices = [
					(highlight_lon_bounds[0], highlight_lat_bounds[0]),
					(highlight_lon_bounds[1], highlight_lat_bounds[0]),
					(highlight_lon_bounds[1], highlight_lat_bounds[1]),
					(highlight_lon_bounds[0], highlight_lat_bounds[1]),
				]
				highlight_bounds = (highlight_lon_bounds, highlight_lat_bounds)
			else:
				center_lat = float(np.mean(lat_bounds))
				lon_width = (lon_bounds[1] - lon_bounds[0]) % 360.0
				center_lon = float(((lon_bounds[0] + 0.5 * lon_width + 180.0) % 360.0) - 180.0)
				map_widget = folium.Map(
					location=[center_lat, center_lon],
					zoom_start=3 if abs(lon_bounds[1] - lon_bounds[0]) > 60 else 5,
					control_scale=True,
				)
				Draw(
					export=False,
					draw_options={
						"polyline": False,
						"rectangle": False,
						"circle": False,
						"circlemarker": False,
						"marker": False,
						"polygon": True,
					},
					edit_options={"edit": True, "remove": True},
				).add_to(map_widget)
				map_event = st_folium(map_widget, height=320, key="highlight-polygon-map")
				st.caption("Enter vertices clockwise or counter-clockwise as longitude,latitude pairs.")
				drawings = map_event.get("all_drawings") if isinstance(map_event, dict) else None
				if drawings:
					coordinates = drawings[-1].get("geometry", {}).get("coordinates", [[]])[0]
					if len(coordinates) >= 4:
						drawn_text = "; ".join(f"{point[0]:.8g},{point[1]:.8g}" for point in coordinates[:-1])
					else:
						drawn_text = ""
				else:
					drawn_text = f"{lon_bounds[0]},{lat_bounds[0]}; {lon_bounds[1]},{lat_bounds[0]}; {lon_bounds[1]},{lat_bounds[1]}; {lon_bounds[0]},{lat_bounds[1]}"
				polygon_text = st.text_area(
					"Highlight polygon vertices",
					value=drawn_text,
					key="highlight-polygon-vertices",
					help="The polygon boundary is included in the spatial average. Dateline-crossing longitudes are unwrapped automatically.",
				)
				try:
					highlight_vertices = _parse_polygon_vertices(polygon_text)
				except ValueError as exc:
					st.error(f"Invalid highlight polygon: {exc}")
					highlight_vertices = None
			highlight_color = st.color_picker(
				"Highlight boundary color",
				"#111111",
				help="Choose the color of the hollow highlighted boundary.",
			)
			highlight_linewidth = st.number_input(
				"Highlight boundary line width",
				min_value=0.1,
				max_value=10.0,
				value=1.5,
				step=0.1,
				help="Set the projected boundary line thickness in points.",
			)
			if highlight_vertices and highlight_bounds is None:
				lons = [point[0] for point in highlight_vertices]
				lats = [point[1] for point in highlight_vertices]
				highlight_bounds = ((min(lons), max(lons)), (min(lats), max(lats)))
			annual_weighting = st.selectbox(
				"Annual-cycle spatial mean",
				["coslat", "equal"],
				format_func=lambda value: "cos(latitude) area-weighted" if value == "coslat" else "Equal grid-cell weights",
				help="Area weighting is recommended for geographic latitude/longitude grids.",
			)
			annual_running_mean = st.checkbox("Apply running mean", value=False)
			annual_running_window = st.number_input(
				f"Running-mean window ({detected_resolution.name if detected_resolution else 'samples'})",
				min_value=1,
				max_value=3650,
				value=3,
				step=1,
				disabled=not annual_running_mean,
				help="Smoothing is applied to the continuous time series before it is split into years.",
			)
			annual_target_color = st.color_picker(
				"Annual-cycle highlighted-line color",
				"#0072B2",
				help="Used for a single target year. Multiple target years use a distinct, colour-blind-friendly publication palette.",
			)
			annual_target_linewidth = st.number_input(
				"Annual-cycle highlighted-line width",
				min_value=0.5,
				max_value=8.0,
				value=2.2,
				step=0.1,
			)
			annual_marker_enabled = st.checkbox(
				"Add markers to annual-cycle highlighted lines",
				value=False,
			)
			annual_marker = st.selectbox(
				"Annual-cycle marker style",
				["o", "s", "^", "D", "v", "*", "P", "X"],
				disabled=not annual_marker_enabled,
			)
			annual_marker_facecolor = st.color_picker(
				"Annual-cycle marker face color",
				"#FFFFFF",
				disabled=not annual_marker_enabled,
			)
			annual_marker_size = st.number_input(
				"Annual-cycle marker size",
				min_value=0.5,
				max_value=20.0,
				value=5.0,
				step=0.5,
				disabled=not annual_marker_enabled,
			)
			annual_anchor_month = st.selectbox(
				"Annual-cycle centre month",
				list(range(1, 13)),
				index=0,
				format_func=lambda value: pd.Timestamp(2000, value, 1).strftime("%B"),
			)
			annual_month_tick_position = st.selectbox(
				"Annual-cycle month tick position",
				["start", "middle"],
				format_func=lambda value: "Month start" if value == "start" else "Month middle",
				help="For fixed multi-day products such as 8-, 11-, or 15-day data, place each month label near the first or middle day of the month.",
			)
			annual_month_shading_enabled = st.checkbox(
				"Highlight target-year months with shading",
				value=False,
				help="Highlight one or more months in the target-year [0] portion of the annual-cycle plot.",
			)
			annual_shading_months = st.multiselect(
				"Target months to shade",
				list(range(1, 13)),
				default=[],
				format_func=lambda value: pd.Timestamp(2000, value, 1).strftime("%B"),
				disabled=not annual_month_shading_enabled,
			)
			annual_month_shading_color = st.color_picker(
				"Target-month shading color",
				"#F2C14E",
				disabled=not annual_month_shading_enabled,
			)
			download_annual_netcdf = st.checkbox(
				"Offer annual-cycle NetCDF download",
				value=False,
				help="Export the spatially averaged time series used to construct each annual-cycle figure.",
			)
			other_year_range = st.slider(
				"Other-year comparison range",
				periods.year_min,
				periods.year_max,
				(periods.year_min, periods.year_max),
				disabled=not periods.years,
			) if periods.years else None
		else:
			annual_weighting = "coslat"
			annual_running_mean = False
			annual_running_window = 1
			annual_target_color = "#0072B2"
			annual_target_linewidth = 2.2
			annual_marker_enabled = False
			annual_marker = "o"
			annual_marker_facecolor = "#FFFFFF"
			annual_marker_size = 5.0
			annual_anchor_month = 1
			annual_month_tick_position = "start"
			annual_month_shading_enabled = False
			annual_shading_months = []
			annual_month_shading_color = "#F2C14E"
			download_annual_netcdf = False
			other_year_range = None

		st.header("Map style")
		land_color = st.color_picker(
			"Land color",
			"#e8e5dc",
			help="Color used for geographic land areas where the selected field is missing.",
		)
		coastline_color = st.color_picker(
			"Coastline color",
			"#202020",
			help="Color used for coastlines drawn above the land and data layers.",
		)
		projection = st.selectbox(
			"Projection",
			["Automatic (recommended)", "Equal Earth (equal-area)", "Mollweide (equal-area)", "PlateCarree (regional)", "Orthographic (hemisphere)"],
			help="Select the map projection. The data coordinates remain geographic longitude/latitude and are transformed only for display.",
		)
		cmap = st.selectbox(
			"Colormap",
			[
				"cmocean:balance", "cmocean:haline", "cmocean:thermal", "cmocean:amp",
				"cmocean:algae", "cmocean:ice", "cmocean:dense",
				"RdBu_r", "BrBG", "PiYG", "PuOr", "RdYlBu_r", "RdGy", "seismic", "coolwarm",
				"viridis", "plasma", "inferno", "magma", "cividis", "cubehelix", "YlGnBu", "YlOrRd", "PuBuGn", "GnBu", "BuPu", "OrRd", "Greens", "Blues", "Reds",
				"terrain", "ocean", "gist_earth", "gist_ncar", "hot", "pink",
			],
		)
		custom_endpoint_colors = st.checkbox("Customize colormap endpoint colors")
		cmap_upload = st.file_uploader("Upload colormap image", type=["png", "jpg", "jpeg"], help="Upload a horizontal or vertical color strip. The image is sampled into a continuous colormap.")
		cmap_low_color = st.color_picker("Low-value endpoint color", "#313695", disabled=not custom_endpoint_colors)
		cmap_high_color = st.color_picker("High-value endpoint color", "#A50026", disabled=not custom_endpoint_colors)
		color_scale = st.selectbox(
			"Color scale type",
			["Linear", "Logarithmic"],
			help="Linear scales support signed fields; logarithmic scales require strictly positive values.",
		)
		logarithmic_base = 10.0
		if color_scale == "Logarithmic":
			logarithmic_base = st.number_input("Logarithmic tick base", min_value=2.0, value=10.0, step=1.0)
		scale_mode = st.selectbox(
			"Color scale",
			["Robust percentile", "Data min/max", "Manual strict limits"],
			help="Choose how the color limits are calculated. Robust percentile limits reduce the influence of outliers.",
		)
		vmin = vmax = None
		if scale_mode == "Manual strict limits":
			vmin = st.number_input(
				"Color minimum",
				value=float(selected_range[0]),
				format="%.12g",
				help=f"Observed scalar range: {selected_range[0]:.12g} to {selected_range[1]:.12g}.",
			)
			vmax = st.number_input(
				"Color maximum",
				value=float(selected_range[1]),
				format="%.12g",
				help=f"Observed scalar range: {selected_range[0]:.12g} to {selected_range[1]:.12g}.",
			)
		colorbar_tick_mode = st.selectbox("Colorbar tick interval", ["Automatic", "Custom"])
		colorbar_tick_interval = None
		if colorbar_tick_mode == "Custom":
			default_tick_interval = max(abs(selected_range[1] - selected_range[0]) / 10.0, 1e-12)
			colorbar_tick_interval = st.number_input(
				"Tick interval",
				min_value=1e-12,
				value=float(default_tick_interval),
				format="%.12g",
				help="Choose the spacing between colorbar ticks. Scientific labels retain enough digits to distinguish these values.",
			)
		colorbar_tick_format = st.selectbox(
			"Colorbar tick label format",
			["Automatic", "Integer", "One decimal", "Two decimals", "Scientific"],
		)
		download_colorbar = st.checkbox(
			"Download standalone transparent colorbar PNG",
			help="Uses the final map color scale, limits, colormap, and tick settings.",
		)
		colorbar_orientation = st.radio(
			"Standalone colorbar orientation",
			["horizontal", "vertical"],
			horizontal=True,
			disabled=not download_colorbar,
		)
		colorbar_tick_rotation = st.number_input(
			"Standalone colorbar tick-label rotation (degrees)",
			min_value=-180.0,
			max_value=180.0,
			value=0.0,
			step=1.0,
			disabled=not download_colorbar,
		)
		colorbar_label_location = st.selectbox(
			"Standalone display-unit label position",
			["auto", "top", "bottom"] if colorbar_orientation == "horizontal" else ["auto", "left", "right"],
			disabled=not download_colorbar,
			help="Horizontal bars support auto/top/bottom; vertical bars support auto/left/right.",
		)
		colorbar_label_rotation = st.number_input(
			"Standalone display-unit label rotation (degrees)",
			min_value=-180.0,
			max_value=180.0,
			value=0.0,
			step=1.0,
			disabled=not download_colorbar,
		)
		colorbar_label_size = st.number_input(
			"Standalone display-unit label font size",
			min_value=1.0,
			max_value=72.0,
			value=12.0,
			step=1.0,
			disabled=not download_colorbar,
		)
		colorbar_label_font = st.selectbox(
			"Standalone display-unit label font",
			["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"],
			disabled=not download_colorbar,
		)
		colorbar_label_color = st.color_picker(
			"Standalone display-unit label color",
			"#000000",
			disabled=not download_colorbar,
		)
		st.subheader("Fill style")
		fill_style = st.radio(
			"Shading method",
			["Contourf", "Pcolormesh (continuous)", "Pcolormesh (discrete)"],
			horizontal=True,
			help="Contourf: interpolate contour levels, edges have half a grid. Pcolormesh: each grid point has a complete square.",
		)
		if fill_style == "Pcolormesh (continuous)":
			st.caption("Continuous color scale: \"Number of contour levels\" only affects the overlaid contour lines.")
		elif fill_style == "Pcolormesh (discrete)":
			st.caption("Discrete color scale: \"Number of contour levels\" determines the number of color blocks.")
		contour_display = st.radio(
			"Contour display",
			["Do not draw contours", "Lines without labels", "Lines with labels", "Specified values"],
			horizontal=True,
		)
		show_contours = contour_display != "Do not draw contours"
		show_contour_labels = contour_display == "Lines with labels"
		contour_count = 20
		contour_min = contour_max = contour_interval = None
		contour_levels = None
		if contour_display == "Specified values":
			contour_values_text = st.text_input(
				"Contour values",
				value="0, 1, 2",
				help="Enter finite contour values separated by commas, spaces, or semicolons.",
			)
			try:
				contour_levels = tuple(sorted({
					float(value)
					for value in contour_values_text.replace(";", ",").split(",")
					if value.strip()
				}))
				if not contour_levels:
					raise ValueError("Enter at least one contour value.")
			except ValueError as exc:
				st.error(f"Invalid contour values: {exc}")
				contour_levels = None
		elif show_contours:
			contour_mode = st.radio("Contour level definition", ["Automatic", "Specific range"], horizontal=True)
		else:
			contour_mode = "Automatic"
		if show_contours and contour_display != "Specified values" and contour_mode == "Automatic":
			contour_count = st.slider("Number of contour levels", 8, 40, 20)
		elif show_contours and contour_display != "Specified values":
			contour_min = st.number_input(
				"Contour minimum",
				value=float(contour_range[0]),
				format="%.12g",
				help=f"Observed contour range: {contour_range[0]:.12g} to {contour_range[1]:.12g}.",
			)
			contour_max = st.number_input(
				"Contour maximum",
				value=float(contour_range[1]),
				format="%.12g",
				help=f"Observed contour range: {contour_range[0]:.12g} to {contour_range[1]:.12g}.",
			)
			default_contour_interval = max(abs(contour_range[1] - contour_range[0]) / 10.0, 1e-12)
			contour_interval = st.number_input(
				"Contour interval",
				min_value=1e-12,
				value=float(default_contour_interval),
				format="%.12g",
			)
			st.caption("The plotted levels are minimum, minimum + interval, ..., maximum.")
		contour_color = "#404040"
		contour_linestyle = "-"
		contour_linewidth = 0.35
		if show_contours:
			contour_color = st.color_picker("Contour line color", "#404040")
			contour_linestyle = st.selectbox("Contour line style", ["-", "--", "-.", ":"])
			contour_linewidth = st.number_input("Contour line width", min_value=0.1, max_value=8.0, value=0.35, step=0.05)
		st.subheader("Map orientation and scale")
		show_north_arrow = st.checkbox("Show north arrow")
		show_scale_bar = st.checkbox("Show geographic scale bar")
		scale_bar_unit = st.radio("Scale-bar label unit", ["kilometer", "meter"], horizontal=True, disabled=not show_scale_bar)
		st.subheader("Value conversion")
		scalar_transformation, scalar_operand, scalar_output_units = _value_conversion_controls(
			"Pcolor / primary plotted field", "scalar"
		)
		contour_transformation, contour_operand, contour_output_units = _value_conversion_controls(
			"Contour field", "contour"
		)
		vector_transformation, vector_operand, vector_output_units = _value_conversion_controls(
			"Vector field", "vector"
		)
		show_quiver = st.checkbox(
			"Overlay quiver vectors",
			help="Draw sampled U/V vectors as arrows. Select the U and V components below.",
		)
		show_streamplot = st.checkbox(
			"Overlay streamlines",
			help="Draw continuous flow lines from the U/V field. Streamlines require a rectilinear grid.",
		)
		show_vectors = show_quiver or show_streamplot
		vector_units = st.text_input(
			"Quiver unit label (LaTeX, optional)",
			value="",
			disabled=not show_vectors,
			help=r"Use LaTeX math text for the quiver reference, for example \mathrm{m\,s^{-1}}. Leave blank to use the U-component unit.",
		)
		stride = st.slider(
			"Vector stride",
			1,
			20,
			10,
			disabled=not show_vectors,
			help="Controls vector sampling density. A value of 1 draws every grid point; larger values skip grid points and draw fewer arrows.",
		)
		vector_scale = st.slider(
			"Vector scale",
			5,
			500,
			300,
			disabled=not show_quiver,
			help="Controls the displayed arrow length. It changes arrow size, not the number of vectors or their physical values.",
		)
		vector_reference = st.number_input(
			"Reference arrow magnitude (0 = automatic median)",
			min_value=0.0,
			value=0.0,
			disabled=not show_quiver,
			help="The reference arrow is drawn below the map. Its label uses the same units as the U component.",
		)
		compute_wind_curl = st.checkbox("Compute wind stress curl from U/V", help="Compute wind stress curl before temporal aggregation using the Large-Pond drag formulation.")
		u_name = v_name = None
		annual_source_choice = "Pcolor"
		if show_vectors or compute_wind_curl:
			vector_variables = list(vector_ds.data_vars)
			candidates = [name for name in vector_variables if name != variable]
			u_name = st.selectbox(
				"U component",
				candidates,
				index=0,
				help="Select the vector component in the longitudinal or x direction.",
			) if candidates else None
			v_candidates = [name for name in candidates if name != u_name]
			v_name = st.selectbox(
				"V component",
				v_candidates,
				index=0,
				help="Select the vector component in the latitudinal or y direction.",
			) if v_candidates else None
		if highlight_region:
			annual_source_options = ["Pcolor", "Contour"]
			if u_name and v_name:
				annual_source_options.extend(["Quiver U", "Quiver V", "Quiver speed", "Quiver U and V"])
			annual_source_choice = st.radio(
				"Annual-cycle data source",
				annual_source_options,
				horizontal=True,
				help="Choose which field supplies the highlight-region annual cycle. Quiver speed is sqrt(U² + V²).",
			)
		show_hovmoller = st.checkbox(
			"Display Hovmoller diagram",
			help="Show a time-versus-space section by averaging the field along the direction perpendicular to the selected axis.",
		)
		hovmoller_axis = st.radio("Hovmoller axis", ["Longitude", "Latitude"], horizontal=True, disabled=not show_hovmoller)
		hovmoller_flip_x = st.checkbox(
			"Reverse Hovmoller x-axis",
			value=False,
			disabled=not show_hovmoller,
			help="Reverse longitude or latitude ordering in the displayed diagram.",
		)
		hovmoller_flip_y = st.checkbox(
			"Reverse Hovmoller y-axis",
			value=False,
			disabled=not show_hovmoller,
			help="Reverse analysis-year ordering in the displayed diagram.",
		)
		download_hovmoller_netcdf = st.checkbox(
			"Offer Hovmoller NetCDF download",
			value=False,
			disabled=not show_hovmoller,
		)
		hovmoller_region = st.radio("Hovmoller spatial bounds", ["Current map region", "Custom"], horizontal=True, disabled=not show_hovmoller)
		if show_hovmoller and hovmoller_region == "Custom":
			hov_lon_bounds = _number_range("Hovmoller longitude range", lon_minimum, lon_maximum, lon_bounds)
			hov_lat_bounds = _number_range("Hovmoller latitude range", -90.0, 90.0, lat_bounds)
		else:
			hov_lon_bounds, hov_lat_bounds = lon_bounds, lat_bounds
		show_section = st.checkbox("Display perspective vertical section", disabled="level" not in selected.dims)
		section_axis = st.selectbox("Section orientation", ["Longitude", "Latitude", "Oblique"], disabled=not show_section)
		section_coordinate = None
		section_start = section_end = None
		if show_section and section_axis == "Longitude":
			section_coordinate = st.number_input("Section latitude", min_value=-90.0, max_value=90.0, value=float(np.mean(lat_bounds)), step=1.0)
		elif show_section and section_axis == "Latitude":
			section_coordinate = st.number_input("Section longitude", min_value=lon_minimum, max_value=lon_maximum, value=float(np.mean(lon_bounds)), step=1.0)
		elif show_section:
			section_start = (
				st.number_input("Oblique start longitude", min_value=lon_minimum, max_value=lon_maximum, value=float(lon_bounds[0]), step=1.0),
				st.number_input("Oblique start latitude", min_value=-90.0, max_value=90.0, value=float(lat_bounds[0]), step=1.0),
			)
			section_end = (
				st.number_input("Oblique end longitude", min_value=lon_minimum, max_value=lon_maximum, value=float(lon_bounds[1]), step=1.0),
				st.number_input("Oblique end latitude", min_value=-90.0, max_value=90.0, value=float(lat_bounds[1]), step=1.0),
			)
		section_elevation = st.slider("Section view elevation", -10.0, 80.0, 25.0, step=1.0, disabled=not show_section)
		section_azimuth = st.slider("Section view azimuth", -180.0, 180.0, -60.0, step=1.0, disabled=not show_section)
	if not auto_update:
		if not st.button("Update plot", type="primary"):
			st.info("Adjust the controls, then click Update plot to render the figures.")
			st.stop()

	try:
		if compute_wind_curl:
			if not u_name or not v_name:
				raise DatasetError("Wind stress curl requires both U and V component variables.")
			analysis_source = wind_stress_curl(
				mask_invalid_values(vector_ds[u_name], missing_lower if use_missing_lower else None, missing_upper if use_missing_upper else None),
				mask_invalid_values(vector_ds[v_name], missing_lower if use_missing_lower else None, missing_upper if use_missing_upper else None),
			)
		else:
			analysis_source = selected
		analysis_selected = _prepare_analysis_source(
			analysis_source,
			lon_bounds,
			lat_bounds,
			lon_convention,
			missing_lower,
			missing_upper,
			use_missing_lower,
			use_missing_upper,
			detrend_enabled,
			detrend_type,
			detrend_reference_year,
			year,
			baseline_years,
			month,
			season,
		)
		trend_significant = None
		vector_significant = None
		ensemble_agreement = None
		if display_mode == "Linear trend":
			trend_input = select_region(analysis_selected, lon_bounds, lat_bounds, lon_convention)
			scalar, _, trend_significant = calculate_linear_trend(
				trend_input,
				baseline_years,
				month=month,
				season=season,
				level=level_value,
				confidence_level=confidence_level,
				significance_method=trend_method,
				time_unit=trend_time_unit,
			)
		elif display_mode == "Anomaly":
			member_field = calculate_anomaly(
				analysis_selected,
				year,
				month=month,
				season=season,
				level=level_value,
				baseline_years=baseline_years,
				collapse_ensemble=ensemble_member_count <= 1,
			)
			if ensemble_member_count > 1:
				ensemble_agreement = ensemble_agreement_fraction(member_field)
				scalar = member_field.mean("ensemble", skipna=True)
			else:
				scalar = member_field
		else:
			member_field = select_period(
				analysis_selected,
				year,
				month=month,
				season=season,
				level=level_value,
				collapse_ensemble=ensemble_member_count <= 1,
			)
			if ensemble_member_count > 1:
				ensemble_agreement = ensemble_agreement_fraction(member_field)
				scalar = member_field.mean("ensemble", skipna=True)
			else:
				scalar = member_field
		if scalar_transformation == "Automatic CF unit conversion":
			scalar = convert_display_units(scalar, anomaly=display_mode != "Absolute value")
		else:
			scalar = transform_values(scalar, scalar_transformation, scalar_operand, scalar_output_units)
		scalar = _apply_display_units(scalar, scalar_output_units)
		scalar = select_region(scalar, lon_bounds, lat_bounds, lon_convention)
		scalar_range_after_processing = finite_data_range(scalar)
		st.caption(
			f"Final pcolor range after detrending/conversion: "
			f"{scalar_range_after_processing[0]:.12g} to {scalar_range_after_processing[1]:.12g}."
		)
		contour_field = None
		if show_contours:
			contour_source_field = _prepare_analysis_source(
				contour_selected,
				lon_bounds,
				lat_bounds,
				lon_convention,
				missing_lower,
				missing_upper,
				use_missing_lower,
				use_missing_upper,
				detrend_enabled,
				detrend_type,
				detrend_reference_year,
				year,
				baseline_years,
				month,
				season,
			)
			contour_level = level_value if "level" in contour_source_field.coords else None
			if display_mode == "Linear trend":
				contour_field, _, _ = calculate_linear_trend(
					contour_source_field,
					baseline_years,
					month=month,
					season=season,
					level=contour_level,
					confidence_level=confidence_level,
					significance_method=trend_method,
					time_unit=trend_time_unit,
				)
			elif display_mode == "Anomaly":
				contour_field = calculate_anomaly(
					contour_source_field,
					year,
					month=month,
					season=season,
					level=contour_level,
					baseline_years=baseline_years,
				)
			else:
				contour_field = select_period(
					contour_source_field,
					year,
					month=month,
					season=season,
					level=contour_level,
				)
			if contour_transformation == "Automatic CF unit conversion":
				contour_field = convert_display_units(contour_field, anomaly=display_mode != "Absolute value")
			else:
				contour_field = transform_values(contour_field, contour_transformation, contour_operand, contour_output_units)
			contour_field = _apply_display_units(contour_field, contour_output_units)
			contour_field = select_region(contour_field, lon_bounds, lat_bounds, lon_convention)
			contour_range_after_processing = finite_data_range(contour_field)
			st.caption(
				f"Final contour range after detrending/conversion: "
				f"{contour_range_after_processing[0]:.12g} to {contour_range_after_processing[1]:.12g}."
			)
		u_field = v_field = None
		if show_vectors and u_name and v_name:
			if display_mode == "Linear trend":
				u_input = _prepare_analysis_source(vector_ds[u_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season)
				v_input = _prepare_analysis_source(vector_ds[v_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season)
				u_field, _, u_significant = calculate_linear_trend(u_input, baseline_years, month=month, season=season, level=level_value, confidence_level=confidence_level, significance_method=trend_method, time_unit=trend_time_unit)
				v_field, _, v_significant = calculate_linear_trend(v_input, baseline_years, month=month, season=season, level=level_value, confidence_level=confidence_level, significance_method=trend_method, time_unit=trend_time_unit)
				vector_significant = u_significant & v_significant
			elif display_mode == "Anomaly":
				u_field = calculate_anomaly(_prepare_analysis_source(vector_ds[u_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season), year, month=month, season=season, level=level_value, baseline_years=baseline_years)
				v_field = calculate_anomaly(_prepare_analysis_source(vector_ds[v_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season), year, month=month, season=season, level=level_value, baseline_years=baseline_years)
			else:
				u_field = select_period(_prepare_analysis_source(vector_ds[u_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season), year, month=month, season=season, level=level_value)
				v_field = select_period(_prepare_analysis_source(vector_ds[v_name], lon_bounds, lat_bounds, lon_convention, missing_lower, missing_upper, use_missing_lower, use_missing_upper, detrend_enabled, detrend_type, detrend_reference_year, year, baseline_years, month, season), year, month=month, season=season, level=level_value)
			if vector_transformation == "Automatic CF unit conversion":
				u_field = convert_display_units(u_field, anomaly=display_mode != "Absolute value")
				v_field = convert_display_units(v_field, anomaly=display_mode != "Absolute value")
			else:
				u_field = transform_values(u_field, vector_transformation, vector_operand, vector_output_units)
				v_field = transform_values(v_field, vector_transformation, vector_operand, vector_output_units)
			u_field = select_region(u_field, lon_bounds, lat_bounds, lon_convention)
			v_field = select_region(v_field, lon_bounds, lat_bounds, lon_convention)
			u_field = _apply_display_units(u_field, vector_output_units or vector_units)
			v_field = _apply_display_units(v_field, vector_output_units or vector_units)
		reference = vector_reference if vector_reference > 0 else None
		plot_options = PlotOptions(
			projection,
			cmap,
			contour_count,
			vmin,
			vmax,
			"quiver" if show_quiver else "streamplot",
			stride,
			vector_scale,
			scale_mode,
			contour_min,
			contour_max,
			contour_interval,
			vector_reference=reference,
				center_colorbar_on_zero=display_mode in {"Anomaly", "Linear trend"},
				colorbar_tick_interval=colorbar_tick_interval,
				colorbar_tick_format=colorbar_tick_format,
				color_scale=color_scale,
				logarithmic_base=logarithmic_base,
				show_quiver=show_quiver,
				show_streamplot=show_streamplot,
				vector_significant=vector_significant,
				trend_significant=trend_significant,
				trend_marker_size=trend_marker_size,
				vector_label=u_name,
				cmap_low_color=cmap_low_color if custom_endpoint_colors else None,
				cmap_high_color=cmap_high_color if custom_endpoint_colors else None,
				cmap_image=cmap_upload.getvalue() if cmap_upload else None,
				figure_annotation=(f"ensemble members={ensemble_member_count}\n" if ensemble_member_count > 1 else "") + ("field=wind stress curl (N m-3)\n" if compute_wind_curl else "") + (f"vector stride={stride} (every {stride}th grid point)\n" if show_vectors else "") + _analysis_annotation(
					display_mode, temporal_mode, year, baseline_years, month, season,
					season_start_month if temporal_mode == "Seasonal" else None,
					lon_bounds, lat_bounds, lon_convention, detrend_enabled, detrend_type,
					scalar_transformation, trend_time_unit,
				),
				fill_style=fill_style,
				highlight_bounds=highlight_bounds,
				highlight_vertices=tuple(highlight_vertices) if highlight_vertices else None,
				plotted_bounds=(lon_bounds, lat_bounds),
				highlight_color=highlight_color,
				highlight_linewidth=highlight_linewidth,
				land_color=land_color,
				coastline_color=coastline_color,
				ensemble_agreement=ensemble_agreement,
				agreement_threshold=ensemble_threshold,
				show_contours=show_contours,
				show_contour_labels=show_contour_labels,
				contour_levels=contour_levels,
				contour_color=contour_color,
				contour_linestyle=contour_linestyle,
				contour_linewidth=contour_linewidth,
				show_scale_bar=show_scale_bar,
				scale_bar_unit=scale_bar_unit,
				show_north_arrow=show_north_arrow,
		)
		figure = draw_map(
			scalar,
			u_field,
			v_field,
			plot_options,
			contour_field=contour_field,
		)
	except (DatasetError, OSError, ValueError, KeyError) as exc:
		st.warning(
			"The selected operation cannot be completed because the remote data service "
			f"is temporarily unavailable or the selection is invalid: {exc}"
		)
		st.stop()

	st.pyplot(figure, clear_figure=False, use_container_width=True)
	png = io.BytesIO()
	figure.savefig(png, format="png", dpi=300, bbox_inches="tight")
	st.download_button(
		"Download 300 DPI PNG",
		png.getvalue(),
		"climate_map.png",
		"image/png",
		on_click="ignore",
	)
	png.close()
	del png
	pdf = io.BytesIO()
	figure.savefig(pdf, format="pdf", bbox_inches="tight")
	st.download_button(
		"Download publication PDF",
		pdf.getvalue(),
		"climate_map.pdf",
		"application/pdf",
		on_click="ignore",
	)
	pdf.close()
	del pdf
	if download_colorbar:
		colorbar_figure = None
		try:
			colorbar_figure = draw_colorbar(
				scalar,
				plot_options,
				orientation=colorbar_orientation,
				tick_rotation=float(colorbar_tick_rotation),
				label=_units_label_for_export(
					scalar_output_units
					or str(scalar.attrs.get("units_latex", scalar.attrs.get("units", "")))
				),
				label_location=colorbar_label_location,
				label_rotation=float(colorbar_label_rotation),
				label_size=float(colorbar_label_size),
				label_font=colorbar_label_font,
				label_color=colorbar_label_color,
			)
			colorbar_png = io.BytesIO()
			colorbar_figure.savefig(
				colorbar_png,
				format="png",
				dpi=300,
				bbox_inches="tight",
				transparent=True,
			)
			st.download_button(
				"Download standalone colorbar PNG",
				colorbar_png.getvalue(),
				"colorbar.png",
				"image/png",
				on_click="ignore",
			)
			colorbar_png.close()
			del colorbar_png
		except (DatasetError, OSError, ValueError, KeyError) as exc:
			st.warning(f"The standalone colorbar cannot be generated: {exc}")
		finally:
			if colorbar_figure is not None:
				plt.close(colorbar_figure)
	if highlight_region and highlight_vertices:
		annual_figure = None
		try:
			def prepare_annual_source(source, conversion):
				transformation, operand, output_units = conversion
				source = mask_invalid_values(
					source,
					lower=missing_lower if use_missing_lower else None,
					upper=missing_upper if use_missing_upper else None,
				)
				source = select_region(source, highlight_bounds[0], highlight_bounds[1], lon_convention) if highlight_bounds else source
				if detrend_enabled:
					source = detrend_data(source, detrend_type, detrend_reference_year)
				if transformation == "Automatic CF unit conversion":
					return convert_display_units(source, anomaly=display_mode != "Absolute value")
				return transform_values(source, transformation, operand, output_units)

			annual_sources = []
			if annual_source_choice in {"Pcolor", "Contour"}:
				conversion = (
					(scalar_transformation, scalar_operand, scalar_output_units)
					if annual_source_choice == "Pcolor"
					else (contour_transformation, contour_operand, contour_output_units)
				)
				source = prepare_annual_source(
					analysis_source if annual_source_choice == "Pcolor" else contour_selected,
					conversion,
				)
				annual_sources.append((annual_source_choice, source))
			elif annual_source_choice == "Quiver U":
				annual_sources.append(("Quiver U", prepare_annual_source(
					vector_ds[u_name], (vector_transformation, vector_operand, vector_output_units)
				)))
			elif annual_source_choice == "Quiver V":
				annual_sources.append(("Quiver V", prepare_annual_source(
					vector_ds[v_name], (vector_transformation, vector_operand, vector_output_units)
				)))
			elif annual_source_choice == "Quiver speed":
				annual_sources.append((
					"Quiver speed",
					prepare_annual_source(
						vector_magnitude(vector_ds[u_name], vector_ds[v_name]),
						(vector_transformation, vector_operand, vector_output_units),
					),
				))
			else:
				annual_sources.extend([
					("Quiver U", prepare_annual_source(
						vector_ds[u_name], (vector_transformation, vector_operand, vector_output_units)
					)),
					("Quiver V", prepare_annual_source(
						vector_ds[v_name], (vector_transformation, vector_operand, vector_output_units)
					)),
				])

			for source_index, (source_label, annual_source) in enumerate(annual_sources):
				annual_series = polygon_mean_timeseries(
					annual_source,
					highlight_vertices,
					annual_weighting,
					level=level_value,
				)
				resolution = temporal_resolution(annual_series)
				if resolution.name == "yearly":
					st.info(f"{source_label} data are yearly; a monthly annual-cycle figure is not applicable.")
					continue
				annual_result = build_annual_cycles(
					annual_series,
					year or (),
					baseline_years,
					other_year_range,
					anchor_month=annual_anchor_month,
					running_mean_window=int(annual_running_window) if annual_running_mean else 1,
					month_tick_position=annual_month_tick_position,
				)
				for warning in annual_result.warnings:
					st.warning(f"{source_label}: {warning}")
				annual_unit_label = annual_series.attrs.get("units") or (
					scalar_output_units if source_label == "Pcolor" else
					contour_output_units if source_label == "Contour" else
					vector_output_units
				) or "Value"
				st.subheader(f"Highlight-region annual cycles — {source_label}")
				annual_figure = draw_annual_cycles(
					annual_result,
					title=f"{source_label} inside highlight region",
					unit_label=annual_unit_label,
					anchor_month=annual_anchor_month,
					running_mean_window=int(annual_running_window) if annual_running_mean else 1,
					weighting=annual_weighting,
					target_color=annual_target_color,
					target_linewidth=float(annual_target_linewidth),
					target_marker=annual_marker if annual_marker_enabled else None,
					target_marker_facecolor=annual_marker_facecolor,
					target_marker_size=float(annual_marker_size),
					highlighted_months=annual_shading_months if annual_month_shading_enabled else None,
					month_shading_color=annual_month_shading_color,
				)
				st.pyplot(annual_figure, clear_figure=False, use_container_width=True)
				file_stem = f"highlight_annual_cycles_{source_label.lower().replace(' ', '_')}"
				annual_png = io.BytesIO()
				annual_figure.savefig(annual_png, format="png", dpi=300, bbox_inches="tight")
				st.download_button(
					"Download annual-cycle PNG",
					annual_png.getvalue(),
					f"{file_stem}.png",
					"image/png",
					key=f"annual-png-{source_index}",
					on_click="ignore",
				)
				annual_png.close()
				del annual_png
				annual_pdf = io.BytesIO()
				annual_figure.savefig(annual_pdf, format="pdf", bbox_inches="tight")
				st.download_button(
					"Download annual-cycle PDF",
					annual_pdf.getvalue(),
					f"{file_stem}.pdf",
					"application/pdf",
					key=f"annual-pdf-{source_index}",
					on_click="ignore",
				)
				annual_pdf.close()
				del annual_pdf
				if download_annual_netcdf:
					try:
						annual_bytes = dataset_to_netcdf_bytes(
							annual_series.to_dataset(name=annual_series.name or "annual_cycle_source")
						)
						st.download_button(
							"Download annual-cycle NetCDF",
							annual_bytes,
							f"{file_stem}.nc",
							"application/x-netcdf",
							key=f"annual-netcdf-{source_index}",
							on_click="ignore",
						)
					except DatasetError as exc:
						st.warning(f"The annual-cycle NetCDF cannot be generated: {exc}")
				plt.close(annual_figure)
				annual_figure = None
		except (DatasetError, OSError, ValueError, KeyError) as exc:
			st.warning(f"The highlight-region annual cycle cannot be generated: {exc}")
		finally:
			if annual_figure is not None:
				plt.close(annual_figure)
	if ensemble_member_count > 1:
		try:
			ensemble_bytes = dataset_to_netcdf_bytes(ensemble_mean_dataset(ds))
			st.download_button(
				"Download all-time ensemble-mean NetCDF",
				ensemble_bytes,
				"ensemble_mean_all_available_times.nc",
				"application/x-netcdf",
				on_click="ignore",
				help="Arithmetic mean across ensemble folders, retaining every available time and spatial coordinate.",
			)
		except DatasetError as exc:
			st.warning(f"The ensemble-mean NetCDF cannot be generated: {exc}")
	plt.close(figure)
	if show_hovmoller:
		try:
			hovmoller = build_hovmoller(
				analysis_selected,
				baseline_years,
				"lon" if hovmoller_axis == "Longitude" else "lat",
				hov_lon_bounds,
				hov_lat_bounds,
				lon_convention,
				month=month,
				season=season,
				level=level_value,
			)
			if scalar_transformation == "Automatic CF unit conversion":
				hovmoller = convert_display_units(hovmoller, anomaly=False)
			else:
				hovmoller = transform_values(hovmoller, scalar_transformation, scalar_operand, scalar_output_units)
			st.subheader("Hovmoller diagram")
			hovmoller_figure = draw_hovmoller(
				hovmoller,
				"lon" if hovmoller_axis == "Longitude" else "lat",
				cmap=cmap,
				color_scale=color_scale,
				vmin=vmin if scalar_transformation == "No conversion" else None,
				vmax=vmax if scalar_transformation == "No conversion" else None,
				tick_format=colorbar_tick_format,
				low_color=cmap_low_color if custom_endpoint_colors else None,
				high_color=cmap_high_color if custom_endpoint_colors else None,
				image_bytes=cmap_upload.getvalue() if cmap_upload else None,
				annotation=("field=wind stress curl (N m-3)\n" if compute_wind_curl else "") + "Hovmoller | " + _analysis_annotation(
					display_mode, temporal_mode, year, baseline_years, month, season,
					season_start_month if temporal_mode == "Seasonal" else None,
					hov_lon_bounds, hov_lat_bounds, lon_convention, detrend_enabled, detrend_type,
					scalar_transformation, trend_time_unit,
				),
				fill_style=fill_style,
				flip_x=hovmoller_flip_x,
				flip_y=hovmoller_flip_y,
			)
			st.pyplot(hovmoller_figure, clear_figure=False, use_container_width=True)
			if download_hovmoller_netcdf:
				hovmoller_bytes = dataset_to_netcdf_bytes(
					hovmoller.to_dataset(name=hovmoller.name or "hovmoller")
				)
				st.download_button(
					"Download Hovmoller NetCDF",
					hovmoller_bytes,
					"hovmoller.nc",
					"application/x-netcdf",
					on_click="ignore",
				)
			plt.close(hovmoller_figure)
		except (DatasetError, OSError, ValueError, KeyError) as exc:
			st.warning(f"The Hovmoller diagram cannot be generated: {exc}")
	if show_section:
		try:
			section = select_vertical_section(
				analysis_selected,
				year,
				section_axis,
				coordinate=section_coordinate,
				start=section_start,
				end=section_end,
				month=month,
				season=season,
			)
			if scalar_transformation == "Automatic CF unit conversion":
				section = convert_display_units(section, anomaly=False)
			else:
				section = transform_values(section, scalar_transformation, scalar_operand, scalar_output_units)
			st.subheader("Perspective vertical section")
			section_figure = draw_vertical_section(
				section,
				section_axis,
				cmap=cmap,
				color_scale=color_scale,
				view_elevation=section_elevation,
				view_azimuth=section_azimuth,
				low_color=cmap_low_color if custom_endpoint_colors else None,
				high_color=cmap_high_color if custom_endpoint_colors else None,
				image_bytes=cmap_upload.getvalue() if cmap_upload else None,
				annotation="Vertical section | " + _analysis_annotation(
					display_mode, temporal_mode, year, baseline_years, month, season,
					season_start_month if temporal_mode == "Seasonal" else None,
					lon_bounds, lat_bounds, lon_convention, detrend_enabled, detrend_type,
					scalar_transformation, trend_time_unit,
				),
			)
			st.pyplot(section_figure, clear_figure=False, use_container_width=True)
			plt.close(section_figure)
		except (DatasetError, OSError, ValueError, KeyError) as exc:
			st.warning(f"The vertical section cannot be generated: {exc}")
	with st.expander("Dataset metadata"):
		st.write({"source": source_key, "dimensions": dict(ds.sizes), "coordinates": coord_names, "variables": variables})


if __name__ == "__main__":
	if not st.runtime.exists():
		raise SystemExit(
			"This is a Streamlit app. Start it with: "
			"streamlit run data_view.py"
		)
	main()

# %%
