"""Publication-oriented Cartopy rendering for scalar and vector fields."""

from __future__ import annotations
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm, Normalize, TwoSlopeNorm, to_rgba
import calendar
import io
from dataclasses import dataclass

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm, to_rgba
from matplotlib.ticker import FormatStrFormatter, LogFormatter, LogLocator
import numpy as np
import xarray as xr

from shared_helpers.publication_colors import distinct_publication_colors

plt.rcParams["axes.unicode_minus"] = True


def _annual_cycle_ticks(result):
    """Return tick metadata while tolerating results from an older helper."""
    month_positions = getattr(result, "month_tick_positions", None)
    month_labels = getattr(result, "month_tick_labels", None)
    day_positions = getattr(result, "day_tick_positions", None)
    day_labels = getattr(result, "day_tick_labels", None)
    if month_positions is not None and month_labels is not None:
        return month_positions, month_labels, day_positions, day_labels

    labels = tuple(getattr(result, "labels", ()))
    x = np.asarray(result.x, dtype=float)
    if result.resolution == "monthly":
        if len(labels) != 12:
            labels = tuple(
                np.datetime_as_string(np.datetime64(f"2000-{month:02d}-01"), unit="M")[-2:]
                for month in range(1, 13)
            )
        return np.arange(min(12, len(x)), dtype=float), labels, np.array([]), ()
    if result.resolution == "daily":
        month_positions = np.linspace(0.0, max(0.0, len(x) - 1.0), 12)
        month_labels = labels[:12] if len(labels) >= 12 else tuple(
            str(month) for month in range(1, 13)
        )
        day_positions = np.linspace(0.0, max(0.0, len(x) - 1.0), 13)[:-1]
        day_labels = tuple(str(int(value)) for value in np.linspace(1, 366, 13)[:-1])
        return month_positions, month_labels, day_positions, day_labels
    return np.array([]), (), np.array([]), ()


def _target_month_spans(result, months):
    """Return x-axis spans for selected calendar months in the target year."""
    if not months:
        return []
    month_names = {
        month: np.datetime_as_string(
            np.datetime64(f"2000-{month:02d}-01"), unit="M"
        )
        for month in range(1, 13)
    }
    month_names = {month: calendar.month_abbr[month] for month in month_names}
    positions, labels, _, _ = _annual_cycle_ticks(result)
    target = [
        float(position)
        for position, label in zip(positions, labels)
        if label.endswith(" [0]")
    ]
    if not target:
        return []
    selected_positions = []
    for month in months:
        if not 1 <= int(month) <= 12:
            raise ValueError("Highlighted annual-cycle months must be between 1 and 12.")
        prefix = month_names[int(month)]
        selected_positions.extend(
            float(position)
            for position, label in zip(positions, labels)
            if label.startswith(prefix) and label.endswith(" [0]")
        )
    target = sorted(set(target))
    spans = []
    for position in sorted(set(selected_positions)):
        index = target.index(position)
        left = (target[index - 1] + position) / 2 if index else position - 0.5
        right = (position + target[index + 1]) / 2 if index + 1 < len(target) else position + 0.5
        spans.append((left, right))
    return spans


def draw_annual_cycles(
    result,
    title: str = "Annual cycles",
    unit_label: str | None = None,
    anchor_month: int = 1,
    running_mean_window: int = 1,
    weighting: str = "coslat",
    target_color: str = "#0072B2",
    target_linewidth: float = 2.2,
    highlighted_months=None,
    month_shading_color: str = "#F2C14E",
    target_marker: str | None = None,
    target_marker_facecolor: str = "white",
    target_marker_size: float = 5.0,
):
    """Render a restrained publication-style annual-cycle comparison plot."""
    if not np.isfinite(target_linewidth) or target_linewidth <= 0:
        raise ValueError("The highlighted-line width must be finite and greater than zero.")
    try:
        to_rgba(month_shading_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("The month-shading color must be a valid Matplotlib color.") from exc
    if target_marker is not None and not np.isfinite(target_marker_size):
        raise ValueError("The annual-cycle marker size must be finite.")
    if target_marker is not None and target_marker_size <= 0:
        raise ValueError("The annual-cycle marker size must be greater than zero.")
    try:
        to_rgba(target_marker_facecolor)
    except (TypeError, ValueError) as exc:
        raise ValueError("The annual-cycle marker face color must be valid.") from exc
    fig, ax = plt.subplots(figsize=(11.0, 6.5), constrained_layout=True)
    target_years = sorted(result.target)
    target_colors = (
        (target_color,)
        if len(target_years) <= 1
        else distinct_publication_colors(len(target_years))
    )
    for left, right in _target_month_spans(result, highlighted_months):
        ax.axvspan(left, right, color=month_shading_color, alpha=0.25, zorder=0)
    for year, values in sorted(result.other.items()):
        ax.plot(result.x, values, color="#b8b8b8", linewidth=0.8, alpha=0.45, zorder=1)
    for color, year in zip(target_colors, target_years):
        values = result.target[year]
        ax.plot(
            result.x,
            values,
            color=color,
            linewidth=target_linewidth,
            label=str(year),
            marker=target_marker,
            markerfacecolor=target_marker_facecolor if target_marker is not None else None,
            markersize=target_marker_size,
            markevery=1,
            zorder=3,
        )
    ax.plot(result.x, result.climatology, color="black", linewidth=2.4, label="Climatology", zorder=4)
    month_tick_positions, month_tick_labels, day_tick_positions, day_tick_labels = _annual_cycle_ticks(result)
    is_day_based = result.resolution == "daily" or result.resolution.endswith("-day") or result.resolution == "sub-daily"
    if result.resolution == "monthly":
        ax.set_xticks(month_tick_positions)
        ax.set_xticklabels(month_tick_labels, rotation=45, ha="right")
        ax.set_xlabel("Month (two-year continuity window)")
    elif is_day_based:
        ax.set_xticks(month_tick_positions)
        ax.set_xticklabels(month_tick_labels, rotation=45, ha="right")
        ax.tick_params(axis="x", pad=8)
        if result.resolution == "daily":
            secondary = ax.secondary_xaxis("bottom")
            secondary.spines["bottom"].set_position(("outward", 38))
            secondary.set_xticks(day_tick_positions)
            secondary.set_xticklabels(day_tick_labels)
            secondary.set_xlabel("Day of year")
        else:
            ax.set_xlabel("Month (two-year continuity window)")
    else:
        ax.set_xlabel("Time")
    ylabel = unit_label or "Value"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n{result.resolution} data | {weighting} spatial mean | previous-to-target-year continuity"
        + (f" | {running_mean_window}-sample running mean" if running_mean_window > 1 else ""),
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, ncol=min(4, len(labels)), loc="upper left")
    return fig


def _nice_scale_distance(max_distance_m: float, unit: str) -> float:
    """Choose a readable 1/2/5 scale-bar length below the requested width."""
    factor = 1000.0 if unit == "kilometer" else 1.0
    available = max_distance_m / factor
    if available <= 0:
        raise ValueError("The map is too small to calculate a scale bar.")
    exponent = np.floor(np.log10(available))
    for multiplier in (5.0, 2.0, 1.0):
        candidate = multiplier * 10.0 ** exponent
        if candidate <= available:
            return float(candidate)
    return float(10.0 ** (exponent - 1.0))


def _add_geographic_annotations(ax, field, projection, unit: str, show_scale: bool, show_north: bool) -> None:
    """Add a geodesic scale bar and geographic north arrow to a Cartopy map."""
    if not show_scale and not show_north:
        return
    lon_min, lon_max = float(field.lon.min()), float(field.lon.max())
    lat_min, lat_max = float(field.lat.min()), float(field.lat.max())
    center_lat = 0.5 * (lat_min + lat_max)
    center_lon = 0.5 * (lon_min + lon_max)
    try:
        from pyproj import Geod
    except ImportError as exc:
        raise ValueError("Geodesic support is required for map annotations.") from exc
    geodesic = Geod(ellps="WGS84")
    if show_scale:
        # Draw the bar in axes coordinates. A geographic line is curved by
        # global projections (especially Equal Earth/Mollweide), which makes
        # the scale bar look distorted even though its geodesic length is
        # correct.
        bar_width_axes = 0.18
        anchor_lon = lon_min + 0.08 * (lon_max - lon_min)
        anchor_lat = lat_min + 0.08 * (lat_max - lat_min)
        geographic_width = bar_width_axes * max(lon_max - lon_min, 1.0)
        _, _, width_m = geodesic.inv(
            anchor_lon,
            anchor_lat,
            anchor_lon + geographic_width,
            anchor_lat,
        )
        distance = _nice_scale_distance(width_m, unit)
        ax.plot(
            [0.08, 0.08 + bar_width_axes],
            [0.08, 0.08],
            transform=ax.transAxes,
            color="black",
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=10,
        )
        ax.plot(
            [0.08, 0.08, 0.08 + bar_width_axes, 0.08 + bar_width_axes],
            [0.065, 0.095, 0.065, 0.095],
            transform=ax.transAxes,
            color="black",
            linewidth=1.2,
            zorder=10,
        )
        label = f"{distance:g} {'km' if unit == 'kilometer' else 'm'}"
        ax.text(
            0.08,
            0.10,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
            zorder=11,
        )
    if show_north:
        arrow_lon = lon_min + 0.94 * (lon_max - lon_min)
        arrow_lat = lat_min + 0.10 * (lat_max - lat_min)
        arrow_length = max(1.0, 0.10 * (lat_max - lat_min))
        north_lon, north_lat, _ = geodesic.fwd(
            arrow_lon, arrow_lat, 0.0, arrow_length * 111_000.0
        )
        ax.annotate(
            "N",
            xy=(north_lon, north_lat),
            xytext=(arrow_lon, arrow_lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.4},
            zorder=11,
        )

_HIGHLIGHT_EDGE_SAMPLES = 64


def _densify_geographic_path(vertices: np.ndarray, samples_per_edge: int) -> np.ndarray:
    """Interpolate polygon edges before projection so global boundaries curve."""
    points = []
    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        edge = np.column_stack(
            (
                np.linspace(start[0], end[0], samples_per_edge, endpoint=False),
                np.linspace(start[1], end[1], samples_per_edge, endpoint=False),
            )
        )
        points.append(edge)
    return np.vstack(points + [vertices[:1]])


@dataclass(frozen=True)
class PlotOptions:
    projection: str = "Automatic (recommended)"
    cmap: str = "cmocean:balance"
    contour_count: int = 20
    vmin: float | None = None
    vmax: float | None = None
    vector_mode: str = "quiver"
    vector_stride: int = 10
    vector_scale: int = 300
    scale_mode: str = "Robust percentile"
    contour_min: float | None = None
    contour_max: float | None = None
    contour_interval: float | None = None
    contour_levels: tuple[float, ...] | None = None
    contour_color: str = "0.25"
    contour_linestyle: str = "-"
    contour_linewidth: float = 0.35
    vector_reference: float | None = None
    center_colorbar_on_zero: bool = False
    colorbar_tick_interval: float | None = None
    colorbar_tick_format: str = "Automatic"
    color_scale: str = "Linear"
    logarithmic_base: float = 10.0
    show_quiver: bool | None = None
    show_streamplot: bool | None = None
    vector_significant: xr.DataArray | None = None
    trend_significant: xr.DataArray | None = None
    trend_marker_size: float = 12.0
    cmap_low_color: str | None = None
    cmap_high_color: str | None = None
    cmap_image: bytes | None = None
    figure_annotation: str | None = None
    fill_style: str = "Contourf"
    show_contours: bool = True
    show_contour_labels: bool = True
    contour_label_size: float = 12.0
    ensemble_consensus: xr.DataArray | None = None
    ensemble_threshold: float = 0.8
    highlight_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    highlight_vertices: tuple[tuple[float, float], ...] | None = None
    plotted_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    highlight_color: str = "#111111"
    highlight_linewidth: float = 1.5
    ensemble_agreement: xr.DataArray | None = None
    agreement_threshold: float = 0.80
    vector_label: str | None = None
    show_scale_bar: bool = False
    scale_bar_unit: str = "kilometer"
    show_north_arrow: bool = False
    land_color: str = "#e8e5dc"
    coastline_color: str = "#202020"


def _angular_width(bounds: tuple[float, float], period: float = 360.0) -> float:
    start, end = (float(value) for value in bounds)
    raw_width = end - start
    if abs(raw_width) >= period:
        return period
    return raw_width % period


def highlight_bounds_within(
    outer: tuple[tuple[float, float], tuple[float, float]],
    inner: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    """Return whether an inner geographic box is contained by a smaller outer box."""
    (outer_west, outer_east), (outer_south, outer_north) = outer
    (inner_west, inner_east), (inner_south, inner_north) = inner
    bounds = np.asarray(
        [outer_west, outer_east, outer_south, outer_north, inner_west, inner_east, inner_south, inner_north],
        dtype=float,
    )
    if not np.all(np.isfinite(bounds)):
        return False
    if float(outer_south) >= float(outer_north) or float(inner_south) >= float(inner_north):
        return False
    outer_width = _angular_width((outer_west, outer_east))
    inner_width = _angular_width((inner_west, inner_east))
    if not 0.0 < inner_width < outer_width:
        return False
    # Longitude is cyclic. Try equivalent representations of the inner
    # interval so a valid region crossing +/-180 degrees is not rejected just
    # because its endpoints use a different 360-degree branch.
    longitude_inside = outer_width >= 360.0 - 1e-9 or any(
        float(outer_west) <= float(inner_west) + 360.0 * shift
        and float(inner_east) + 360.0 * shift <= float(outer_east)
        for shift in range(-2, 3)
    )
    latitude_inside = (
        float(outer_south) <= float(inner_south)
        and float(inner_north) <= float(outer_north)
    )
    return longitude_inside and latitude_inside


def _cmap(name: str, low_color: str | None = None, high_color: str | None = None, image_bytes: bytes | None = None):
    if image_bytes:
        image = np.asarray(plt.imread(io.BytesIO(image_bytes)))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        rgb = image[..., :3]
        if np.nanmax(rgb) > 1.0:
            rgb = rgb / 255.0
        colors = np.nanmean(rgb, axis=0 if rgb.shape[0] <= rgb.shape[1] else 1)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("The uploaded colormap image must contain a readable color strip.")
        return LinearSegmentedColormap.from_list("uploaded_colormap", colors)
    if name.startswith("cmocean:"):
        try:
            import cmocean
            colormap = getattr(cmocean.cm, name.split(":", 1)[1])
        except (ImportError, AttributeError):
            colormap = plt.get_cmap("viridis")
    else:
        colormap = plt.get_cmap(name)
    if low_color is None and high_color is None:
        return colormap
    positions = np.linspace(0.0, 1.0, 256)
    colors = colormap(positions)
    if low_color is not None:
        colors[0] = to_rgba(low_color)
    if high_color is not None:
        colors[-1] = to_rgba(high_color)
    return LinearSegmentedColormap.from_list(f"custom_{name}", colors)
def _discrete_cmap_norm(levels, cmap, norm):
    """Let the continuous color scale be cut into discrete color blocks corresponding to the levels, with the same color selection method as contourf."""
    levels = np.asarray(levels, dtype=float)
    mids = 0.5 * (levels[:-1] + levels[1:])   # contourf also uses the midpoint of each interval to determine the color
    if norm is None:
        norm = Normalize(vmin=levels[0], vmax=levels[-1])
    d_cmap = ListedColormap(cmap(norm(mids)))
    d_cmap.set_under(cmap(-1))                # Integer index -1 / N will get the under / over color of cmap
    d_cmap.set_over(cmap(cmap.N))
    return d_cmap, BoundaryNorm(levels, d_cmap.N)
def _fill(ax, x, y, data, style, levels, cmap, norm, **kwargs):
    """Depending on style, choose contourf or pcolormesh(連續 / 離散)。"""
    values = np.ma.masked_invalid(np.asarray(data.values if hasattr(data, "values") else data, dtype=float))
    if style.startswith("Pcolormesh"):
        if style.endswith("(discrete)"):
            cmap, norm = _discrete_cmap_norm(levels, cmap, norm)
        elif norm is None:
            norm = Normalize(vmin=levels[0], vmax=levels[-1])
        return ax.pcolormesh(x, y, values, cmap=cmap, norm=norm, shading="auto", **kwargs)
    return ax.contourf(x, y, values, levels=levels, cmap=cmap, norm=norm, extend="both", **kwargs)

def _color_limits(values: np.ndarray, options: PlotOptions) -> tuple[float, float]:
    """Resolve finite color limits according to the selected scale policy."""
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            "The selected field contains no finite values after the requested "
            "time, level, region, and missing-value selections."
        )
    if options.color_scale == "Logarithmic":
        finite = finite[finite > 0]
        if finite.size == 0:
            raise ValueError(
                "A logarithmic color scale requires at least one positive "
                "value; use Linear for zero or negative anomalies."
            )
    if options.scale_mode == "Manual strict limits":
        if options.vmin is None or options.vmax is None:
            raise ValueError("Manual color limits must be specified.")
        vmin, vmax = options.vmin, options.vmax
    elif options.scale_mode == "Data min/max":
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))
    if options.color_scale == "Logarithmic" and (vmin <= 0 or vmax <= 0):
        raise ValueError("Logarithmic color scales require strictly positive limits.")
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError("The selected field does not provide a valid color scale.")
    return vmin, vmax


def _colorbar_mappable(
    values: np.ndarray,
    options: PlotOptions,
) -> tuple[float, float, object, object]:
    """Build the exact color mapping used by the map and its standalone bar."""
    lower, upper = _color_limits(values, options)
    if options.center_colorbar_on_zero and options.color_scale != "Logarithmic":
        limit = max(abs(lower), abs(upper))
        lower, upper = -limit, limit
    colormap = _cmap(options.cmap, options.cmap_low_color, options.cmap_high_color, options.cmap_image)
    norm = LogNorm(lower, upper) if options.color_scale == "Logarithmic" else (
        TwoSlopeNorm(vcenter=0.0, vmin=lower, vmax=upper)
        if options.center_colorbar_on_zero else Normalize(lower, upper)
    )
    if options.color_scale == "Logarithmic":
        levels = np.geomspace(lower, upper, options.contour_count)
    else:
        levels = np.linspace(lower, upper, options.contour_count)
    if options.fill_style == "Contourf" or options.fill_style.endswith("(discrete)"):
        colormap, norm = _discrete_cmap_norm(levels, colormap, norm)
    return lower, upper, colormap, norm


def _symmetric_colorbar_ticks(limit: float) -> np.ndarray:
    """Return symmetric, human-readable ticks for a zero-centered colorbar."""
    if not np.isfinite(limit) or limit <= 0:
        return np.array([0.0])
    raw_step = limit / 4.0
    exponent = np.floor(np.log10(raw_step))
    scale = 10.0**exponent
    normalized = raw_step / scale
    base_step = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 2.5 if normalized <= 2.5 else 5.0
    step = base_step * scale
    step = round(step, max(0, int(-exponent) + 1))
    tick_limit = np.ceil(limit / step) * step
    ticks = np.arange(-tick_limit, tick_limit + step * 0.5, step)
    return np.round(ticks, max(0, int(-np.floor(np.log10(step))) + 2))


def _apply_numeric_tick_format(axis, ticks: np.ndarray) -> None:
    """Format numeric ticks with integers or at most one decimal place."""
    if ticks.size == 0:
        return
    decimals = 1 if np.any(np.abs(ticks - np.round(ticks)) > 1e-8) else 0
    axis.set_major_formatter(FormatStrFormatter("%.1f" if decimals else "%.0f"))


def _decimal_places(values: np.ndarray, maximum: int = 6) -> int:
    """Choose the shortest stable decimal representation for numeric labels."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0
    for places in range(maximum + 1):
        rounded = np.round(finite, places)
        if np.allclose(finite, rounded, rtol=0.0, atol=10 ** (-(places + 1))):
            return places
    return maximum


def _clean_ticks(ticks: np.ndarray) -> np.ndarray:
    """Remove near-duplicate ticks and avoid the visually noisy -0.0 label."""
    if ticks.size == 0:
        return ticks
    finite = np.asarray(ticks, dtype=float)
    nonzero = np.abs(finite[np.isfinite(finite) & (finite != 0.0)])
    magnitude_precision = (
        int(max(0.0, -np.floor(np.log10(np.max(nonzero))) + 2.0))
        if nonzero.size
        else 0
    )
    precision = max(_decimal_places(ticks), magnitude_precision)
    precision = min(15, precision)
    cleaned = np.round(np.asarray(ticks, dtype=float), precision)
    cleaned[np.isclose(cleaned, 0.0)] = 0.0
    return np.unique(cleaned)


def _units_label(data: xr.DataArray) -> str:
    """Render the user-provided unit expression as Matplotlib math text."""
    units = str(data.attrs.get("units_latex", data.attrs.get("units", ""))).strip()
    if not units:
        return ""
    if units.startswith("$") and units.endswith("$"):
        units = units[1:-1].strip()
    if r"\mathrm" not in units and r"\text" not in units:
        units = rf"\mathrm{{{units}}}"
    return f"${units}$"


def _spatial_field(field: xr.DataArray) -> xr.DataArray:
    """Return a strictly two-dimensional latitude/longitude plotting field."""
    if not {"lat", "lon"}.issubset(field.dims):
        raise ValueError("Map data must contain latitude and longitude dimensions.")
    extra_dims = [dimension for dimension in field.dims if dimension not in {"lat", "lon"}]
    non_singleton = {
        dimension: field.sizes[dimension]
        for dimension in extra_dims
        if field.sizes[dimension] != 1
    }
    if non_singleton:
        dimensions = ", ".join(f"{name}={size}" for name, size in non_singleton.items())
        raise ValueError(
            "Map data must be reduced to a 2-D (lat, lon) field before plotting; "
            f"remaining dimensions: {dimensions}."
        )
    result = field.squeeze(drop=True).transpose("lat", "lon")
    if result.lat.ndim != 1 or result.lon.ndim != 1:
        raise ValueError(
            "Map plotting requires one-dimensional latitude and longitude coordinates."
        )
    return result


def _field_label(data: xr.DataArray) -> str:
    label = data.attrs.get("long_name", data.name or "value")
    units = _units_label(data)
    return f"{label} [{units}]" if units else str(label)


def _apply_scientific_tick_format(axis, ticks: np.ndarray) -> None:
    """Choose enough scientific mantissa digits to distinguish adjacent ticks."""
    finite_ticks = np.asarray(ticks, dtype=float)
    finite_ticks = finite_ticks[np.isfinite(finite_ticks)]
    if finite_ticks.size > 1:
        unique_ticks = np.unique(np.sort(finite_ticks))
        spacing = np.diff(unique_ticks)
        spacing = spacing[spacing > 0]
        scale = np.max(np.abs(unique_ticks))
        if spacing.size and scale > 0:
            precision = int(np.ceil(-np.log10(np.min(spacing) / scale)))
        else:
            precision = 1
    else:
        precision = 1
    precision = int(np.clip(precision, 1, 10))
    axis.set_major_formatter(FormatStrFormatter(f"%.{precision}e"))


def _configure_colorbar(colorbar, lower: float, upper: float, options: PlotOptions, axis: str) -> None:
    """Apply the map colorbar's tick policy to any Matplotlib colorbar."""
    tick_axis = colorbar.ax.xaxis if axis == "x" else colorbar.ax.yaxis
    if options.colorbar_tick_interval is not None:
        if not np.isfinite(options.colorbar_tick_interval) or options.colorbar_tick_interval <= 0:
            raise ValueError("Colorbar tick interval must be finite and greater than zero.")
        if options.color_scale == "Logarithmic":
            colorbar.locator = LogLocator(base=options.logarithmic_base)
            colorbar.formatter = LogFormatter(base=options.logarithmic_base)
            colorbar.update_ticks()
        else:
            interval = options.colorbar_tick_interval
            first_tick = np.ceil(lower / interval) * interval
            last_tick = np.floor(upper / interval) * interval
            ticks = np.arange(first_tick, last_tick + interval * 0.5, interval)
            if options.center_colorbar_on_zero and not np.any(np.isclose(ticks, 0.0)):
                ticks = np.sort(np.append(ticks, 0.0))
            ticks = _clean_ticks(ticks)
            colorbar.set_ticks(ticks)
            if options.colorbar_tick_format == "Automatic":
                _apply_numeric_tick_format(tick_axis, ticks)
    elif options.center_colorbar_on_zero and options.color_scale != "Logarithmic":
        ticks = _symmetric_colorbar_ticks(max(abs(lower), abs(upper)))
        colorbar.set_ticks(ticks)
        _apply_numeric_tick_format(tick_axis, ticks)
    format_strings = {
        "Integer": "%.0f",
        "One decimal": "%.1f",
        "Two decimals": "%.2f",
        "Scientific": "%.1e",
    }
    if options.colorbar_tick_format in format_strings:
        if options.colorbar_tick_format == "Scientific":
            _apply_scientific_tick_format(tick_axis, colorbar.get_ticks())
        else:
            tick_axis.set_major_formatter(FormatStrFormatter(format_strings[options.colorbar_tick_format]))


def draw_colorbar(
    field: xr.DataArray,
    options: PlotOptions = PlotOptions(),
    orientation: str = "horizontal",
    tick_rotation: float = 0.0,
    label: str | None = None,
    label_location: str = "auto",
    label_rotation: float | None = None,
    label_size: float = 12.0,
    label_font: str = "DejaVu Sans",
    label_color: str = "#000000",
) -> plt.Figure:
    """Render the map's color scale alone on a transparent figure."""
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("Colorbar orientation must be horizontal or vertical.")
    if not np.isfinite(tick_rotation):
        raise ValueError("Colorbar tick rotation must be finite.")
    if not np.isfinite(label_size) or label_size <= 0:
        raise ValueError("Colorbar label size must be finite and greater than zero.")
    values = np.asarray(field.values, dtype=float)
    lower, upper, colormap, norm = _colorbar_mappable(values, options)
    figure = plt.figure(figsize=(8.0, 1.35) if orientation == "horizontal" else (1.7, 7.0))
    figure.patch.set_alpha(0.0)
    axis_object = figure.add_axes((0.16, 0.35, 0.68, 0.28) if orientation == "horizontal" else (0.35, 0.08, 0.28, 0.84))
    mappable = ScalarMappable(norm=norm, cmap=colormap)
    colorbar = figure.colorbar(mappable, cax=axis_object, orientation=orientation, extend="both")
    _configure_colorbar(colorbar, lower, upper, options, "x" if orientation == "horizontal" else "y")
    colorbar.ax.tick_params(
        axis="x" if orientation == "horizontal" else "y",
        labelrotation=float(tick_rotation),
    )
    if label:
        if orientation == "horizontal":
            if label_location not in {"auto", "top", "bottom"}:
                raise ValueError("Horizontal colorbar labels must use auto, top, or bottom.")
        elif label_location not in {"auto", "left", "right"}:
            raise ValueError("Vertical colorbar labels must use auto, left, or right.")
        fontproperties = {"family": label_font, "size": float(label_size)}
        if orientation == "horizontal":
            colorbar.set_label(
                label,
                loc="center" if label_location == "auto" else label_location,
                rotation=0.0 if label_rotation is None else label_rotation,
                color=label_color,
                fontproperties=fontproperties,
            )
        else:
            colorbar.ax.set_ylabel(
                label,
                rotation=90.0 if label_rotation is None else label_rotation,
                color=label_color,
                fontproperties=fontproperties,
            )
            colorbar.ax.yaxis.set_label_position("left" if label_location == "auto" else label_location)
    return figure


def draw_map(
    field: xr.DataArray,
    u_field: xr.DataArray | None = None,
    v_field: xr.DataArray | None = None,
    options: PlotOptions = PlotOptions(),
    contour_field: xr.DataArray | None = None,
):
    """Render a scalar map and optionally overlay wind vectors."""
    field = _spatial_field(field)
    if contour_field is not None:
        contour_field = _spatial_field(contour_field)
    if u_field is not None:
        u_field = _spatial_field(u_field)
    if v_field is not None:
        v_field = _spatial_field(v_field)
    longitude_span = float(field.lon.max() - field.lon.min())
    central_longitude = 180.0 if float(field.lon.max()) > 180.0 else float(field.lon.mean())
    if options.projection == "Automatic (recommended)":
        projection = ccrs.EqualEarth(central_longitude=central_longitude) if longitude_span > 240 else ccrs.PlateCarree(central_longitude=central_longitude)
    elif options.projection == "Equal Earth (equal-area)":
        projection = ccrs.EqualEarth()
    elif options.projection == "Mollweide (equal-area)":
        projection = ccrs.Mollweide(central_longitude=central_longitude)
    elif options.projection == "Orthographic (hemisphere)":
        projection = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=float(field.lat.mean()))
    elif options.projection == "Azimuthal Equidistant (polar)":
        latitude = float(field.lat.mean())
        projection = ccrs.AzimuthalEquidistant(
            central_longitude=central_longitude,
            central_latitude=90.0 if latitude >= 0 else -90.0,
        )
    elif options.projection == "Stereographic (polar)":
        latitude = float(field.lat.mean())
        projection = ccrs.Stereographic(
            central_longitude=central_longitude,
            central_latitude=90.0 if latitude >= 0 else -90.0,
        )
    else:
        projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 7.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.16)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    try:
        to_rgba(options.land_color)
        to_rgba(options.coastline_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("Land and coastline colors must be valid Matplotlib colors.") from exc
    ax.add_feature(cfeature.LAND, facecolor=options.land_color, zorder=0)
    latitude_span = float(field.lat.max() - field.lat.min())
    map_span = max(longitude_span, latitude_span)
    coastline_resolution = "10m" if map_span <= 20 else "50m" if map_span <= 80 else "110m"
    ax.coastlines(resolution=coastline_resolution, color=options.coastline_color, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.7)
    contour_data = contour_field if contour_field is not None else field
    contour_values = np.asarray(contour_data.values, dtype=float)
    values = np.asarray(field.values, dtype=float)
    vmin, vmax = _color_limits(values, options)
    color_norm = None
    if options.color_scale == "Logarithmic":
        color_norm = LogNorm(vmin=vmin, vmax=vmax)
    elif options.center_colorbar_on_zero:
        color_limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -color_limit, color_limit
        color_norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    else:
        color_norm = Normalize(vmin=vmin, vmax=vmax)
    if options.contour_levels is not None:
        levels = np.asarray(options.contour_levels, dtype=float)
        if levels.size == 0 or not np.isfinite(levels).all() or np.unique(levels).size != levels.size:
            raise ValueError("Specific contour values must be finite and unique.")
        levels = np.sort(levels)
        line_levels = levels
    elif options.contour_interval is not None:
        if options.contour_min is None or options.contour_max is None or options.contour_interval <= 0 or options.contour_min >= options.contour_max:
            raise ValueError("Contour range requires minimum < maximum and a positive interval.")
        if options.color_scale == "Logarithmic":
            if options.contour_min <= 0 or options.contour_max <= 0:
                raise ValueError("Logarithmic contour ranges require strictly positive limits.")
            levels = np.geomspace(options.contour_min, options.contour_max, max(2, options.contour_count))
        else:
            levels = np.arange(options.contour_min, options.contour_max + options.contour_interval * 0.5, options.contour_interval)
            
            levels = np.round(levels, _decimal_places(np.asarray([options.contour_interval])))
        if levels.size < 2:
            raise ValueError("Contour range must contain at least two levels.")
        line_levels = levels
    else:
        line_levels = None
    fill_levels = np.geomspace(vmin, vmax, options.contour_count) if options.color_scale == "Logarithmic" else np.linspace(vmin, vmax, options.contour_count)
    if line_levels is None and options.show_contours:
        finite_contour = contour_values[np.isfinite(contour_values)]
        if finite_contour.size == 0:
            raise ValueError("The contour field contains no finite values.")
        contour_vmin = float(np.nanmin(finite_contour))
        contour_vmax = float(np.nanmax(finite_contour))
        if contour_vmin >= contour_vmax:
            # A constant contour field simply has no isolines. Keep the
            # rendering path valid so the pcolor field can still be shown.
            line_levels = fill_levels[::2]
        elif options.color_scale == "Logarithmic":
            if contour_vmin <= 0:
                raise ValueError("Logarithmic contour data requires strictly positive values.")
            line_levels = np.geomspace(contour_vmin, contour_vmax, options.contour_count)[::2]
        else:
            line_levels = np.linspace(contour_vmin, contour_vmax, options.contour_count)[::2]
    colormap = _cmap(options.cmap, options.cmap_low_color, options.cmap_high_color, options.cmap_image)
    if options.show_contours:
        if not np.isfinite(options.contour_linewidth) or options.contour_linewidth <= 0:
            raise ValueError("Contour line width must be finite and greater than zero.")
        try:
            to_rgba(options.contour_color)
        except (TypeError, ValueError) as exc:
            raise ValueError("Contour color must be a valid Matplotlib color.") from exc
        contours = ax.contour(
            contour_data.lon,
            contour_data.lat,
            np.ma.masked_invalid(np.asarray(contour_data.values, dtype=float)),
            levels=line_levels,
            colors=options.contour_color,
            linestyles=options.contour_linestyle,
            linewidths=options.contour_linewidth,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
        )
        if options.show_contour_labels:
            decimals = _decimal_places(line_levels)
            ax.clabel(contours, inline=True, inline_spacing=2, fontsize=options.contour_label_size, fmt=f"%.{decimals}f")
    mesh = _fill(ax, field.lon, field.lat, field, options.fill_style,
                 fill_levels, colormap, color_norm, transform=ccrs.PlateCarree())
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, extend="both")
    cb.set_label(_field_label(field))
    _configure_colorbar(cb, vmin, vmax, options, "x")
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.35, color="0.35", alpha=0.5, linestyle=":")
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xformatter = LongitudeFormatter(zero_direction_label=True)
    gridlines.yformatter = LatitudeFormatter()
    gridlines.xlabel_style = {"size": 12}
    gridlines.ylabel_style = {"size": 12}
    _add_geographic_annotations(
        ax,
        field,
        projection,
        options.scale_bar_unit,
        options.show_scale_bar,
        options.show_north_arrow,
    )
    if options.trend_significant is not None:
        significance = options.trend_significant.transpose("lat", "lon")
        lon_grid, lat_grid = np.meshgrid(np.asarray(significance.lon), np.asarray(significance.lat))
        mask = np.asarray(significance.values) == True
        mask &= np.isfinite(lon_grid) & np.isfinite(lat_grid)
        ax.plot(
            lon_grid[mask],
            lat_grid[mask],
            linestyle="None",
            marker="o",
            markersize=max(0.35, min(float(options.trend_marker_size) * 0.08, 1.25)),
            markerfacecolor="#555555",
            markeredgecolor="none",
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=2,
            clip_on=True,
        )
    if options.ensemble_consensus is not None:
        consensus = options.ensemble_consensus
        if consensus.ndim != 2 or not {"lat", "lon"}.issubset(consensus.dims):
            raise ValueError("The ensemble consensus mask must be a two-dimensional latitude/longitude grid.")
        consensus = consensus.transpose("lat", "lon")
        mask = np.asarray(consensus.values, dtype=bool)
        lon_values = np.asarray(consensus.lon.values)
        lat_values = np.asarray(consensus.lat.values)
        if mask.shape != (lat_values.size, lon_values.size):
            raise ValueError("The ensemble consensus mask does not match the plotted grid.")
        disagreement = (~mask).astype(float)
        ax.contourf(
            lon_values,
            lat_values,
            disagreement,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.contourf(
            lon_values,
            lat_values,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["..."],
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    if options.ensemble_agreement is not None:
        if not 0.5 <= options.agreement_threshold <= 1.0:
            raise ValueError("The ensemble agreement threshold must be between 0.5 and 1.0.")
        agreement = options.ensemble_agreement
        if agreement.ndim != 2 or not {"lat", "lon"}.issubset(agreement.dims):
            raise ValueError("The ensemble agreement must be a two-dimensional latitude/longitude grid.")
        agreement = agreement.transpose("lat", "lon")
        agreement_values = np.asarray(agreement.values, dtype=float)
        longitude = np.asarray(agreement.lon.values)
        latitude = np.asarray(agreement.lat.values)
        robust = np.where(agreement_values >= options.agreement_threshold, 1.0, np.nan)
        uncertain = np.where(
            np.isfinite(agreement_values) & (agreement_values < options.agreement_threshold),
            0.0,
            np.nan,
        )
        if np.isfinite(uncertain).any():
            ax.contourf(
                longitude,
                latitude,
                uncertain,
                levels=[-0.5, 0.5],
                colors="none",
                hatches=["///"],
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
        if np.isfinite(robust).any():
            ax.contourf(
                longitude,
                latitude,
                robust,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["..."],
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
    if options.highlight_bounds is not None:
        field_west = float(field.lon.min())
        field_east = float(field.lon.max())
        field_center = 0.5 * (field_west + field_east)
        validation_bounds = options.plotted_bounds or (
            (field_west, field_east),
            (float(field.lat.min()), float(field.lat.max())),
        )

        def align_longitude(value: float) -> float:
            return float(value) + 360.0 * round((field_center - float(value)) / 360.0)

        highlight_west = align_longitude(options.highlight_bounds[0][0])
        highlight_east = align_longitude(options.highlight_bounds[0][1])
        if highlight_east <= highlight_west:
            highlight_east += 360.0
        plotted_highlight_bounds = (
            (highlight_west, highlight_east),
            options.highlight_bounds[1],
        )
        if options.highlight_vertices is None:
            if not highlight_bounds_within(
                validation_bounds,
                plotted_highlight_bounds,
            ):
                raise ValueError("The highlighted region must be contained within the plotted region and smaller than it.")
        else:
            (outer_west, outer_east), (outer_south, outer_north) = validation_bounds
            vertices = np.asarray(options.highlight_vertices, dtype=float)
            aligned = vertices[:, 0] + 360.0 * np.round(
                (0.5 * (outer_west + outer_east) - vertices[:, 0]) / 360.0
            )
            if (
                np.any(vertices[:, 1] < outer_south)
                or np.any(vertices[:, 1] > outer_north)
                or np.any(aligned < outer_west)
                or np.any(aligned > outer_east)
            ):
                raise ValueError("Every polygon vertex must be contained within the plotted region.")
        if not np.isfinite(options.highlight_linewidth) or options.highlight_linewidth <= 0:
            raise ValueError("Highlight line width must be finite and greater than zero.")
        try:
            to_rgba(options.highlight_color)
        except (TypeError, ValueError) as exc:
            raise ValueError("Highlight color must be a valid Matplotlib color code.") from exc
        if options.highlight_vertices is not None:
            vertices = np.asarray(options.highlight_vertices, dtype=float)
            center = 0.5 * (field_west + field_east)
            vertices[:, 0] += 360.0 * np.round((center - vertices[:, 0]) / 360.0)
            vertices[:, 0] = np.unwrap(np.deg2rad(vertices[:, 0])) * 180.0 / np.pi
            corners = _densify_geographic_path(vertices, _HIGHLIGHT_EDGE_SAMPLES)
        else:
            (west, east), (south, north) = plotted_highlight_bounds
            edge_count = _HIGHLIGHT_EDGE_SAMPLES
            edges = [
                np.column_stack((np.linspace(west, east, edge_count), np.full(edge_count, south))),
                np.column_stack((np.full(edge_count, east), np.linspace(south, north, edge_count))),
                np.column_stack((np.linspace(east, west, edge_count), np.full(edge_count, north))),
                np.column_stack((np.full(edge_count, west), np.linspace(north, south, edge_count))),
            ]
            corners = np.concatenate([edge[:-1] for edge in edges] + [edges[-1][-1:]])
        ax.plot(
            corners[:, 0],
            corners[:, 1],
            color=options.highlight_color,
            linewidth=options.highlight_linewidth,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    if u_field is not None and v_field is not None:
        u, v = xr.align(u_field, v_field, join="inner")
        stride = max(1, options.vector_stride)
        lon = np.asarray(u.lon.values)[::stride]
        lat = np.asarray(u.lat.values)[::stride]
        uu = np.asarray(u.values)[::stride, ::stride]
        vv = np.asarray(v.values)[::stride, ::stride]
        vector_mask = None
        if options.vector_significant is not None:
            vector_significance = options.vector_significant.transpose("lat", "lon")
            vector_mask = np.asarray(vector_significance.values, dtype=bool)[::stride, ::stride]
        show_quiver = options.show_quiver if options.show_quiver is not None else options.vector_mode == "quiver"
        show_streamplot = options.show_streamplot if options.show_streamplot is not None else options.vector_mode == "streamplot"
        if show_streamplot and lon.size >= 3 and lat.size >= 3:
            ax.streamplot(lon, lat, uu, vv, density=1.2, color="k", linewidth=0.5, transform=ccrs.PlateCarree())
        if show_quiver:
            quiver_kwargs = {
                "transform": ccrs.PlateCarree(),
                "scale": options.vector_scale,
                "width": 0.0018,
            }
            if vector_mask is None:
                quiver = ax.quiver(lon, lat, uu, vv, color="black", **quiver_kwargs)
            else:
                opaque_u = np.where(vector_mask, uu, np.nan)
                opaque_v = np.where(vector_mask, vv, np.nan)
                faint_u = np.where(~vector_mask, uu, np.nan)
                faint_v = np.where(~vector_mask, vv, np.nan)
                quiver = ax.quiver(lon, lat, opaque_u, opaque_v, color="black", alpha=1.0, **quiver_kwargs)
                ax.quiver(lon, lat, faint_u, faint_v, color="black", alpha=0.28, **quiver_kwargs)
            auto_magnitude = float(np.nanmedian(np.hypot(uu, vv)))
            magnitude = options.vector_reference if options.vector_reference is not None else auto_magnitude
            if np.isfinite(magnitude) and magnitude > 0:
                vector_units = _units_label(u)
                vector_label = options.vector_label or u.name or "U component"
                reference_label = f"{vector_label}: {magnitude:.1f} [{vector_units}]" if vector_units else f"{vector_label}: {magnitude:.1f}"
                reference_key = ax.quiverkey(
                    quiver,
                    0.04,
                    1.08,
                    magnitude,
                    reference_label,
                    labelpos="E",
                    coordinates="axes",
                    labelsep=0.08,
                    fontproperties={"size": 12},
                    color="black",
                )
                reference_key.text.set_bbox({"facecolor": "white", "edgecolor": "0.35", "alpha": 0.85, "pad": 2.5})
    ax.set_title(field.attrs.get("long_name", field.name or "Climate field"))
    if options.figure_annotation:
        fig.text(0.01, 0.975, options.figure_annotation, ha="left", va="top", fontsize=12, color="0.25", linespacing=1.15)
    return fig


def draw_hovmoller(
    field: xr.DataArray,
    axis: str,
    cmap: str = "cmocean:balance",
    color_scale: str = "Linear",
    vmin: float | None = None,
    vmax: float | None = None,
    tick_format: str = "Automatic",
    low_color: str | None = None,
    high_color: str | None = None,
    image_bytes: bytes | None = None,
    annotation: str | None = None,
    fill_style: str = "Contourf",
    flip_x: bool = False,
    flip_y: bool = False,
) -> plt.Figure:
    """Render a year-by-longitude or year-by-latitude Hovmoller diagram."""
    if field.ndim != 2 or field.dims[0] != "analysis_year" or field.dims[1] != axis:
        raise ValueError("Hovmoller data must have dimensions ('analysis_year', 'lat' or 'lon').")
    if flip_x:
        field = field.isel({axis: slice(None, None, -1)})
    if flip_y:
        field = field.isel(analysis_year=slice(None, None, -1))
    values = np.asarray(field.values, dtype=float)
    options = PlotOptions(color_scale=color_scale, vmin=vmin, vmax=vmax, scale_mode="Data min/max")
    lower, upper = _color_limits(values, options)
    norm = LogNorm(lower, upper) if color_scale == "Logarithmic" else None
    levels = np.geomspace(lower, upper, 20) if color_scale == "Logarithmic" else np.linspace(lower, upper, 20)
    figure, axis_object = plt.subplots(figsize=(12, 5))
    figure.subplots_adjust(left=0.08, right=0.92, top=0.84, bottom=0.14)
    mesh = _fill(axis_object, field[axis], field.analysis_year, field, fill_style, levels, _cmap(cmap, low_color, high_color, image_bytes), norm)
    colorbar = figure.colorbar(mesh, ax=axis_object, pad=0.02, extend="both")
    label = field.attrs.get("long_name", field.name or "value")
    units = _units_label(field)
    if units:
        label += f" [{units}]"
    colorbar.set_label(label)
    format_strings = {"Integer": "%.0f", "One decimal": "%.1f", "Two decimals": "%.2f"}
    if tick_format in format_strings:
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter(format_strings[tick_format]))
    elif tick_format == "Scientific":
        _apply_scientific_tick_format(colorbar.ax.yaxis, colorbar.get_ticks())
    axis_object.set_xlabel("Longitude (degrees)" if axis == "lon" else "Latitude (degrees)")
    axis_object.set_ylabel("Analysis year")
    axis_object.set_title("Hovmoller diagram")
    axis_object.fontsize = 12
    if annotation:
        figure.text(0.01, 0.975, annotation, ha="left", va="top", fontsize=8, color="0.25", linespacing=1.15)
    return figure


def draw_vertical_section(
    section: xr.DataArray,
    horizontal_axis: str,
    cmap: str = "cmocean:balance",
    color_scale: str = "Linear",
    view_elevation: float = 25.0,
    view_azimuth: float = -60.0,
    low_color: str | None = None,
    high_color: str | None = None,
    image_bytes: bytes | None = None,
    annotation: str | None = None,
) -> plt.Figure:
    """Render a depth-resolved section as a perspective scientific surface."""
    if section.ndim != 2 or "level" not in section.dims:
        raise ValueError("A vertical section must contain a level dimension and one horizontal dimension.")
    horizontal_axis = next(dimension for dimension in section.dims if dimension != "level")
    section = section.transpose("level", horizontal_axis)
    values = np.asarray(section.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The vertical section contains no finite values.")
    options = PlotOptions(color_scale=color_scale, scale_mode="Data min/max")
    lower, upper = _color_limits(values, options)
    norm = LogNorm(lower, upper) if color_scale == "Logarithmic" else Normalize(lower, upper)
    horizontal = np.asarray(section[horizontal_axis].values, dtype=float)
    depth = np.asarray(section.level.values, dtype=float)
    horizontal_grid, depth_grid = np.meshgrid(horizontal, depth)
    figure = plt.figure(figsize=(12, 7.8))
    figure.subplots_adjust(left=0.06, right=0.9, top=0.84, bottom=0.1)
    axis_object = figure.add_subplot(111, projection="3d")
    colormap = _cmap(cmap, low_color, high_color, image_bytes)
    surface = axis_object.plot_surface(
        horizontal_grid,
        depth_grid,
        np.ma.masked_invalid(values),
        facecolors=colormap(norm(np.nan_to_num(values, nan=lower))),
        rstride=1,
        cstride=1,
        linewidth=0.15,
        edgecolor=(1.0, 1.0, 1.0, 0.15),
        antialiased=True,
        shade=False,
    )
    surface.set_cmap(colormap)
    surface.set_norm(norm)
    colorbar = figure.colorbar(surface, ax=axis_object, pad=0.12, shrink=0.75)
    colorbar.set_label(_field_label(section))
    axis_object.set_xlabel("Longitude (degrees)" if horizontal_axis == "lon" else "Latitude (degrees)")
    axis_object.set_ylabel("Depth / level")
    axis_object.set_zlabel(section.attrs.get("units", "Value"))
    axis_object.invert_yaxis()
    axis_object.view_init(elev=view_elevation, azim=view_azimuth)
    axis_object.set_title("Perspective vertical section")
    axis_object.fontsize = 12
    if annotation:
        figure.text(0.01, 0.975, annotation, ha="left", va="top", fontsize=8, color="0.25", linespacing=1.15)
    return figure