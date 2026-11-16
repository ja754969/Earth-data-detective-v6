# Climate Grid Explorer

Climate Grid Explorer is a modular Streamlit application for inspecting gridded climate and ocean data stored in NetCDF files. It is intended for ERA5, NCEP, MERRA-2, satellite, and other CF-like datasets and provides a Climate Reanalyzer-style workflow without downloading a copy of a remote dataset to the local disk.

## Architecture

```text
data_view.py
  Streamlit controls, validation messages, and download buttons
src/climate_data.py
  Dataset discovery, lazy xarray loading, coordinate detection,
  time aggregation, climatology, anomaly, and spatial selection
src/climate_plot.py
  Cartopy projections, normalized shading, colorbars, vector overlays,
  trend significance, and Hovmoller diagrams
cmd-based validation
  Targeted regression checks for detrending, decade trends, and date-line selection
../common-files/function/shared_helpers/annual_cycle.py
  Shared polygon, temporal-resolution, running-mean, and annual-cycle analysis
../common-files/function/shared_helpers/publication_colors.py
  Shared colour-blind-friendly publication palette
```

The `src` directory is intentional: the data and plotting helpers are independent of Streamlit and can be imported from a notebook, a batch script, or a future Panel application.

## Installation

The supplied `requirements.txt` matches the packages already verified in the `OC` conda environment. From a `cmd` terminal:

```bat
conda activate OC
cd /d path\to\Earth-data-detective-v6
python -m pip install -r requirements.txt
```

Cartopy may require a binary wheel appropriate for the Python version. The current `OC` environment already imports Cartopy successfully, so no separate installation is needed there.

## Security and publication checklist

This repository is intended to contain no personal data, credentials, access
tokens, private endpoints, local data paths, or user-specific configuration.
Before creating a public commit, keep only `.env.example` and ensure the local
`.env`, netrc files, credential JSON files, downloaded datasets, caches, and
IDE settings remain untracked. The provided `.gitignore` excludes these files,
but ignore rules take effect only after Git has been initialized and before a
file has been added to the index.

Run the following review immediately before publishing:

```bat
git status --ignored
git diff --cached --check
git ls-files
```

If a secret has ever been committed, removing the file in a later commit is
not sufficient: revoke or rotate the secret, then remove it from Git history
using an approved repository-security procedure.

## Start the application

```bat
conda activate OC
cd /d path\to\Earth-data-detective-v6
streamlit run data_view.py
```

Streamlit prints a local browser URL. The application starts with no dataset loaded. Select a local NetCDF file, upload one, or enter a trusted HTTP/HTTPS NetCDF or OPeNDAP URL.
The project includes `.streamlit/config.toml`, which binds the development
server to the local IPv4 loopback interface (`127.0.0.1`) so the command above
does not require a repeated `--server.address` argument. On the same host,
open `http://127.0.0.1:8501`. For a remote host, use authenticated SSH
port forwarding or VS Code port forwarding; do not change the address to
`0.0.0.0` unless a separately secured reverse proxy, authentication layer, and
firewall policy are in place. The setting is portable because it contains no
machine-specific path or address.

For SSH forwarding, run this from the client machine:

```bat
ssh -N -L 8501:127.0.0.1:8501 user@remote-host
```

Then open `http://127.0.0.1:8501` locally. The Streamlit process remains
reachable only from the remote host itself, while the SSH connection provides
the authenticated transport.

The v5 project is independent of the original applications, but intentionally
uses only the shared helper path under `common-files\function\python`. When
the repositories are siblings, this path is detected automatically. On a
different layout, set `EARTH_DATA_COMMON_FUNCTION_ROOT` to the absolute
`common-files\function` directory before starting Streamlit. The old
shared-helper root must be configured with
`EARTH_DATA_COMMON_FUNCTION_ROOT`.
It does not import code from `common-files` or `Earth-data-detective` outside
that helper path. Running it does not start either original application. If
more than one Streamlit application is running, assign distinct ports
explicitly, for example `streamlit run data_view.py --server.port 8502`.

### Download behavior and resource use

The map, colorbar, annual-cycle, and ensemble NetCDF download controls use
Streamlit's `on_click="ignore"` behavior. Clicking a download control therefore
downloads the already prepared bytes without submitting the page again; the
selected data are not reloaded and the figures are not regenerated for each
download. Changing an analysis control still performs the normal Streamlit
rerun so that the displayed result stays synchronized with the user's
selection.

For lower resource use, keep **Update plot automatically** disabled while
adjusting several controls, then click **Update plot** once. Also restrict the
spatial range, choose a suitable field, and avoid enabling optional contour,
vector, Hovmoller, vertical-section, and annual-cycle figures unless they are
needed. The application retains the existing safe plot-size and export-size
limits rather than silently creating oversized in-memory figures or files.

## Highlight-region annual cycles

Enable **Highlight an additional spatial range** to generate a second figure.
The existing rectangle mode remains available; **Polygon** mode accepts
`longitude,latitude` vertices separated by semicolons or new lines. Polygon
boundaries are included in the grid mask, and longitude coordinates are
unwrapped around the first vertex so regions crossing the international date
line remain contiguous. The annual-cycle spatial mean can use either
cosine-of-latitude area weighting (the default) or equal grid-cell weights.

The shared `annual_cycle.py` helper detects sub-daily, daily, fixed multi-day
(for example `8-day`), monthly, yearly, and irregular sampling. Annual-cycle
slots are derived from the observed timestamps rather than being forced into
daily or monthly bins. Consequently, an 8-day product is plotted at its actual
sample positions and is never interpreted as an 8-day accumulation: values are
averaged or converted only according to the units and transformation selected
by the user. For fixed multi-day products, the **Annual-cycle month tick
position** control places each month label near either the first or the middle
calendar day of that month; the nearest observed sample is used for 8-, 11-,
15-day, and other fixed-interval products. Tick labels use the suffix `[-1]`
for the previous-year continuity segment and `[0]` for the target-year
segment. The optional **Highlight target-year months with shading** control
supports one or more selected months and a custom color; shading is applied
only to the target-year `[0]` segment. Yearly data are deliberately not
represented as a fabricated monthly cycle.
Target-year curves can optionally use a custom marker style, marker face
color, and marker size. Highlight polygons are densified before Cartopy
projection, so their edges follow the projected geographic curvature on
global maps instead of connecting only the four corner vertices.

When more than one target year is selected, each target-year curve receives a
different colour from the shared Okabe-Ito/Nature-compatible accessible
publication palette. A single target year continues to use the user-selected
highlight colour. The legend always retains the year labels; markers can be
enabled as an additional distinction for unusually long year lists. The
palette is deterministic, so the same selected years produce the same colours
across reruns.

The optional running mean is applied to the complete, chronologically sorted
polygon time series before it is split into years. December-to-January windows
therefore remain continuous. Large data gaps are treated as separate segments,
so a rolling window never silently bridges an unobserved interval. The figure
contains target-year curves, faint comparison-year curves, and a black
climatology curve. The selected centre month defines a continuous two-year display window. The
previous-year samples are placed before the target-year samples, so January is
always preceded by the previous December (and, for a January centre, all
previous-year months remain connected and highlighted on the left). Missing
previous-year observations are left as NaN and reported as an explicit warning;
they are never replaced with fabricated values.

The annual-cycle controls and figure preserve the main analysis source,
missing-value handling, level selection, detrending, unit conversion, and
ensemble-mean behavior. PNG and vector-friendly PDF downloads are provided at
300 DPI or better. The calculation is performed only after a valid highlight
polygon exists, which avoids loading a second large time series during normal
map-only use.

The annual-cycle source can be selected independently as the pcolor field,
contour field, quiver U component, quiver V component, quiver speed, or both
quiver components. Quiver speed is calculated as `sqrt(U^2 + V^2)`. Selecting
both quiver components renders one annual-cycle figure per component.

Map contours can be disabled, drawn without labels, drawn with labels, or
restricted to explicitly entered values. When contours are enabled, their
color, line style, and line width are configurable. The optional north arrow
and scale bar use WGS84 geodesic calculations on the displayed Cartopy map
domain. The scale-bar distance is selected from readable 1/2/5 values and its
label can use metres or kilometres; it is not a decorative fixed-length line.
The bar itself is drawn in axes coordinates so global projections do not bend
or skew it.

Grid-cell detrending fits each cell from its finite time samples (at least two
are required). Missing samples remain missing instead of causing a partially
valid high-latitude cell to be discarded completely. After the map color scale
is finalized, the sidebar can also export the same colormap, limits, and tick
policy as a transparent standalone PNG. Its orientation, tick-label rotation,
and optional LaTeX display-unit label (position, rotation, font, size, and
color) are independently configurable.

Polygon mode uses `folium` and `streamlit-folium` for browser-side drawing.
The drawn GeoJSON polygon is copied into the editable vertex text area, so
users can refine or reproduce a selection numerically before rendering.

## Resource and path behavior

NetCDF files remain lazy until a selected field is calculated. Plot-size and
download-size limits prevent accidental multi-gigabyte materialization. Annual
cycle processing reduces the selected field to one polygon-mean time series
before smoothing, which keeps the running mean and plotting memory bounded.
Annual-cycle source data are additionally rejected when the estimated
in-memory size of the polygon's bounding-box subset exceeds 768 MiB; the
full-grid size is not used for this check. The polygon reduction first
subsets latitude and longitude lazily, then applies the exact boundary mask.
Map and annual-cycle PNG/PDF exports are
created one format at a time and their temporary buffers are released
immediately after each Streamlit download control is created. Figures are
closed after rendering, including error paths.
Optional analyses are trigger-gated: the annual cycle is not built unless
**Highlight an additional spatial range** is enabled and a valid polygon is
available; Hovmoller and vertical-section arrays are not built unless their
respective display controls are enabled; vector fields are not calculated
unless a vector overlay or wind-stress-curl analysis is requested; and the contour
source is not transformed or reduced when contour drawing is disabled and it
is not selected as the annual-cycle source.
The dataset handle and coordinate/time metadata may still be opened to
populate the controls, but these operations do not materialize the data
variables.
The application depends on the target project, the shared Python helper
directory, and the selected local/uploaded/remote data source; it does not
start or import the unrelated applications in the neighboring repositories.

## ECCO-Darwin v5 CO2 flux streaming test

`src/ecco_darwin_flux.py` reads the MITgcm MDS files in the ECCO-Darwin v5
`CO2_flux` directory. It does not use `xarray.open_dataset()` for `.data`
files: each selected field reads and validates its adjacent `.meta` object,
then decodes the big-endian payload with `numpy.frombuffer()` in memory.
Remote URLs are indexed first; field binaries are never downloaded merely to
build the available time axis.

The reader accepts configuration from `.env` and command-line overrides. Copy
`.env.example` to `.env`, edit the grid URL and dates, and run from `cmd.exe`:

```bat
conda activate OC
cd /d path\to\Earth-data-detective-v6
python -m src.ecco_darwin_flux --min-lon 350 --max-lon 20
```

Every remote URL is restricted to `https://data.nas.nasa.gov` and every
connection uses a certificate-verifying `certifi` SSL context with TLS 1.2 or
newer. The longitude mask supports both `-180..180` and `0..360` inputs and
selects both sides of the international date line when the lower bound is
greater than the upper bound. The output contains an area-weighted flux
density (`mol C m-2 yr-1`) and integrated regional carbon flux (`Pg C yr-1`).
No `.data`, `.meta`, or output file is written by the reader.

The reported `Provider 'nasa' requires Earthdata credentials` message comes
from the existing generic OPeNDAP loader. The public `data.nas.nasa.gov`
directory is not an OPeNDAP dataset and should use the MDS reader above.
`earthaccess.login(persist=True)` is appropriate for Earthdata Login-protected
services, but it does not make `xarray.open_dataset()` understand MITgcm MDS
files and it does not automatically authenticate arbitrary `fsspec` reads.
Use it only when the specific endpoint requires Earthdata Login; never place
credentials in Python, JSON manifests, or `.env` files.

### ECCO-Darwin mode in Streamlit

The project-local configuration file is:

```text
.env
```

It is intentionally ignored by Git and currently contains only public NASA
URLs, dates, timestep metadata, and the spatial region. It must not contain
NASA usernames, passwords, tokens, or Earthdata credentials. The committed
template is `.env.example`; copy it to `.env` only when setting up another
working copy.

Start the application from `cmd.exe` in the `OC` environment:

```bat
conda activate OC
cd /d path\to\Earth-data-detective-v6
streamlit run data_view.py
```

In the sidebar, select **ECCO-Darwin CO2 flux**, then click **Read
ECCO-Darwin CO2 flux**. The application loads the project `.env`, validates
both URLs against the HTTPS `data.nas.nasa.gov` allowlist, reads the fixed LLC
longitude/latitude grid once, and creates a lazy `time × lat × lon` xarray
dataset. ECCO LLC270 has a nominal horizontal resolution of approximately
1/3 degree. The **ECCO display resolution** control offers 1 degree
(default), 0.5 degree, 0.25 degree, and the nominal 1/3-degree spacing. The
curvilinear LLC grid is represented on the chosen regular grid using a cached
nearest-neighbour index. Finer display grids retain more spatial detail but
increase the memory required for a selected monthly field and the cost of
multi-month operations. A selected month, season, anomaly, or trend
calculation downloads only the required monthly MDS fields. The ECCO
field therefore uses the same map, region, `-180..180`/`0..360` convention,
date-line wrapping, colour, Hovmoller, highlight-region, and export controls
as an Existing NetCDF source. `CO2_flux` is a sea-surface field with no
vertical coordinate, so a vertical section is correctly disabled rather than
inventing depth values. Selecting **Existing NetCDF / remote sources** leaves
the original viewer path unchanged.

NASA's public endpoint can occasionally return a transient HTTP 429 or 5xx
response. `fsspec` sometimes represents those failures as `FileNotFoundError`
during its preliminary metadata request. The ECCO reader inspects the entire
exception chain, limits the process to two concurrent public-server reads,
and retries transient statuses with exponential waits. After seven failed
attempts for one monthly field, that field is represented as an all-missing
slice and a warning is logged; no values are invented and the rest of a
multi-year operation can still complete. A true HTTP 404 remains a missing
object and is not retried.

## Execution flow

1. `data_view.py` first selects one of two source paths. The Existing NetCDF path scans the application directory and an optional local folder for a bounded list of `.nc`, `.nc4`, and `.cdf` files. The ECCO path validates `.env`, constructs the lazy regular-grid adapter in `src/ecco_darwin_flux.py`, and bypasses NetCDF discovery. An additional local folder, such as `D:\climate_data`, is only scanned in the Existing NetCDF path. Discovery inspects paths only; it neither loads all data nor uploads local files. If a folder contains separate time chunks, enable **Combine all NetCDF files in this folder**.
2. A selected local path, a tuple of local paths, an uploaded byte buffer, or a remote URL is passed to `load_dataset`. A tuple is opened with xarray `open_mfdataset(combine="by_coords")` using the `netcdf4` engine. `chunks="auto"` keeps the individual files lazy until a selected field is calculated. For ensemble processing, enter one folder per member in **Ensemble member folders** and enable **Calculate ensemble mean from folders**. Files within each folder are combined along time; the folders are then aligned and concatenated along `ensemble`, and the plotted target-period field averages that dimension.
3. Coordinate aliases such as `latitude`, `longitude`, `valid_time`, `lev`, and `isobaricInhPa` are detected and renamed internally to `lat`, `lon`, `time`, and `level`. One-dimensional latitude and longitude indexes are sorted; two-dimensional curvilinear coordinates are retained in their native dimension order and are selected with a geographic mask rather than one-dimensional `.sel()` indexing. Longitudes are normalized internally to `[-180, 180]`, while the sidebar can expose either `-180 to 180` or `0 to 360` to the user.
4. The sidebar selects a scalar variable, optional pressure level, target year, month or season, longitude convention, bounding box, projection, colormap, linear or logarithmic color scaling, and contour count. **Use a second folder for comparison overlays** opens a second local collection with its own filename pattern; the user can independently select that collection for the pcolor field, isoline contour field, or quiver/streamline vectors. Each field can use its own variable and grid, so the sources may have different spatial extents and resolutions without being forcibly aligned. Logarithmic scaling is intended for strictly positive quantities such as chlorophyll-a.
5. Time samples matching the selection are averaged. For anomaly mode, the selected target mean is reduced by a matching month or season climatology over the chosen baseline years.
6. Spatial selection normalizes the source longitude coordinate to the convention selected by the user. It handles ordinary ranges and ranges that wrap around the international date line by selecting both sides, concatenating them in geographic order, and unwrapping the second side by 360 degrees. For example, `350 to 20` in the `0 to 360` convention becomes `350, ..., 360, ..., 380`, even when the file stores longitudes as `-180 to 180`; a Pacific range `120 to -70` becomes `120, ..., 290`. This prevents contour polygons from being split at 180 degrees and makes `[0, 360]` convenient for Pacific-spanning regions.
7. Display units are normalized conservatively from unambiguous CF-style units: Kelvin to degrees Celsius, Pascal to hPa, and precipitation fluxes to mm/day. Temperature anomalies keep their difference value and only change the displayed unit label; they are never shifted by 273.15. Optional display units and the separate **Quiver unit label** accept LaTeX math text and are rendered upright inside the colorbar/reference labels' square brackets. The quiver label defaults to the U-component unit when left blank.
8. The contour control can use automatic levels or an explicit minimum, maximum, and interval. For example, minimum `500`, maximum `600`, and interval `20` produces `500, 520, ..., 600`.
9. Value conversion is explicit in the sidebar. Automatic CF conversion remains available, while arithmetic operations can add, subtract, multiply, or divide by a user-provided value. For geopotential `Z` in `m^2/s^2`, choose `Geopotential (m^2/s^2) / 9.8 -> height (m)`; the application evaluates `height = Z / 9.8` and labels the result in metres. The gravity constant is editable for reproducibility.
10. `draw_map` chooses Equal Earth automatically for global-scale views and Plate Carree for regional views. Equal Earth and Mollweide are available as equal-area projections; Plate Carree is retained for regional gridded inspection and Orthographic for hemisphere views. Coastline resolution increases from 110m to 50m and 10m as the selected spatial domain becomes smaller. The map adds land, coastlines, borders, labeled gridlines, filled contours, labeled isolines, a unit-aware colorbar, and optional quiver and streamplot overlays. Both vector styles can be enabled simultaneously. The quiver reference key uses the median plotted speed unless an explicit reference magnitude is supplied. Curvilinear grids use two-dimensional longitude/latitude arrays; streamlines are intentionally limited to rectilinear grids, while quiver vectors remain available. In the **Region** section, **Highlight an additional spatial range** enables a separate hollow boundary even when the main region is a preset; its longitude/latitude bounds, color, and line width are configurable. The highlighted region is validated against the user-requested plotted bounds (not merely the available coordinate endpoints), must be smaller and contained within that region, and may touch its boundary. The boundary is projected after being defined geographically, so its four corners and edges are not assumed to remain a rectangle in the projected map.
11. Color limits can be robust percentile limits, exact finite data minimum/maximum limits, or manual strict limits. Manual limits are rejected unless `minimum < maximum`, preventing silently misleading color scales.
12. Selecting **Linear trend** constructs one monthly or seasonal mean per analysis year and calculates a grid-cell ordinary least-squares slope. The slope time basis can be year, month, day, or decade. Decade slopes are annual-grid slopes rescaled to change per ten years, while month and day retain their actual elapsed-time spacing. Significance can use a two-sided Student's t test or a two-sided Kendall rank test at a user-selected confidence level. Significant cells are marked with adjustable gray dots. Selecting **Display Hovmoller diagram** adds a separate year-by-space diagram below the map, with independent longitude/latitude bounds and selectable spatial axis.
13. The figure is rendered in Streamlit and encoded in memory as a 300 DPI PNG and a publication PDF. No output image is written to disk.
14. When ensemble processing is enabled, the figure annotation reports the number of members. **Download all-time ensemble-mean NetCDF** writes an in-memory NetCDF containing the arithmetic mean across ensemble members for every available time and spatial coordinate, including all shared data variables. Members must have identical data-variable sets, coordinates, and time axes; mismatches are reported instead of silently regridding or concatenating incompatible data.
15. Resource safeguards prevent accidental workstation overload. Local discovery is bounded to 80 files, xarray opens datasets lazily with automatic chunks, and plotting rejects spatial fields larger than five million grid cells before converting them to NumPy arrays. The in-memory NetCDF export is refused when its estimated uncompressed size exceeds 512 MiB. These limits produce an explicit message rather than silently dropping data or allowing an uncontrolled allocation.
16. CMIP6 files with time, pressure, ensemble, and other non-spatial dimensions are reduced only after the requested period and level are selected. The final map contract is a two-dimensional spatial field. One-dimensional regular grids are transposed to `(lat, lon)` before rendering; two-dimensional curvilinear longitude/latitude coordinates are passed directly to Cartopy and are never sent to one-dimensional index helpers.
17. For an ensemble or multi-model mean, the annotation reports the participating model count. For absolute fields and anomalies, the map overlays `...` stippling where the selected fraction of valid models agrees with the MMM sign and `///` hatching elsewhere. The agreement threshold is selectable from 80% to 99%. Trend maps retain their selectable Student's t or Mann-Kendall confidence level.

### Seasonal linear-trend interpretation

For a **Year** or **Decade** trend, the application first creates one spatial
field per analysis year and then fits an independent ordinary least-squares
line at every grid cell. A seasonal field is the arithmetic mean of the time
samples selected for that season at that grid cell.

The **Season start month** defines the seasonal-year label. For the customary
DJF ordering `December, January, February`, with December selected as the
start month, `DJF 1998` means December 1998 plus January and February 1999.
Consequently, a requested 1998–2018 DJF trend requires data through February
2019. Selecting a different start month changes this attribution; for example,
an ordering beginning in January treats January, February, and December as
months of the same calendar-year label.

For monthly input, a complete DJF seasonal value is the mean of its three
monthly values. For daily or sub-monthly input, it is the mean of every
selected time sample in the season; it is not an equal-weight mean of three
separately calculated monthly means. This distinction matters when sampling
frequency differs among months.

The application does not require every selected month to be present before it
forms a seasonal mean. A season containing only available samples can therefore
contribute a partial-season value, especially at the beginning or end of a
dataset. Users should verify coverage through the month following the final
seasonal-year label and exclude incomplete seasons when scientifically
appropriate.

Missing values are handled independently at each grid cell. Each regression
uses that cell's finite seasonal values, so the number of fitted years can vary
spatially. Year and Decade trends require at least two valid analysis years;
Month and Day trends instead regress the selected individual observations and
require at least three time samples. When detrending is enabled, it is applied
to the complete selected source time series before seasonal aggregation and
trend estimation. Unit conversion is applied after the analysis result is
calculated.
18. The **Longitude range**, **Latitude range**, and additional highlighted-range controls use independent numeric inputs with decimal precision. This permits values such as `121.25` without dragging a coarse slider and preserves dateline-wrapping selections.

## CMIP6 troubleshooting

`Input DataArray is not 1-D` means that a one-dimensional coordinate/index operation received a two-dimensional array, or that a field still contains a non-spatial dimension when it reaches the map renderer. CMIP6 data can legitimately contain `time`, `plev`, `member_id`, bounds, or curvilinear `lat(y, x)`/`lon(y, x)` coordinates. This application handles those cases as follows:

1. Select the variable and pressure level, then select a month/season and target years.
2. The time aggregation and anomaly/trend calculation reduce time and unselected analysis dimensions.
3. Regular `lat`/`lon` coordinates are selected with one-dimensional slices. Curvilinear coordinates use a geographic mask and do not call one-dimensional indexing.
4. The renderer verifies that the final field is exactly two-dimensional and uses Cartopy's `PlateCarree` as the source-coordinate transform, even when the display projection is Equal Earth or Mollweide.

If a file still fails, the error is displayed explicitly in the Streamlit page. Check that the selected variable has latitude and longitude coordinates, that the selected period contains observations, and that the file's backend is available in the `OC` environment.

## Data sources

The application accepts data already available on the PC, including a folder explicitly entered in the sidebar, as well as uploaded files. For remote data, use an institution-provided HTTPS NetCDF endpoint or OPeNDAP endpoint from a trusted provider, for example:

- Copernicus Climate Data Store / ERA5: `https://cds.climate.copernicus.eu/` (download or authenticated API workflows are external to this viewer).
- NOAA PSL datasets and OPeNDAP services: `https://psl.noaa.gov/data/gridded/`.
- NASA Earthdata and GES DISC: `https://disc.gsfc.nasa.gov/` (Earthdata authentication may be required).
- ESGF/CMIP data nodes: use the node's published OPeNDAP or HTTPS URL.

Use **Secure registered remote dataset** for new cloud connections. Its entries
come from `datasets.yaml`, are checked against the `.env` allowlist before
network I/O, and retain a lazy spatial grid for the existing map, Hovmöller,
vertical-section, vector, and regional-analysis controls. Select a concise
time range and bbox in the sidebar before opening the provider; this prevents
an accidental global materialization.

Each `datasets.yaml` entry may declare non-secret `available_start` and
`available_end` ISO dates. The secure-remote sidebar uses these values as its
date-picker limits. If either bound is omitted, the interface uses 1900-01-01
for the lower bound and the current date for the upper bound, allowing new
providers to be registered without changing the UI.

The older **Configured remote dataset** option and `data_config.json` remain
available only for backward compatibility with existing user workflows. They
are not the enterprise security path and should be migrated to
`datasets.yaml`; do not add credentials, passwords, JSON key paths, SAS
tokens, or signed URLs to either registry. New protected NASA access must use
owner-only `.netrc`; Azure and Google access must use their passwordless
default identity chains described below.

## Multiple files in one folder

For data organized like this:

```text
D:\climate_data\temperature_2023_01.nc
D:\climate_data\temperature_2023_02.nc
D:\climate_data\temperature_2023_03.nc
```

enter `D:\climate_data` in **Additional local data folder**, enable **Combine all NetCDF files in this folder**, and then select the scalar variable. The files should describe compatible grids and contain non-overlapping or consistently matching `time` coordinates. The resulting virtual dataset can then be used with:

- a single month, such as January;
- seasons such as `JJA`;
- cross-year seasons such as `NDJ` (for example, `NDJ 2023` means November 2023, December 2023, and January 2024);
- absolute values; or
- anomalies against the selected baseline years.

Time coordinates should be CF-compatible NetCDF time values, not arbitrary text strings. If files contain overlapping time records, duplicated timestamps may need to be removed or reconciled before analysis.

## Supported data contract

Required coordinates are a one-dimensional latitude and longitude coordinate. Time is required for monthly, seasonal, or anomaly operations. A pressure coordinate is optional. Variable names are not hard-coded: the user selects scalar, U, and V variables from the dataset. Variable attributes `long_name` and `units` are used for titles and colorbar labels when present.

Malformed files and missing coordinates are reported in the UI rather than producing a traceback. A dataset with no time coordinate can still be inspected as a static field, but time controls cannot be used.

For a dataset containing only one year, the climatology baseline is fixed to that year and the year-range slider is intentionally replaced with a notice. Streamlit sliders require different minimum and maximum values, so this avoids an invalid control while preserving the ability to plot the selected period.

## Runtime paths and resource controls

At runtime the application directly uses:

- `data_view.py` as the Streamlit entry point;
- `src/climate_data.py` and
  `src\climate_plot.py` for data and rendering logic;
- a sibling `../common-files/function/shared_helpers` directory for shared annual-cycle,
  detrending, ensemble, regridding, and publication-colour helpers;
- the selected local/uploaded/remote NetCDF source; and
- the `OC` conda environment and packages listed in `requirements.txt`.

The application does not import or launch neighboring applications. It writes
no map or annual-cycle image to disk; downloads are built in memory.

- xarray opens datasets lazily and Streamlit caches the opened result for the current source.
- The selected longitude/latitude region is reduced before detrending. The
  complete available time series for that source is then detrended, preserving
  the original workflow. Only after detrending do year, month, and season
  selections feed linear-trend or period calculations. This ordering is kept
  per source, so different remote/local time lengths and resolutions do not
  get forced into the primary source's time axis.
- The vector, contour, annual-cycle, and Hovmoller paths use the same
  preselection rule; they do not bypass the spatial limit by detrending the
  original dataset first.
- Local file discovery is bounded to 80 paths and does not read NetCDF contents.
- Enter a specific data folder instead of scanning an entire drive; this keeps Windows file discovery responsive and avoids unrelated directories.
- Vector fields are subsampled using the stride control before quiver or streamplot rendering. The selected stride is recorded in the figure annotation as the post-processing sampling interval. When quiver is enabled, its reference key is placed above the map with a separate margin and is labeled with the selected U-variable name rather than a generic "Reference:" prefix.
- The figure is closed after creating the download buffers.
- Keep contour levels and vector density moderate for high-resolution grids. For very large remote files, prefer an OPeNDAP endpoint that supports server-side slicing or a pre-gridded product.
- Hovmoller diagrams provide independent x- and y-axis reversal controls.
  Optional NetCDF downloads contain the exact Hovmoller field used for the
  figure. Optional annual-cycle NetCDF downloads contain the spatially averaged
  time series used to construct the corresponding annual-cycle figure.
- If the same dataset is used repeatedly, create a coarser analysis product with xarray or Zarr outside the UI; do not load the full-resolution global field into every Streamlit rerun.

## Runtime workflow

1. Streamlit starts `data_view.py`, configures the page, and builds the sidebar.
2. The source selector discovers local NetCDF files, accepts an uploaded file,
   or passes a trusted URL to `load_dataset`. Dataset handles are cached by
   source and upload payload.
3. Coordinate names, available years, temporal resolution, variables, levels,
   and comparison sources are inspected before any selected field is loaded.
4. The selected scalar field is masked using CF fill metadata and optional
   user thresholds. The optional vector fields use the same masking rules.
5. The analysis pipeline applies optional detrending, then performs period
   selection, anomaly calculation, or linear-trend estimation. Detrending is
   grid-cell based: each cell uses its finite samples, requires at least two
   distinct time values, and preserves missing samples as missing.
   `normalize` removes the fitted change relative to the selected reference
   year; `anomaly` removes the complete fitted line and therefore produces
   residuals around zero. Missing land cells remain masked in both modes and
   are never converted into plotted data values.
6. Unit conversion and the optional LaTeX display-unit label are applied after
   the analysis result is produced. Longitude selection unwraps cyclic
   coordinates when a requested region crosses the international date line.
7. `draw_map` chooses a Cartopy projection, computes limits and normalization,
   renders the scalar field, optional contours/vectors/significance overlays,
   and applies the colorbar tick policy. Invalid cells are passed to
   Matplotlib as an explicit masked array, so missing land data cannot be
   interpreted as an anomaly. **Land color** and **Coastline color** are
   independent user controls. Geographic scale-bar distances use
   WGS84 geodesics; the visible bar is drawn in axes coordinates so global
   projections cannot bend it.
8. Optional annual-cycle, Hovmoller, and vertical-section calculations run only
   when their controls are enabled. Each optional figure is closed after its
   download controls are created.
9. For an annual cycle, the selected highlight polygon is converted to a
   boundary-inclusive grid mask. The spatial mean is computed with either
   cosine-of-latitude or equal-cell weights, then the continuous running mean
   (if enabled) is applied before year grouping. Previous-year and target-year
   samples are concatenated into one continuity window. Each selected target
   year is plotted with a unique accessible publication colour; comparison
   years are intentionally faint and the climatology is black.
   The annual-cycle source follows the same missing-value, detrend, unit
   conversion, and spatial-weighting pipeline as the selected analysis source.
   Therefore enabling detrending also detrends the annual-cycle data before
   the polygon mean and before the running mean.
10. The map is exported as PNG and PDF. If enabled, `draw_colorbar` reuses the
   finalized map color limits, normalization, colormap, fill style, contour
   count, and tick settings to create a separate transparent PNG. `Contourf`
   and `Pcolormesh (discrete)` use the same `BoundaryNorm` interval mapping as
   the plotted color blocks; continuous pcolormesh and logarithmic scales keep
   their continuous mapping. Only the standalone bar's orientation, tick
   rotation, and display-unit label presentation are independently configured.

### Data-driven manual ranges and numeric precision

The scalar and contour controls compute finite numeric extrema from the
selected variables before the plotting controls are shown. The values are
computed through xarray reductions, cached by Streamlit, and do not require
materializing the complete array in Python. The observed scalar and contour
ranges are displayed beside the controls and are used as the initial values
for **Manual strict limits**, **Contour minimum**, and **Contour maximum**.
Users can still enter limits outside the observed range. The default custom
tick and contour intervals are one tenth of the corresponding observed
range, with a small positive fallback for constant fields.
After analysis, detrending, unit conversion, and spatial selection, the
application reports the final pcolor and contour extrema again. Automatic
color and contour limits are calculated from these final fields rather than
from the raw source range, so multiplication, division, and detrending cannot
leave a stale displayed scale.

Custom values are passed through unchanged until Matplotlib creates the
labels. Scientific colorbar labels no longer use a fixed one-decimal format:
the renderer examines the actual finite tick spacing and selects enough
scientific mantissa digits to distinguish adjacent ticks (up to ten decimal
places). This is important for small intervals such as `1e-8`, where a
fixed `%.1e` formatter can hide meaningful differences.

### Longitude conventions and the international date line

The **Longitude convention** control lets users enter either `-180 to 180`
or `0 to 360` coordinates. Source coordinates are normalized to the selected
cyclic convention for the requested operation. If the west bound is
numerically greater than the east bound, the selection is treated as a
wrapped interval: the western segment is concatenated with the eastern
segment, and the second segment is unwrapped by adding 360 degrees. This
keeps the two sides of the international date line adjacent for maps,
contours, vectors, Hovmoller sections, and highlighted regions. A global
`0 to 360` selection is restored to the conventional `0..360` labels after
selection.

## Tests

Run focused validation from `cmd`, as required for this workspace:

```bat
conda activate OC
cd /d path\to\Earth-data-detective-v6
python -m py_compile data_view.py src\climate_data.py src\climate_plot.py
python -m pytest -q
```

The checked pipeline covers Python syntax/imports, ECCO MDS metadata parsing,
rate-limit retry, deferred monthly reads, grid construction, and cyclic
longitude selection.

## Related files and cleanup policy

The application directly uses `data_view.py`, `src\climate_data.py`,
`src\climate_plot.py`, the original shared helpers under
the configured shared-helper directory, `requirements.txt`, and
`tests\test_climate_data.py`. It does not import the MATLAB files or any
other project paths. `folium` is listed explicitly because `data_view.py`
imports it during Streamlit startup; it is not assumed to be installed only
as a transitive dependency of `streamlit-folium`.

The runtime path chain is therefore:

```text
data_view.py
  -> src\climate_data.py and src\climate_plot.py
  -> configured shared_helpers directory
  -> the user-selected local, uploaded, or remote NetCDF source
```

The shared helper root can be overridden with
`EARTH_DATA_COMMON_FUNCTION_ROOT`; otherwise the sibling `common-files`
folder is preferred. New helpers should be placed in the configured
`shared_helpers` directory only when they are genuinely
generic and independently reusable. `publication_colors.py` is shared because
its deterministic accessible palette is independent of this application;
plot-specific rendering remains in `src\climate_plot.py`.

No obsolete project script or documentation file was found among the source
files: `data_view.py`, the two `src` modules, and this
README are all active runtime or maintenance surfaces. Generated
`.pytest_cache` and `__pycache__` directories are disposable and are excluded
from the application workflow.

For lower resource use, prefer lazy xarray operations, narrow the spatial and
time ranges before selecting optional overlays, disable automatic updates
while editing several controls, and avoid simultaneous map, Hovmoller,
vertical-section, vector, and annual-cycle exports. These recommendations
preserve all user-input features while reducing peak NumPy conversion,
Cartopy rendering, and in-memory export costs.

## MATLAB compatibility

This is a Python/Streamlit application and does not require MATLAB. It does not use syntax from MATLAB. If you exchange NetCDF products with MATLAB R2020a, keep the files CF-like with explicit `lat`, `lon`, and `time` coordinates; the Python viewer reads those files independently.
# Secure remote geospatial ingestion

## Remote data architecture

The application keeps visualization and ingestion separate. `data_view.py`
continues to supply the existing NetCDF upload, local-folder, map, Hovmöller,
vertical-section, vector, anomaly, trend, and annual-cycle workflows. The
provider framework in `src/remote_ingestion.py` is responsible only for safe
remote access, spatial subsetting, unit conversion, and optional regional
area-weighted time-series extraction.

`BaseDataSourceProvider` is the provider contract and `ProviderRegistry` is
the factory. To add a provider, implement one adapter, register it, and add a
non-secret entry to `datasets.yaml`; neither the core pipeline nor downstream
analysis scripts need to change.

## Configuration and identity

Copy `.env.example` to an ignored `.env`. It contains only security policy and
authentication modes. `ALLOWED_DOMAINS` is an exact hostname allowlist; every
HTTP(S) endpoint is validated before the framework begins network I/O.
Credentials, URLs containing credentials, tokens, SAS strings, passwords, and
service-account JSON paths are prohibited from `datasets.yaml` and code.

| Provider | Identity mechanism |
| --- | --- |
| NASA Earthdata | `earthaccess.login(strategy="netrc")`; protect `~/.netrc` with owner-only permissions. |
| Azure / Planetary Computer | Planetary Computer signing for public assets or `DefaultAzureCredential` for private deployments. |
| Google Cloud Storage | Google Application Default Credentials, refreshed by `google-auth` / `gcsfs`. |
| Google Earth Engine | OAuth, ADC, or workload identity through the Earth Engine API. |

All direct HTTPS implementations use certifi roots and require TLS 1.2 or
newer. Do not log authentication headers, tokens, or credential paths.

For NASA Earthdata, create `%USERPROFILE%\_netrc` outside this repository with
your Earthdata Login username and password, using the host `urs.earthdata.nasa.gov`.
Restrict the file to your Windows account through NTFS permissions. The
application never reads credentials from `.env`, `datasets.yaml`, or the
project directory. If netrc authentication is unavailable, it stops before any
granule file is opened and reports the configuration requirement.

The application also recognizes an existing `%USERPROFILE%\.netrc` by setting
the Earthaccess `NETRC` path internally; no credentials are copied. Earthaccess
uses `_netrc` by default on Windows.

## Data products and execution flow

`fetch_standardized_grid()` returns a lazy `xarray.DataArray` with standard
`time`, `lat`, and `lon` coordinates, so it can directly feed the map,
Hovmöller, vertical-section, and existing analysis controls. It recognizes
both `[-180, 180]` and `[0, 360]` longitudes. A wrapped bbox is split at the
International Date Line and concatenated, preventing an accidental
near-global read.

`fetch_standardized_timeseries()` derives a `MS` Month-Start regional series.
It uses cosine-latitude weights for rectilinear grids and native `RAC` weights
for ECCO MDS data. ECCO first enumerates small metadata records, filters dates,
and only then streams required binary data. MERRA-2 uses CMR discovery and
remote Earthaccess handles. STAC COG, GCS Zarr, and Earth Engine adapters keep
data remote; Earth Engine reduces regions server-side before returning only a
small time series.

Run a validated extraction from `cmd` with the OC environment:

```cmd
conda run -n OC python -m src.remote_ingestion --dataset ecco_darwin_v5_co2_flux --start-date 2000-01-01 --end-date 2000-12-31 --min-lon 120 --max-lon 150 --min-lat 20 --max-lat 40 --config-path datasets.yaml
```

The optional `pystac-client`, `planetary-computer`, `azure-identity`, and
`gcsfs` dependencies are intentionally imported only when their provider is
selected. This keeps local NetCDF use lightweight and portable.
