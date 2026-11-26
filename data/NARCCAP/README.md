# NARCCAP Climate Data Processing — Illinois

FIND ALL THE INFOR HERE, SOME FILES ARE ABOVE THE 25MB!
https://uillinoisedu-my.sharepoint.com/:f:/g/personal/oscarmc3_illinois_edu/IgBwv2KZqvAjRbaGgA5twRBYAWfMGhQmh544BTumscSM_Pw?e=OYxKbt

## 1. Information downloaded

- **Source:** UCAR THREDDS / NARCCAP archive (NCSS subsets).
- **Variables:** `pr` (3-hourly precipitation), `tasmin` and `tasmax` (daily temperature).
- **Models:** CRCM–CGCM3, ECP2–HadCM3, HRM3–GFDL, MM5I–CCSM.
- **Original periods:** 1968–2000 and 2038–2070.
- **Requested bounding box:** west −92°, east −87°, south 36°, north 43°.

## 2. Conversion and merging

- Downloaded NCSS NetCDF subsets and merged chronological source segments using **CDO `mergetime`**.
- Selected and verified each climate variable before merging.
- Aligned inconsistent **MM5I/HRM3** segment grids by retaining their common overlapping cells; used a `1e-4`-cell tolerance for coordinate roundoff.
- Repaired NetCDF data-type compatibility; corrected outputs use NetCDF4.

## 3. Spatial and temporal clipping

| Model | Final historical period | Final future period |
|---|---|---|
| MM5I–CCSM | 1968–1998 | 2038–2069 |
| CRCM–CGCM3 | 1968–Nov 2000 | 2038–Nov 2070 |
| HRM3–GFDL | 1968–2000 | 2038–2070 |
| ECP2–HadCM3 | 1968–2000 | 2038–2070 |

## 4. Reprojection / WGS84 geolocation

Added 2-D `lon_wgs84` and `lat_wgs84` coordinates referenced to **EPSG:4326**, using existing coordinates, CDO, or `sf` with source projection metadata. **Native model grids and values were preserved; no common regular WGS84 raster was generated.** Geographic coverage was audited using the new coordinates.

## 5. Filling missing data

- Converted `365_day` calendars by adding missing leap days; mapped `360_day` dates proportionally into Gregorian years.
- Interpolated bounded gaps of **≤24 hours**; longer supported gaps and endpoints use the nearest available value.
- Blocked repairs when a continuous **missing-timestamp gap exceeds 72 hours**; permanently masked cells stay missing.
- Saved corrected datasets and CSV audits separately from the originals.

## 6. Considerations

- Calendar conversion creates **synthetic timestamps and values**, especially for `360_day` datasets.
- Precipitation interpolation may change rainfall totals and extremes; verify these statistics before analysis.
- WGS84 geolocation **does not equal spatial regridding**: model resolutions and actual cell coverage remain different.
- Verify final time axes, missing values, coordinate consistency, and corrected-file integrity before machine-learning applications.

