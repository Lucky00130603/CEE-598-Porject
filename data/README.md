# Data documentation

Large data files are not stored in this repository. This directory documents the inputs needed to reproduce the analysis.

## NOAA observations

Training and testing data come from NOAA Climate Data Online/NCEI:

- hourly precipitation (`HPCP`);
- 15-minute precipitation (`QPCP`);
- daily precipitation;
- daily maximum temperature;
- daily minimum temperature; and
- station ID, time, latitude, longitude, elevation, units, and quality flags.

The initial Illinois collection contains 91 stations with 15-minute data and 161 stations with hourly data. These are raw-source counts and may decrease after temporal alignment and quality control. Stations in neighboring states may be added to reduce boundary effects, following the regional approach in the reference study.

## NARCCAP simulations

Final application requires Illinois grid cells from the NARCCAP historical period (1971-2000) and future period (2041-2070):

- 3-hour precipitation;
- daily maximum temperature;
- daily minimum temperature; and
- grid-cell latitude and longitude.

## Processing requirements

The processing workflow must:

1. remove sentinel and quality-flagged values;
2. standardize units;
3. distinguish zero precipitation from unavailable gauge periods;
4. apply documented zero-padding rules where appropriate;
5. aggregate consecutive observations to construct 3-hour, 2-hour, and 30-minute precipitation;
6. join daily precipitation and temperature by station and date;
7. remove physically inconsistent observations; and
8. export the final aligned model table.

## Processed table schema

Required columns:

| Column | Description |
|---|---|
| `station_id` | NOAA station identifier |
| `timestamp` | Observation timestamp |
| `p_3h` | Three-hour precipitation depth |
| `p_2h` | Two-hour precipitation depth |
| `p_1h` | One-hour precipitation depth |
| `p_30min` | Thirty-minute precipitation depth |
| `p_15min` | Fifteen-minute precipitation depth |
| `daily_precip` | Daily precipitation depth |
| `tmax` | Daily maximum temperature |
| `tmin` | Daily minimum temperature |

Optional spatial columns:

| Column | Description |
|---|---|
| `latitude` | Station latitude |
| `longitude` | Station longitude |
| `elevation` | Station elevation |
| `neighbor_idw_mean` | Inverse-distance-weighted neighboring precipitation |
| `neighbor_max` | Maximum neighboring precipitation |
| `neighbor_std` | Standard deviation of neighboring precipitation |

All precipitation columns must use the same depth unit. Temperature units must also be consistent across NOAA training data and NARCCAP application data.
