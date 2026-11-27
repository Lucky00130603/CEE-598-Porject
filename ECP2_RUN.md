# ECP2–HadCM3 baseline run

The NOAA ZIP tracked in this repository supplied the training pairs. All four models ran for 60 epochs using the default random record split and four CPU threads. See `results/final-ecp2/evaluation/model_metrics.csv` for test metrics.

The 1971 pilot completed with all 260 grid cells. The full run covered 1971–2000 and 2041–2070, with all 260 grid cells complete in every year. The annual maxima and Urbana-area empirical IDF are in `results/final-ecp2`. The per-window NetCDF outputs are stored with Git LFS.

## Reproduce

Obtain Oscar’s six original ECP2 NetCDFs using the link in `data/NARCCAP/README.md`. Keep the originals and prepare a separate working directory. The future temperature files have the same native x/y coordinates and stereographic projection parameters as future precipitation, but their attached geographic coordinates differ. The inverse projection agrees with the precipitation coordinates. After independently verifying those axes and projection attributes, copy only the future precipitation file’s `lat_wgs84` and `lon_wgs84` arrays to both future temperature working copies. Leave temperature payloads and timestamps intact. The working copies and source files are excluded from this sync.

Use `--stamp end` for both periods. The corrected files lack precipitation time bounds, and the [NARCCAP time convention](https://www.narccap.ucar.edu/data/data-tables.html) defines precipitation flux as the average over the three hours preceding the timestamp.

```sh
python -m src.dashboard --run-dir results/final-ecp2
python -m src.project train --run-dir results/final-ecp2 --threads 4
python -m src.climate --model ecp2 --period historical --stamp end --climate-dir /path/to/working-copies
python -m src.climate --model ecp2 --period future --stamp end --climate-dir /path/to/working-copies
python -m src.project predict --model ecp2 --period historical --historical-years 1971 1971 --weights results/final-ecp2/training --run-dir results/pilot-ecp2 --stamp end --climate-dir /path/to/working-copies
python -m src.project predict --model ecp2 --period both --run-dir results/final-ecp2 --stamp end --save-windows --climate-dir /path/to/working-copies
```

The dashboard runs in a separate terminal. See `PROJECT_HANDOFF.md` for environment setup and interpretation limits. Each three-hour window produces one nested 15-minute depth estimate. The IDF uses 30 annual maxima per period.
