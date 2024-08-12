Level-1b Sentinel-5P data for the central band of all detectors:
* ~300 nm for UV (270-320 nm),
* ~400 nm for UVIS (320-490 nm),
* ~725 nm for NIR (710-775 nm),
* ~2343 nm for SWIR (2305-2385 nm).

The images have a size of 512x256. The indices along-track are 2561-3073, while
the indices across-track are 66-322 for all detectors except for SWIR (in this case
the whole swath is considered).

The covered areas are:
* IN: India and Sri Lanka,
* US: California and part of Mexico.

The original files from which the images are retrieved are:
* IN: S5P_OFFL_L1B_RA_BDX_20230401T071049_20230401T085220_28317_03_020100_20230401T103831.nc,
* US: S5P_OFFL_L1B_RA_BDX_20230709T195054_20230709T213224_29729_03_020100_20230709T232157.nc.
where X is either 2,4,6, or 8.