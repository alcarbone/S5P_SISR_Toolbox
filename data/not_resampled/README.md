All images in this directory are extracted from level-1b Sentinel-5P radiance products freely available from [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2F9LF5Al1oqZalcdpGVkR1qrWF1qXaGBGJgtUPGvCxewM2prABJE8y0ckZxFpQGkP8qedMcSC960rAQW5eAu%2BFhiwrWaqmkEsoA6tRwveOS5r61S3jGWLBZ&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE).

Two scenes are retrieved from:
	o **IN**  -> S5P_OFFL_L1B_RA_BDX_20230401T071049_20230401T085220_28317_03_020100_20230401T103831.nc, with _X = {2,4,6,8}_.
	o **US** -> S5P_OFFL_L1B_RA_BDX_20230709T195054_20230709T213224_29729_03_020100_20230709T232157.nc, with _X = {2,4,6,8}_.

Only the central channel of each image is retrieved:
	o ~300 nm for UV (270-320 nm), _X = 2_.
	o ~400 nm for UVIS (320-490 nm), _X = 4_.
	o ~725 nm for NIR (710-775 nm), _X = 6_.
	o ~2343 nm for SWIR (2305-2385 nm), _X = 8_.

The images have a size of 512x256. The indices along-track are 2561-3073, while the indices across-track are 66-322 for all detectors except for SWIR (in this case the whole swath is considered).

