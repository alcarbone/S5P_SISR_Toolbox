All images in this directory are extracted from level-1b Sentinel-5P radiance products freely available from [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2F9LF5Al1oqZalcdpGVkR1qrWF1qXaGBGJgtUPGvCxewM2prABJE8y0ckZxFpQGkP8qedMcSC960rAQW5eAu%2BFhiwrWaqmkEsoA6tRwveOS5r61S3jGWLBZ&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE) and are available [here](https://drive.google.com/drive/folders/1vG4QOVafxFis5HinjvQmPkquoDnIf9R9?usp=sharing).

Two scenes are retrieved from:
* **IN**  -> S5P_OFFL_L1B_RA_BDX_20230401T071049_20230401T085220_28317_03_020100_20230401T103831.nc, X = {2,3,4,5,6,7,8}.
* **US** -> S5P_OFFL_L1B_RA_BDX_20230709T195054_20230709T213224_29729_03_020100_20230709T232157.nc, X = {2,3,4,5,6,7,8}.
* **EG** -> S5P_OFFL_L1B_RA_BDX_20240804T102902_20240804T121032_35285_03_020100_20240804T140028.nc, X = {2,3,4,5,6,7,8}.

BD1 is avoided as its signal-to-noise ratio is quite low and its minimum resolution is too low.

The orbits are all pre-processed in this way:
* The concatenated images are cropped across-track in the pixels whose across-track resolution is maximum equal to 8 km.
* The interesting maximum and minimum latitudes are chosen and the orbit is cropped along-track in the chosen area.
* An evenly distributed and geometrically aligned grid of latitudes and longitudes is created from the available coordinates (no extrapolation of data is performed).
* The image is resampled with linear interpolation in the given grids.




