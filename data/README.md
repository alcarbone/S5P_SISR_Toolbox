All images in this directory are extracted from level-1b Sentinel-5P radiance products freely available from [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2F9LF5Al1oqZalcdpGVkR1qrWF1qXaGBGJgtUPGvCxewM2prABJE8y0ckZxFpQGkP8qedMcSC960rAQW5eAu%2BFhiwrWaqmkEsoA6tRwveOS5r61S3jGWLBZ&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE) and are available [here](https://drive.google.com/drive/folders/1vG4QOVafxFis5HinjvQmPkquoDnIf9R9?usp=sharing).

The covered areas are:
* AG: Algeria.
* CG: Congo.
* CS: Caspian Sea.
* EE: Eastern Europe.
* EG: Egypt.
* FR: France, part of Spain and Great Britain. 
* IN: India and Sri Lanka.
* IT: Italy.
* RS: Red sea.
* US: California and part of Mexico.

The original files' date and orbit number from which the images are retrieved are:
* AG: 20200623, 13962.
* CG: 20200623, 13962.
* CS: 20200623, 13960.
* EE: 20200623, 13961.
* EG: 20240804, 35285.
* FR: 20200623, 13962.
* IN: 20230401, 28317.
* IT: 20190804, 09365.
* RS: 20190804, 09364.
* US: 20230709, 29729.

BD1 is always avoided as its signal-to-noise ratio is quite low and its minimum resolution is too low.

The orbits are all pre-processed in this way:
* The concatenated images are cropped across-track in the pixels whose across-track resolution is maximum equal to 8 km.
* The interesting maximum and minimum latitudes are chosen from taking a central location and the orbit is cropped along-track in the chosen area.
* An evenly distributed and geometrically aligned grid of latitudes and longitudes is created from the available coordinates (no extrapolation of data is performed).
* The image is resampled with natural interpolation in the given grids.
