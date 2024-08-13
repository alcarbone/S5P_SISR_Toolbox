# Introducing the pioneering super-resolution toolbox for Sentinel-5P data

A set of single-image super-resolution (SISR) algorithms designed specifically for Sentinel-5Precursor (S5P) Level-1b radiance data is offered herein.

S5P is a single-satellite mission launched by the European Space Agency (ESA) as part of the Copernicus program to monitor a large amount of gaseous air pollutants. Some helpful information on S5P is accessible on [ESA's official wiki](https://sentiwiki.copernicus.eu/web/s5p-mission). 

Two distinct types of S5P data are publicly accessible via the [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX19w36SwRKT6qYfJpcRdRdP6X9Z8Cc7xpWPmL6BW1rnaazx1QB4tTcqiQ58clVWtTZih7gZABvqUZFPCvgWbJDDvyxY7AoIg%2BnNKuiMDflT7morMQZBHoJjg&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE): Level-1b radiance data is split per detector's halves (S5P's payload, TROPOMI, has four detectors), whereas Level-2 data is split per product. More information about the data is publicly available in [S5P official reports](https://sentiwiki.copernicus.eu/web/s5p-documents). We chose to work with Level-1b images.

For further information, refer to our publications on this topic:
* [[1]](https://ieeexplore.ieee.org/document/10499875?source=authoralert) A. Carbone, R. Restaino, G. Vivone and J. Chanussot, "Model-Based Super-Resolution for Sentinel-5P Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5617716, doi: 10.1109/TGRS.2024.3387877.
* [[2]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12733/1273306/Super-resolution-techniques-for-Sentinel-5Pproducts/10.1117/12.2684083.short#_=_) A. Carbone, R. Restaino, and G. Vivone, "Super-resolution techniques for Sentinel-5P products," in Image and Signal Processing for Remote Sensing XXIX, vol. 12733, pp. 39-48, SPIE, 2023. 

# Description of the toolbox

This is **Version 1.0.0** of the S5P_SISR_Toolbox.

Version 1.0.0 represents the first use of super-resolution methods on Sentinel-5P Level-1b data. The algorithms are tested on eight monochromatic images taken from two distinct orbits, corresponding to one channel for each detector. The images were simply cropped along- and across- track, with no additional pre-processing. 
The algorithms available are:
  - Interpolation, in particular:
    + Cubic interpolation
  - Non-blind deconvolution solved with Conjugate Gradient Algorithm.
  - DL-based methods, in particular:
    + Some SOTA neural networks ([SRCNN](https://arxiv.org/abs/1501.00092), [PAN](https://arxiv.org/abs/2010.01073), and [HAT](https://arxiv.org/abs/2205.04437)).
    + Our original neural network for S5P, i.e., **S5Net** fine-tuned independently for each channel.
The ablation study we conducted by substituting the transposed convolutional layer with cubic upsampling in our S5Net is available too.
      
This is a simple graphical representation of the proposed methodology, i.e., S5Net.
![S5Net architecture](/figs/S5Net.jpeg)

## Environment
The [_requirements.txt_](/requirements.txt) file contains all the specifications for the environment in which the code will be executed. Please install all the required before using the toolbox by executing the following command:

```
pip install -r requirements.txt
```

## Test the algorithms
The main file is called _Main_SR_Benchmark.py_ from which it is possible to choose the configuration to test the algorithms with. From this file the script _SR_algorithms.py_ is called and all algorithms are tested. 

When the variable _results_ is true, i.e.,
```
results = True
```
the quality indices are saved as .csv files and the super-resolved images as .mat files into the directory _/results/x{ratio}/{protocol}_, as already shown in the repository. All utility scripts are always available in the directory _/scripts_, data to test on is available in the directory _/data_ and pre-trained models in the directory _/trained_models_.

## Main results
The main results we obtained are herein shown. Our main goal was to design algorithms dependent on the imaging model of S5P (__match_ for short). In order to demonstrate that our approach is better than both algorithms usually employed in the literature and the same algorithms we propose but trained independently on how the sensors acquired the images in the first place (__nomatch_ for short), we show the main visual results at full resolution for both the datasets employed, i.e., IN and US. 

![IN dataset](/figs/IN_FR_results.jpeg)
![US dataset](/figs/US_FR_results.jpeg)

The two approaches (__match_ and __nomatch_), for both the algorithms we designed, are compared also in terms of their pixels' errors on both datasets.

![MSE maps](/figs/MSEs_maps.jpeg)
