# Introducing the pioneering super-resolution toolbox for Sentinel-5P data

A set of single-image super-resolution (SISR) algorithms designed specifically for Sentinel-5Precursor (S5P) Level-1b radiance data is offered herein.

S5P is a single-satellite mission launched by the European Space Agency (ESA) as part of the Copernicus program to monitor a large amount of gaseous air pollutants. Some helpful information on S5P is accessible on [ESA's official wiki](https://sentiwiki.copernicus.eu/web/s5p-mission). 

Two distinct types of S5P data are publicly accessible via the [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX19w36SwRKT6qYfJpcRdRdP6X9Z8Cc7xpWPmL6BW1rnaazx1QB4tTcqiQ58clVWtTZih7gZABvqUZFPCvgWbJDDvyxY7AoIg%2BnNKuiMDflT7morMQZBHoJjg&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE): Level-1b radiance data is split per detector's halves (S5P's payload, TROPOMI, has four detectors), whereas Level-2 data is split per product. More information about the data is publicly available in [S5P official reports](https://sentiwiki.copernicus.eu/web/s5p-documents). We chose to work with Level-1b images.

For further information, refer to our publications on this topic:
* [[1]](https://ieeexplore.ieee.org/document/10499875?source=authoralert) A. Carbone, R. Restaino, G. Vivone and J. Chanussot, "Model-Based Super-Resolution for Sentinel-5P Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5617716, doi: 10.1109/TGRS.2024.3387877.
* [[2]](https://ieeexplore.ieee.org/document/10663750) A. Carbone, R. Restaino and G. Vivone, "Efficient Hyperspectral Super-resolution of Sentinel-5P Data via Dynamic Multi-directional Cascade Fine-tuning," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2024.3454155.
* [[3]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12733/1273306/Super-resolution-techniques-for-Sentinel-5Pproducts/10.1117/12.2684083.short#_=_) A. Carbone, R. Restaino, and G. Vivone, "Super-resolution techniques for Sentinel-5P products," in Image and Signal Processing for Remote Sensing XXIX, vol. 12733, pp. 39-48, SPIE, 2023.


## Versions
This is **Version 1.1.0** of the S5P_SISR_Toolbox.

Version 1.1.0 illustrates the time-saving application of super-resolution algorithms on Sentinel-5P Level-1b data. The methods are indeed evaluated on two images with 3445 spectral channels taken from two different orbits. The images were pre-processed, as described [here](/data). The [IQA](/scripts/IQA) directory was changed to render the computation of RR indexes more robust in relation to the range of the images and to save time is the calculation of some of them.
The algorithms available are:
  - Interpolation, in particular:
    + Cubic interpolation
  - Non-blind deconvolution solved with Conjugate Gradient Algorithm.
  - DL-based methods, in particular:
    + Some SOTA neural networks ([SRCNN](https://arxiv.org/abs/1501.00092), [VDSR](https://arxiv.org/abs/1511.04587), [EDSR](https://arxiv.org/abs/1707.02921), [PAN](https://arxiv.org/abs/2010.01073), and [HAT](https://arxiv.org/abs/2205.04437)).
    + Our original neural network for S5P, i.e., **S5Net** fine-tuned with different strategies:
      1. _S5Net_: independent channel per channel fine-tuning.
      2. _GSR_S5Net_st_: static 2-directional cascade fine-tuning.
      3. _DSR_S5Net_st_: static 8-directional cascade fine-tuning.
      4. _GSR_S5Net_dyn_: dynamic 2-directional cascade fine-tuning.
      5. _DSR_S5Net_dyn_: dynamic 8-directional cascade fine-tuning.

This is a simple graphical representation of the proposed methodology, i.e., S5Net.
![S5Net architecture](/figs/S5Net.jpeg)
For clarity, we also report here the graphical representation of the time-saving dynamic multi-directional cascade fine-tuning we propose in contrast to the traditional fine-tuning. 
![Fine-tunings](/figs/finetunings.png)

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
the quality indices are saved as .csv files and the super-resolved images as .nc files into the directory _/results_. All utility scripts are always available in the directory _/scripts_, data to test on is available in the directory _/data_ and pre-trained models in the directory _/trained_models_.

## Main results
The scatter plot of the computational complexity in terms of the logarithmic number of fine-tuning iterations and the averaged PSNR obtained on IN, US, and EG datasets for all our fine-tuning approaches on the S5Net.
<img src="/figs/complexity.png" width="500">
The ground-truth (GT) compared to the super-resolved images obtained by some of all the exploited algorithms, including the original non-efficient S5Net and the best result we propose (DSR-S5Net-dyn) for a close-up of the IN dataset in a false-colour representation in which a single channel of SWIR, NIR, and UV are respectively employed as red, green, and blue.
![IN](/figs/IN.PNG)
The ground-truth (GT) compared to the super-resolved images obtained by some of all the exploited algorithms, including the original non-efficient S5Net and the best result we propose (DSR-S5Net-dyn) for a close-up of the US dataset in a false-colour representation in which a single channel of SWIR, NIR, and UV are respectively employed as red, green, and blue.
![US](/figs/US.PNG)
The ground-truth (GT) compared to the super-resolved images obtained by some of all the exploited algorithms, including the original non-efficient S5Net and the best result we propose (DSR-S5Net-dyn) for a close-up of the EG dataset in a false-colour representation in which a single channel of SWIR, NIR, and UV are respectively employed as red, green, and blue.
![EG](/figs/EG.PNG)
