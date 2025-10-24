# Introducing the pioneering super-resolution toolbox for Sentinel-5P data

A set of single-image super-resolution (SISR) algorithms designed specifically for Sentinel-5Precursor (S5P) Level-1b radiance data is offered herein.

S5P is a single-satellite mission launched by the European Space Agency (ESA) as part of the Copernicus program to monitor a large amount of gaseous air pollutants. Some helpful information on S5P is accessible on [ESA's official wiki](https://sentiwiki.copernicus.eu/web/s5p-mission). 

Two distinct types of S5P data are publicly accessible via the [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX19w36SwRKT6qYfJpcRdRdP6X9Z8Cc7xpWPmL6BW1rnaazx1QB4tTcqiQ58clVWtTZih7gZABvqUZFPCvgWbJDDvyxY7AoIg%2BnNKuiMDflT7morMQZBHoJjg&datasetId=S2_L2A_CDAS&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE): Level-1b radiance data is split per detector's halves (S5P's payload, TROPOMI, has four detectors), whereas Level-2 data is split per product. More information about the data is publicly available in [S5P official reports](https://sentiwiki.copernicus.eu/web/s5p-documents). 

We chose to work with Level-1b images mainly for two reasons:
1. This allows us to prevent all of the degradation that happens during the complicated sequence of algorithms between level-1b and level-2 data.
2. It is therefore possible to generalise the applications. You are free to use the images exactly as they are given as a result or to extract maps of any preferred contaminant afterwards.

For further information, refer to our publications on this topic:
* [[1]](https://ieeexplore.ieee.org/document/10499875?source=authoralert) A. Carbone, R. Restaino, G. Vivone and J. Chanussot, "Model-Based Super-Resolution for Sentinel-5P Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5617716, doi: 10.1109/TGRS.2024.3387877.
* [[2]](https://ieeexplore.ieee.org/document/10663750) A. Carbone, R. Restaino and G. Vivone, "Efficient Hyperspectral Super-resolution of Sentinel-5P Data via Dynamic Multi-directional Cascade Fine-tuning," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2024.3454155.
* [[3]](https://ieeexplore.ieee.org/document/10640942) A. Carbone, R. Restaino, and G. Vivone, "Reduced and Full-Scale Assessment of Super-Resolution of Sentinel-5P Radiance Images," IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, Athens, Greece, 2024, pp. 1240-1244, doi: 10.1109/IGARSS53475.2024.10640942.
* [[4]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12733/1273306/Super-resolution-techniques-for-Sentinel-5Pproducts/10.1117/12.2684083.short#_=_) A. Carbone, R. Restaino, and G. Vivone, "Super-resolution techniques for Sentinel-5P products," in Image and Signal Processing for Remote Sensing XXIX, vol. 12733, pp. 39-48, SPIE, 2023.
* [[5]](https://shop.elsevier.com/books/deep-learning-for-multi-sensor-earth-observation/saha/978-0-443-26484-9) R. Restaino, A. Carbone, and G. Vivone, “Chapter 3 - Deep learning processing of remotely sensed multi-spectral images”, in Deep Learning for Multi-Sensor Earth Observation, S. Saha, Ed., in Earth Observation., Elsevier, 2025, pp. 57–85. doi: 10.1016/B978-0-44-326484-9.00012-9. 


## Versions
This is **Version 1.1.0** of the S5P_SISR_Toolbox.

Version 1.1.0 illustrates the time-saving application of super-resolution algorithms on Sentinel-5P Level-1b data. The methods are indeed evaluated on images with 3445 spectral channels taken from different orbits. The images were pre-processed, as described [here](/data). The [IQA](/scripts/IQA) directory was changed in comparison to the previous version to render the computation of RR indexes more robust in relation to the range of the images and to save time is the calculation of some of them.
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
The [_requirements.txt_](/requirements.txt) file contains all the specifications for the environment in which the code will be executed. Please install all the requirements before using the toolbox by executing the following command:

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
All ground-truth (GT) images that we preprocessed are shown below.
<p align="center">
  <img src="/figs/all_GTs.png">
  
The following table summarises their primary characteristics.

| **Tag** | **Date** | **Orbit** | **Coordinates** | **Location** |
|:--------|:----------|:-----------|:----------------|:-------------|
| AG | 2020-06-23 | 13962 | (32.2° N, 4.6° E) | Algeria |
| CG | 2020-06-23 | 13962 | (1.2° S, 15.1° E) | Congo Republic |
| CS | 2020-06-23 | 13960 | (40.0° N, 55.9° E) | Caspian Sea |
| EE | 2020-06-23 | 13961 | (47.7° N, 22.1° E) | Romania |
| EG | 2024-08-04 | 35285 | (25.5° N, 28.5° E) | Egypt |
| FR | 2020-06-23 | 13962 | (44.1° N, 2.7° E) | France |
| IN | 2023-04-01 | 28317 | (16.5° N, 79.8° E) | India |
| IT | 2019-08-04 | 09365 | (42.0° N, 11.8° E) | Italy |
| RS | 2019-08-04 | 09364 | (17.4° N, 41.8° E) | Red Sea |
| US | 2023-07-09 | 29729 | (35.5° N, 113.5° W) | Arizona |

The following figures display the arctangent of the difference between the MSE maps obtained using a PSF-unaware version of S5Net and our PSF-aware S5Net for a scaling factor of 4 and an area of 128 × 128 pixels from all original images. The same colour scale is applied across all maps. Green regions indicate areas where the PSF-aware approach achieves a lower reconstruction error.
![mse_maps_1](/figs/mse_maps_x4_RR_1.png)
![mse_maps_2](/figs/mse_maps_x4_RR_2-1.png)

The following figures display the ground-truth image, the result of bicubic interpolation, the output of a state-of-the-art approach (SRCNN), and the results obtained with the PSF-unaware and PSF-aware versions of S5Net, for a scaling factor of 4 and an area of 128 × 128 pixels extracted from the original images.
![RR_1](/figs/imgs_x4_RR_1-1.png)
![RR_2](/figs/imgs_x4_RR_2-1.png)

The following figures display the results of bicubic interpolation, a state-of-the-art method (SRCNN), and the PSF-unaware and PSF-aware versions of S5Net, for a scaling factor of 4. Each example corresponds to an area of 512 × 512 pixels, with an additional zoomed-in region of 128 × 128 pixels extracted from the original images.
![FR_1](/figs/imgs_x4_FR_1-1.png)
![FR_2](/figs/imgs_x4_FR_2-1.png)
![FR_3](/figs/imgs_x4_FR_3-1.png)
![FR_4](/figs/imgs_x4_FR_4-1.png)
![FR_5](/figs/imgs_x4_FR_5-1.png)

Scatter plot of the computational complexity, expressed as the logarithm of the number of fine-tuning iterations, versus the average PSNR obtained on the IN dataset for all proposed, state-of-the-art, and baseline fine-tuning approaches applied to S5Net.
<p align="center">
  <img src="/figs/PSNR_vs_epochs-1.png" width="450">
  
The ground-truth images are compared with the super-resolved outputs of several algorithms, including the original non-efficient S5Net and our best-performing method, DSR S5Net-dynamic, for close-ups of 128 × 128 for the CS, IN, IT, RS and US datasets, shown in false-colour representation.
![CS](/figs/CS_zoom-1.png)
![IN](/figs/IN_zoom-1.png)
![IT](/figs/IT_zoom-1.png)
![RS](/figs/RS_zoom-1.png)
![US](/figs/US_zoom-1.png)
