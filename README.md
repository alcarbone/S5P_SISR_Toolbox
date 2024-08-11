# S5P_SISR_Toolbox 
Sentinel-5P Single-Image Super-Resolution Toolbox.

Code for "Model-Based Super-Resolution for Sentinel-5P Data" paper available at https://ieeexplore.ieee.org/document/10499875?source=authoralert. 

## Environment
All requirements for the environment used to run the codes are available in "requirements.txt" file.

## How to test
The main file is "Main_SR_RR_Benchmark.py" from which it is possible to choose the configuration of the image
to test the algorithms on. The following algorithms are tested: bi-cubic interpolation ("Cubic"), non-blind 
deconvolution solved with CGA ("CGA"), SRCNN network ("SRCNN"), VDSR network ("VDSR"), EDSR network ("EDSR"),
PAN network ("PAN"), HAT network ("HAT"), S5Net trained with independent fine-tuning ("S5Net"), S5Net trained with 
cascade fine-tuning: "GSR-S5Net-st", "GSR-S5Net-dyn", "DSR-S5Net-st", and "DSR-S5Net-dyn". 

All utility scripts are in the directory _/scripts_.

## Pre-trained models
Download pre-trained models from ! and put them in _/trained_models/S5Net_.

## Data 
Download pre-processed data from [!](https://drive.google.com/drive/folders/1vG4QOVafxFis5HinjvQmPkquoDnIf9R9?usp=drive_link) and put them in _/data_.

## Results
Once tested all the algorithms, if results == True, the quality indices are saved in 
_/results_ as .csv files and the super-resolved images as .nc files. 
