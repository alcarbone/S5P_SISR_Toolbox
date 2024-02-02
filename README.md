# S5P_SISR_Toolbox 
Sentinel-5P Single-Image Super-Resolution Toolbox.

## Environment
All requirements for the environment used to run the codes are avilable in "requirements.txt" file.

## How to test
The main file is "Main_SR_RR_Benchmark.py" from which it is possible to choose the configuration of the image
to test the algorithms on. The following algorithms are tested: bi-cubic interpolation ("Cubic"), non-blind 
deconvolution solved with CGA and matching filters ("CGA_match"), non-blind deconvolution solved with CGA 
and no-matching filters ("CGA_nomatch"), SRCNN network ("SRCNN"), PAN network ("PAN"), HAT network ("HAT"), 
S5Net trained with matching filters ("S5Net_match"), S5Net trained with no-matching filters ("S5Net_nomatch") 
and S5Net without transposed convolution and bi-cubic interpolation ("S5Net_cubic"). 

All utility script are in the directory _/scripts_.

## Pre-trained models
In _/trained_models_ all pre-trained models can be found: in _/S5Net_ both "S5Net_match" and "S5Net_nomatch" cases
for each image and protocol, in _/S5Net_cubic_ "S5Net_cubic" can be found for each image and protocol, and in 
_/SOTA_ all state-of-the-art models, in particular "SRCNN" in _/SRCNN_, "PAN" in _/PAN_, and "HAT" in _/HAT_. 

## Results
Once tested all the algorithms, if results == True, save_csv.py is run and the quality indices are saved in 
_/results_ as .csv files and the super-resoluted images as .mat files. Q, ERGAS, sCC and PSNR are saved for RR
protocol and BRISQUE for FR protocol.
