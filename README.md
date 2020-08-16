# The Cuttle Shuttle Analysis Code Repository

## Analysis codebase for *An experimental method for evoking and characterizing dynamic color patterning of cuttlefish during prey capture*

This folder contains the entire codebase necessary to replicate the analysis described in the paper ["An experimental method for evoking and characterizing dynamic color patterning of cuttlefish during prey capture" by Danbee Kim, Kendra Buresch, Roger Hanlon, and Adam R. Kampff](publication link coming soon!). 

[Download the full primary dataset](Coming soon!) from the Harvard Dataverse.

## Two Methods of Analysis

This analysis workflow uses two metrics for characterising the "Tentacle Shot Pattern" (TSP), a very brief and highly conspicuous body pattern displayed by cuttlefish just after "tentacles go ballistic" (TGB) during prey capture events, observed both in the lab and in the wild.

1) "Granularity" measure:  For each frame, filter the image at seven octave-wide, isotropic spatial frequency bands (denoted as Frequency Band 0-6, frequency band 0 being the largest spatial frequency and 6 being the smallest).  The sum of the squared pixel values in the resulting filtered images give the total energy of the original video frame in that particular frequency band.  This is a modified version of a granularity analysis method originally developed to discriminate between uniform/stipple, mottle and disruptive patterns in still images (for details, see *Barbosa, Alexandra, et al. "Cuttlefish camouflage: the effects of substrate contrast and size in evoking uniform, mottle or disruptive body patterns." Vision research 48.10 (2008): 1242-1253.*). This resulted in seven timeseries of numeric values, one for each frequency band, which describe the body pattern during each tentacle shot. 

2) "Edginess" measure:  For each frame, detect high-contrast edges using the Canny Edge Detector computer vision algorithm and sum the number of pixels involved in the detected edges in order to generate an “edginess score” for each frame.  This results in a timeseries of numeric values which describe the body pattern during each tentacle shot.  Each timeseries is smoothed in order to remove noise from the overhead fluorescent lighting in the experiment room (which generated moving waves of “flicker” in the video), by applying a Savitzky–Golay filter with a smoothing window of 250 milliseconds and using a third order polynomial for fitting (Python, `scipy.signal library`, function `savgol_filter`).

Our paper focuses on the "granularity" measure, and uses the "edginess" measure as an independent verification of the results from the "granularity" measure. 

## How to run this analysis workflow

### To replicate Results section "Accuracy of prey capture":

1) [Download the .csv files](https://www.dropbox.com/sh/zzm4kk9iis3cue1/AADCdZU6GV-bMp9tUrIL-glQa?dl=0) containing the timestamps of various "moments of interest" (MOIs) during the experimental sessions. 

2) Open file `CuttleShuttle_01_probabilityMOI.py`. Modify variable `dataset_dir` to point to the folder location where you saved the full experimental dataset (or just the `.csv` files). Modify the variable `plots_dir` to a folder where you would like to save the output of this script (`.png` image files). Save the file, then open a development environment set up to run python scripts, navigate to this folder, then run this script by typing ```python CuttleShuttle_01_probabilityMOI.py --MOI catches```. Type `python CuttleShuttle_01_probabilityMOI.py -h` for more info/options. Note that this script will generate a logfile named `probabilityMOIs_[today's date and time].log`.

### To replicate Results section "Numerical characterisations of TSP dynamics":

1) [Download the manually cropped and aligned videos of all tentacle shots made during the Cuttle Shuttle experiment](https://www.dropbox.com/sh/8jv8ngtjk8ngsas/AAAQ22UsdnWxsszJ1nDJnI8Da?dl=0) (referred to as "TGB videos") from the Harvard Dataverse. 

2) Open file `CuttleShuttle_02_ProcessCuttlePython_genBandEnergies.py` and find function `load_data` (line 47). Modify variable `video_dir` to point to the folder location where you saved the TGB videos. Modify variable `plots_dir` to point to a folder where you would like to save the output of this script (`.png` and `.npy` files). Save the file, then open a development environment set up to run python scripts, navigate to this folder, then run this script by typing ```python CuttleShuttle_02_ProcessCuttlePython_genBandEnergies.py```. Type `python CuttleShuttle_02_ProcessCuttlePython_genBandEnergies.py -h` for more info/options. Note that this script will generate a logfile named `process_cuttle_python_01_[today's date and time].log`.

3) Open file `CuttleShuttle_02_CannyEdgeDetector.bonsai` using the Bonsai visual language environment (to download visit the [official Bonsai website](https://bonsai-rx.org/)). Click the node called `GetFiles` and modify the `Path` parameter to point to the folder location where you saved the TGB videos. Click the `Start` button in the upper left of the program window. This will generate `.csv` files in the same folder as the one which contains the TGB videos. Files ending in `CannyCount.csv` contain an edginess score for each frame of the TGB videos; files ending in `PixelSum.csv` contain the sum of all pixels for each frame of the TGB videos. 

4) Open file `CuttleShuttle_03_analyseCatchVMiss.py` and find function `load_data` (line 55). Modify variable `data_dir_percentChange` to point to the folder location where you saved the output of `CuttleShuttle_02_ProcessCuttlePython_genBandEnergies.py`. Modify variable `data_dir_canny` to point to the folder location where you saved the output of `CuttleShuttle_02_CannyEdgeDetector.bonsai`. Modify variable `plots_dir` to point to a folder where you would like to save the output of this script (`.png` files). **WARNING: This script executes a non-parametric statistical test called a "shuffle test" or "permutation test", and will run for a few hours.** Save the file, then run this script by typing `python CuttleShuttle_03_analyseCatchVMiss.py`. Type ```python CuttleShuttle_03_analyseCatchVMiss.py -h``` for more info/options. Note that this script will generate a logfile named `process_cuttle_python_03_[today's date and time].log`.

5) Open file `CuttleShuttle_04_onsetTSP.py` and find function `load_data` (line 54). Modify variable `data_dir` to point to the folder location where you saved the output of `CuttleShuttle_02_ProcessCuttlePython_genBandEnergies.py`. Modify variable `plots_dir` to point to a folder where you would like to save the output of this script (`.png` files). Save the file, then run this script by typing ```python CuttleShuttle_04_onsetTSP.py```. Type `python CuttleShuttle_04_onsetTSP.py -h` for more info/options. Note that this script will generate a logfile named `process_cuttle_python_04_[today's date and time].log`.

