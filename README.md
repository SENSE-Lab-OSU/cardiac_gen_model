# cardio_gen_model
A hierarchical generative model for cardiological signals (PPG,ECG etc.) that keeps the physiological characteristics intact.

## Description
This deep generative model comprises of two W-GAN's, one inside each of 
HR2Rpeaks_Simulator and Rpeaks2EcgPpg_Simulator objects. Both of these are 
conditional generative models which incrementally add desired marginal 
information over the given conditional information.

HR2Rpeaks_Simulator takes smooth Heart Rate (HR) averaged over a sliding 
window of 8s. and low-pass filtered along-with subject class as input 
condition and generates an R-peak train at Fs_out=100Hz with 1's at 
R-peak locations and 0's everywhere else. Internally, the W-GAN models 
uniformly-spaced tachograms at 5 Hz. Hence, HR2Rpeaks_Simulator adds 
subject-specific Heart Rate Variability (HRV) information to the input HR.

Rpeaks2EcgPpg_Simulator takes the R-peak train at Fs_in=100Hz. along-with 
subject class as input condition and generates an ECG signal at Fs_out=100Hz.
Hence, Rpeaks2EcgPpg_Simulator adds subject-specific Morphological (Morph) 
information to the input R-peak train.

Currently, the W-GAN in HR2Rpeaks_Simulator has a single set of weights while 
the W-GAN in Rpeaks2EcgPpg_Simulator has subject specific fine-tuned weights.

## Example Notebooks
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SENSE-Lab-OSU/cardio_gen_model/blob/master/demo.ipynb) 
Follow this Google-Colab demo link (also in demo.ipynb notebook) to get started with modulators.

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SENSE-Lab-OSU/cardio_gen_model/blob/master/demo_CC.ipynb) 
Follow this Google-Colab demo link (also in demo_CC.ipynb notebook) to get started with modulators along-with using data from Cerebral-Cortex and mFlow libraries. Will require more time and space for installing additional dependencies.