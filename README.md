# cardio_gen_model
A hierarchical generative model for cardiological signals (PPG,ECG etc.) that keeps some physiological characteristics intact.

## Description
CardioGen deep generative model comprises of two W-GAN's, one inside each of 
HR2Rpeaks_Simulator and Rpeaks2Sig_Simulator objects. Both of these are 
conditional generative models which incrementally add desired marginal 
information over the given conditional information.

HR2Rpeaks_Simulator takes smooth (filtered) Tachogram along-with stress 
condition and subject class as input and generates an R-peak train at 
a desired sampling frequency Fs_out with 1's at 
R-peak locations and 0's everywhere else. Internally, the W-GAN generates 
uniformly-spaced tachograms at 5 Hz. Hence, HR2Rpeaks_Simulator primarily adds 
subject-specific High Frequency (HF) Heart Rate Variability (HRV) information 
to the input.

Rpeaks2Sig_Simulator takes an R-peak train at Fs_in=100Hz. along-with stress 
condition and subject class as input and generates an ECG/PPG signal at 
Fs_out=100Hz/25Hz.Hence, Rpeaks2Sig_Simulator adds subject-specific 
Morphological (Morph) information to the input R-peak train.

Currently, the W-GAN in HR2Rpeaks_Simulator has a single set of weights while 
the W-GAN in Rpeaks2Sig_Simulator has subject specific fine-tuned weights.
Detailed instructions to reproduce evaluations and training of the paper are 
inside CardioGen/README.md

Code is functional but needs some refactoring to be more user-friendly.

## Example Notebooks
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SENSE-Lab-OSU/cardio_gen_model/blob/master/demo_augment_ppg.ipynb) 
Follow this Google-Colab demo link (also in demo_augment_ppg.ipynb notebook) that demonstrates augmenting PPG signals from WESAD using CardioGen.

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SENSE-Lab-OSU/cardio_gen_model/blob/master/demo_CC_ecg.ipynb) 
Follow this Google-Colab demo link (also in demo_CC_ecg.ipynb notebook) that demonstrates a python library "modulators.py" built using CardioGen to produce synthetic ECG data from WESAD. It also shows accessing data from Cerebral-Cortex libraries. Will require more time and space for installing additional Cerebral-Cortex dependencies.