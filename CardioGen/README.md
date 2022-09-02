Contains the main model & supporting library of functions in the directory "lib".

# Evaluating CardioGen
Uses pretrained weights provided in ..\data\post-training\model_weights_v12.
Generates and saves synthetic data to ..\data\pre-training\WESAD_synth_h30_m28

eval_CG saves the conditional generation ability and physiological feature 
evaluation results in ..\data\post-training\results\eval\h30_m28_dsamp2

1. Run Augmentor_ecg_st_id.py to generate and save synthetic ecg data to 
..\data\pre-training\WESAD_synth_h30_m28\E2StId. This will be used to train 
ecg2id and ecg2stress utility classifiers.

2. Run Augmentor_ecg2ppg_st_id.py to generate and save synthetic ppg data to 
..\data\pre-training\WESAD_synth_h30_m28\P2StId. This will be used to train 
ppg2id and ppg2stress utility classifiers.

3. Comment line 156 and Uncomment line 157 in Augmentor_ecg_st_id.py to use all 
real data for synthetic data generation used in further evaluations. 
Run eval_CG.py to generate and save this new synthetic ecg data to 
..\data\pre-training\WESAD_synth_h30_m28\eval. The same run will also produce 
and save the conditional generation ability and physiological feature 
evaluation results in ..\data\post-training\results\eval\h30_m28_dsamp2

4. Run *2stress and *2id (for both {ecg,ppg})
This will save classifier results for {real, synth, aug} data in 
'../experiments/13_1'.

# Training CardioGen
If you want to retrain CardioGen, HR2Rpeaks.py has the HRV module and 
Rpeaks2Sig.py has the Morphology module. 
1. Change the variable ver (for version) to any integer except 12. 
Then Run HR2Rpeaks.py and Rpeaks2Sig.py. This should save new set of weights in
..\data\post-training\model_weights_v{ver}.

2. Go to directory having common R2S weights saved in ../experiments/condWGAN/ 
after step 1. Manually create a copy of weights named S2_ecg_R2S and S2_ppg_R2S
and delete all except the latest checkpoint.

3. Run bash script in ../experiments/scripts/copy_specialized_models.sh to intialize
for getting model-specific weights. Specify path to the directory above as 
input argument or directly in the script.

4. Follow instruction in line 375 of Rpeaks2Sig.py to make the change. 
Run the same file for saving subject specific models.