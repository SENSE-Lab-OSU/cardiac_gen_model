#!/bin/bash
SOURCEDIR=$1

for i in {3..11..1}
  do 
    cp -r "$SOURCEDIR/S2_ppg_R2S" "$SOURCEDIR/S${i}_ppg_R2S"
    cp -r "$SOURCEDIR/S2_ecg_R2S" "$SOURCEDIR/S${i}_ecg_R2S"
  done

for i in {13..17..1}
  do 
    cp -r "$SOURCEDIR/S2_ppg_R2S" "$SOURCEDIR/S${i}_ppg_R2S"
    cp -r "$SOURCEDIR/S2_ecg_R2S" "$SOURCEDIR/S${i}_ecg_R2S"
  done