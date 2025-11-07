#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --mem 64G
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

source activate pygpu1

nnUNet_train 3d_fullres nnUNetTrainerV2_noDeepSupervision_DiceCE 80 4
