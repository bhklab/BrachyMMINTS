#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --mem 64G
#SBATCH -c 3
#SBATCH --gres=gpu:2
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

source activate pygpu1

CUDA_VISIBLE_DEVICES=0,1
nnUNet_train_DP 3d_fullres nnUNetTrainerV2_DP_DPCE_epoch150 80 4 -gpus 2
