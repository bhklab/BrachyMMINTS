# BrachyMMINTS
**MMINTS (Multi-Modal network with Imaging and Notes for Target Segmentation)**  
This is the official repository for "Clinical Examination-Augmented Multi-Modal Radiotherapy Clinical Target Volume Segmentation".

An extension of **nnU-Net** that integrates medical imaging with clinical text embeddings for improved automatic segmentation.

---

## Overview

MMINTS introduces a **cross-attention fusion module** to combine visual and textual information during training.  
The main trainer script is:

```
nnUNet/nnunet/training/network_training/nnUNetTrainerV2_CA_pubmed_v5.py
```

---

## Setup

1. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install -e ./nnUNet
   pip install transformers sentencepiece nibabel simpleitk
   ```

2. **Set nnU-Net environment variables**
   ```bash
   export nnUNet_raw_data_base=/path/to/nnUNet_raw
   export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
   export RESULTS_FOLDER=/path/to/nnUNet_results
   ```

---

## Training

```bash
python -m nnUNet_train 3d_fullres nnUNetTrainerV2_CA_pubmed_v5 XX 0
```

Replace `XX` with your dataset number (e.g., `72`).

---


## Notes

- Text embeddings should be saved as a .pkl file.  
- The trainer automatically loads embeddings based on case IDs.
