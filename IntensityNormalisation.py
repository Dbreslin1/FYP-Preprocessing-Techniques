#Percentile clipping and foreground only z score normalisation.

#I made this script to create a new nnuNet raw dataset where the iamges are modified in the following way:
# 1. Foreground mask is estimated to reduce air domination in the images.
# 2. Intensities are clipped using percintiles to remove outliers.
# 3. Z score normalisation is applied using the foreground voxels only and not the background.
# 4. Voxels outside the foreground are set to 0.

#The labels are left unchanged.

import json
import shutil 
from pathlib import Path
import numpy as np
import nibabel as nib

#Configuration
srcID = 552 #baseline dataset ID
dstID = 553 #new dataset ID

nnUnet_raw = Path("/content/nnUnet_raw")

SRC = nnUnet_raw / f"Dataset{srcID:03d}_ImageTBAD_subX"
DST = nnUnet_raw / f"Dataset{dstID:03d}_ImageTBAD_subX_clipz"

(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)