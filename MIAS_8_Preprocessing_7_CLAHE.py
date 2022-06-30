import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import CLAHE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import NOCroppedBenignImages
from MIAS_2_Folders import NOCroppedMalignantImages
from MIAS_2_Folders import NOCroppedNormalImages
from MIAS_2_Folders import NOCroppedTumorImages

from MIAS_2_Folders import CLAHECroppedBenignImages
from MIAS_2_Folders import CLAHECroppedMalignantImages
from MIAS_2_Folders import CLAHECroppedNormalImages
from MIAS_2_Folders import CLAHECroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Clip_limit = 0.02

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data1 = CLAHE_Technique(NOCroppedNormalImages, CLAHECroppedNormalImages, Normal, Clip_limit, IN)

DataFrame_Data2 = CLAHE_Technique(NOCroppedTumorImages, CLAHECroppedTumorImages, Tumor, Clip_limit, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_CLAHE' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Benign = 'Benign'
Malignant = 'Malignant'

IN = 0  # Normal Images
IB = 1  # Tumor Images
IM = 2  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data3 = CLAHE_Technique(NOCroppedNormalImages, CLAHECroppedNormalImages, Normal, Clip_limit, IN)

DataFrame_Data4 = CLAHE_Technique(NOCroppedBenignImages, CLAHECroppedBenignImages, Normal, Clip_limit, IB)

DataFrame_Data5 = CLAHE_Technique(NOCroppedMalignantImages, CLAHECroppedMalignantImages, Normal, Clip_limit, IM)

DataFrame_ALL = pd.concat([DataFrame_Data3, DataFrame_Data4, DataFrame_Data5], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_CLAHE' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########