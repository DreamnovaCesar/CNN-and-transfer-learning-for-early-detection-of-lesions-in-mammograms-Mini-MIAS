import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import MedianFilterNoise

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import UMCroppedBenignImages
from MIAS_2_Folders import UMCroppedMalignantImages
from MIAS_2_Folders import UMCroppedNormalImages
from MIAS_2_Folders import UMCroppedTumorImages

from MIAS_2_Folders import MFUMCroppedBenignImages
from MIAS_2_Folders import MFUMCroppedMalignantImages
from MIAS_2_Folders import MFUMCroppedNormalImages
from MIAS_2_Folders import MFUMCroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Division = 3

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data1 = MedianFilterNoise(UMCroppedNormalImages, MFUMCroppedNormalImages, Normal, Division, IN)

DataFrame_Data2 = MedianFilterNoise(UMCroppedTumorImages, MFUMCroppedTumorImages, Tumor, Division, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_MF_UM' + '.csv'
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

DataFrame_Data3 = MedianFilterNoise(UMCroppedNormalImages, MFUMCroppedNormalImages, Normal, Division, IN)

DataFrame_Data4 = MedianFilterNoise(UMCroppedBenignImages, MFUMCroppedBenignImages, Normal, Division, IB)

DataFrame_Data5 = MedianFilterNoise(UMCroppedMalignantImages, MFUMCroppedMalignantImages, Normal, Division, IM)

DataFrame_ALL = pd.concat([DataFrame_Data3, DataFrame_Data4, DataFrame_Data5], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_MF_UM' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########