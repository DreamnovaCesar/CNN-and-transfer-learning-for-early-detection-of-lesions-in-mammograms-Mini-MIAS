import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import UM_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import NOCroppedBenignImages
from MIAS_2_Folders import NOCroppedMalignantImages
from MIAS_2_Folders import NOCroppedNormalImages
from MIAS_2_Folders import NOCroppedTumorImages

from MIAS_2_Folders import UMCroppedBenignImages
from MIAS_2_Folders import UMCroppedMalignantImages
from MIAS_2_Folders import UMCroppedNormalImages
from MIAS_2_Folders import UMCroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'

Radius = 1
Amount = 1

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data1 = UM_Technique(NOCroppedNormalImages, UMCroppedNormalImages, Normal, Radius, Amount, IN)

DataFrame_Data2 = UM_Technique(NOCroppedTumorImages, UMCroppedTumorImages, Tumor, Radius, Amount, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_UM' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IN = 0  # Normal Images
IB = 1  # Tumor Images
IM = 2  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data4 = UM_Technique(NOCroppedBenignImages, UMCroppedBenignImages, Normal, Radius, Amount, IB)

DataFrame_Data5 = UM_Technique(NOCroppedMalignantImages, UMCroppedMalignantImages, Radius, Amount, Amount, IM)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data4, DataFrame_Data5], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_UM' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########