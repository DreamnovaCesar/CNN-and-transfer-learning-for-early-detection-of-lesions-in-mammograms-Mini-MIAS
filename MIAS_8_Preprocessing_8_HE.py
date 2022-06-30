import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import HE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import NOCroppedBenignImages
from MIAS_2_Folders import NOCroppedMalignantImages
from MIAS_2_Folders import NOCroppedNormalImages
from MIAS_2_Folders import NOCroppedTumorImages

from MIAS_2_Folders import HECroppedBenignImages
from MIAS_2_Folders import HECroppedMalignantImages
from MIAS_2_Folders import HECroppedNormalImages
from MIAS_2_Folders import HECroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data1 = HE_Technique(NOCroppedNormalImages, HECroppedNormalImages, Normal, IN)

DataFrame_Data2 = HE_Technique(NOCroppedTumorImages, HECroppedTumorImages, Tumor, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_HE' + '.csv'
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

DataFrame_Data4 = HE_Technique(NOCroppedBenignImages, HECroppedBenignImages, Normal, IB)

DataFrame_Data5 = HE_Technique(NOCroppedMalignantImages, HECroppedMalignantImages, Normal, IM)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data4, DataFrame_Data5], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_HE' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########