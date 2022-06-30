import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_8_Preprocessing_5_Resize import XsizeResized
from MIAS_8_Preprocessing_5_Resize import YsizeResized

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import CroppedBenignImages
from MIAS_2_Folders import CroppedMalignantImages
from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import CroppedTumorImages

from MIAS_2_Folders import NOCroppedBenignImages
from MIAS_2_Folders import NOCroppedMalignantImages
from MIAS_2_Folders import NOCroppedNormalImages
from MIAS_2_Folders import NOCroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'
Benign = 'Benign'
Malignant = 'Malignant'

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_1, DataFrame_Data1 = Normalization(CroppedNormalImages, NOCroppedNormalImages, Normal, XsizeResized, YsizeResized, IN)

Images_2, DataFrame_Data2 = Normalization(CroppedTumorImages, NOCroppedTumorImages, Tumor, XsizeResized, YsizeResized, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_Normalization' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 1  # Normal Images
IM = 2  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_3, DataFrame_Data3 = Normalization(CroppedBenignImages, NOCroppedBenignImages, Benign, XsizeResized, YsizeResized, IB)

Images_4, DataFrame_Data4 = Normalization(CroppedMalignantImages, NOCroppedMalignantImages, Malignant, XsizeResized, YsizeResized, IM)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data3, DataFrame_Data4], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_Normalization' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########