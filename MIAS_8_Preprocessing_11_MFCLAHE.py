import os
import pandas as pd

from MIAS_5_Image_Processing_Functions import MedianFilterNoise

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import CLAHECroppedBenignImages
from MIAS_2_Folders import CLAHECroppedMalignantImages
from MIAS_2_Folders import CLAHECroppedNormalImages
from MIAS_2_Folders import CLAHECroppedTumorImages

from MIAS_2_Folders import MFCLAHECroppedBenignImages
from MIAS_2_Folders import MFCLAHECroppedMalignantImages
from MIAS_2_Folders import MFCLAHECroppedNormalImages
from MIAS_2_Folders import MFCLAHECroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 'Normal'
Tumor = 'Tumor'

IN = 0  # Normal Images
IT = 1  # Tumor Images

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Division = 3

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_Data1 = MedianFilterNoise(CLAHECroppedNormalImages, MFCLAHECroppedNormalImages, Normal, Division, IN)

DataFrame_Data2 = MedianFilterNoise(CLAHECroppedTumorImages, MFCLAHECroppedTumorImages, Tumor, Division, IT)

DataFrame_ALL = pd.concat([DataFrame_Data1, DataFrame_Data2], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Biclass_Dataframe_MF_CLAHE' + '.csv'
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

DataFrame_Data3 = MedianFilterNoise(CLAHECroppedNormalImages, MFCLAHECroppedNormalImages, Normal, Division, IN)

DataFrame_Data4 = MedianFilterNoise(CLAHECroppedBenignImages, MFCLAHECroppedBenignImages, Normal, Division, IB)

DataFrame_Data5 = MedianFilterNoise(CLAHECroppedMalignantImages, MFCLAHECroppedMalignantImages, Normal, Division, IM)

DataFrame_ALL = pd.concat([DataFrame_Data3, DataFrame_Data4, DataFrame_Data5], ignore_index = True, sort = False)

print(DataFrame_ALL)

pd.set_option('display.max_rows', DataFrame_ALL.shape[0] + 1)
print(DataFrame_ALL)

dst = 'Multiclass_Dataframe_MF_CLAHE' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

DataFrame_ALL.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########