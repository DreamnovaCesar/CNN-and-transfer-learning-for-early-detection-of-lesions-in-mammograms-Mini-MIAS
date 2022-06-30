import os
import pandas as pd

from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import CroppedTumorImages
from MIAS_2_Folders import DataCSV

from MIAS_4_MIAS_Functions import RemoveDataMias

from MIAS_8_Preprocessing_3_KMean_1_FO import DataFrame_FO_N
from MIAS_8_Preprocessing_3_KMean_1_SO import DataFrame_SO_T

DataFrame_FO_N_New = RemoveDataMias(CroppedNormalImages, DataFrame_FO_N, 4)

DataFrame_SO_T_New = RemoveDataMias(CroppedTumorImages, DataFrame_SO_T, 1)

print(DataFrame_FO_N_New)

pd.set_option('display.max_rows', DataFrame_FO_N_New.shape[0] + 1)

dst = 'FO_CROPPED_IMAGE_REMOVED' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_FO_N_New.to_csv(dstPath)

print(DataFrame_SO_T_New)

pd.set_option('display.max_rows', DataFrame_SO_T_New.shape[0] + 1)

dst = 'GLCM_CROPPED_IMAGE_REMOVED' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_SO_T_New.to_csv(dstPath)