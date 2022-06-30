from MIAS_2_Folders import CroppedBenignImages
from MIAS_2_Folders import CroppedMalignantImages

from MIAS_4_MIAS_Functions import RemoveDataMias

from MIAS_8_Preprocessing_3_KMean_2_SO_B import DataFrame_SO_B
from MIAS_8_Preprocessing_3_KMean_2_SO_M import DataFrame_SO_M

DataFrame_FO_B_New, Filename_FO_B_New = RemoveDataMias(CroppedBenignImages, DataFrame_SO_B, 1)

DataFrame_SO_M_New, Filename_SO_M_New = RemoveDataMias(CroppedMalignantImages, DataFrame_SO_M, 1)