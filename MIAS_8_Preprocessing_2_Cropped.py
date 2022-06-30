from MIAS_2_Folders import ALLpng

from MIAS_2_Folders import CroppedBenignImages
from MIAS_2_Folders import CroppedMalignantImages
from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import CroppedTumorImages

from MIAS_4_MIAS_Functions import CroppedImages
from MIAS_4_MIAS_Functions import CroppedImagesPreprocessing
from MIAS_4_MIAS_Functions import MeanValue

CSV_Path = "D:\Mini-MIAS\Mini-MIAS\MiniMiasDataMod.csv"

Dataframe = CroppedImagesPreprocessing(CSV_Path)

MeanX = MeanValue(Dataframe, 4)

MeanY = MeanValue(Dataframe, 4)

Cropped_images = CroppedImages(ALLpng, CroppedBenignImages, CroppedMalignantImages, CroppedTumorImages, CroppedNormalImages, Dataframe, 112, MeanX, MeanY)
