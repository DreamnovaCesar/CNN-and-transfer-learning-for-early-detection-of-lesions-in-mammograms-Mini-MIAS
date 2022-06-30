import cv2

from MIAS_4_MIAS_Functions import Resize_Images

from MIAS_2_Folders import CroppedBenignImages
from MIAS_2_Folders import CroppedMalignantImages
from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import CroppedTumorImages

png = '.png'

XsizeResized = 224
YsizeResized = 224

interpolation = cv2.INTER_CUBIC

ResizeImages_RAW_N = Resize_Images(CroppedBenignImages, XsizeResized, YsizeResized, interpolation, png)

ResizeImages_RAW_T = Resize_Images(CroppedMalignantImages, XsizeResized, YsizeResized, interpolation, png)

ResizeImages_RAW_N = Resize_Images(CroppedNormalImages, XsizeResized, YsizeResized, interpolation, png)

ResizeImages_RAW_T = Resize_Images(CroppedTumorImages, XsizeResized, YsizeResized, interpolation, png)
