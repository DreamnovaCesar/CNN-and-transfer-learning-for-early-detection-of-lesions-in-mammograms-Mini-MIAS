
from MIAS_6_Data_Augmentation import dataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import CroppedTumorImages

from MIAS_2_Folders import NOCroppedNormalImages
from MIAS_2_Folders import NOCroppedTumorImages

from MIAS_2_Folders import CLAHECroppedNormalImages
from MIAS_2_Folders import CLAHECroppedTumorImages

from MIAS_2_Folders import HECroppedNormalImages
from MIAS_2_Folders import HECroppedTumorImages

from MIAS_2_Folders import UMCroppedNormalImages
from MIAS_2_Folders import UMCroppedTumorImages

from MIAS_2_Folders import CSCroppedNormalImages
from MIAS_2_Folders import CSCroppedTumorImages


########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import MFCLAHECroppedNormalImages
from MIAS_2_Folders import MFCLAHECroppedTumorImages

from MIAS_2_Folders import MFHECroppedNormalImages
from MIAS_2_Folders import MFHECroppedTumorImages

from MIAS_2_Folders import MFUMCroppedNormalImages
from MIAS_2_Folders import MFUMCroppedTumorImages

from MIAS_2_Folders import MFCSCroppedNormalImages
from MIAS_2_Folders import MFCSCroppedTumorImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_8_Preprocessing_13_CNN_Parameters import RAWTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import NOTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CLAHETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import HETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import UMTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CSTechnique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_4_MIAS_Functions import BiclassPrinting
from MIAS_6_Data_Augmentation import dataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 22
Tumor = 46

NNormal = 'Normal'
NTumor = 'Tumor'

IN = 0
IT = 1


DataAug_0 = dataAugmentation(folder = NOCroppedNormalImages, severity = NNormal, sampling = Normal, label = IN, nfsave = True)
DataAug_1 = dataAugmentation(folder = NOCroppedTumorImages, severity = NTumor, sampling = Tumor, label = IT, nfsave = True)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_Normal, Labels_Normal = DataAug_0.DataAugmentation()

Images_Tumor, Labels_Tumor = DataAug_1.DataAugmentation()

"""
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NOImages_Normal, NOLabels_Normal = DataAugmentation(NOCroppedNormalImages, NNormal, Normal, IN)

NOImages_Tumor, NOLabels_Tumor = DataAugmentation(NOCroppedTumorImages, NTumor, Tumor, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CLAHEImages_Normal, CLAHELabels_Normal = DataAugmentation(CLAHECroppedNormalImages, NNormal, Normal, IN)

CLAHEImages_Tumor, CLAHELabels_Tumor = DataAugmentation(CLAHECroppedTumorImages, NTumor, Tumor, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

HEImages_Normal, HELabels_Normal = DataAugmentation(HECroppedNormalImages, NNormal, Normal, IN)

HEImages_Tumor, HELabels_Tumor = DataAugmentation(HECroppedTumorImages, NTumor, Tumor, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

UMImages_Normal, UMLabels_Normal = DataAugmentation(UMCroppedNormalImages, NNormal, Normal, IN)

UMImages_Tumor, UMLabels_Tumor = DataAugmentation(UMCroppedTumorImages, NTumor, Tumor, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CSImages_Normal, CSLabels_Normal = DataAugmentation(CSCroppedNormalImages, NNormal, Normal, IN)

CSImages_Tumor, CSLabels_Tumor = DataAugmentation(CSCroppedTumorImages, NTumor, Tumor, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BiclassPrinting(Images_Normal, Images_Tumor, RAWTechnique)
BiclassPrinting(NOImages_Normal, NOImages_Tumor, NOTechnique)
BiclassPrinting(CLAHEImages_Normal, CLAHEImages_Tumor, CLAHETechnique)
BiclassPrinting(HEImages_Normal, HEImages_Tumor, HETechnique)
BiclassPrinting(UMImages_Normal, UMImages_Tumor, UMTechnique)
BiclassPrinting(CSImages_Normal, CSImages_Tumor, CSTechnique)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
"""
