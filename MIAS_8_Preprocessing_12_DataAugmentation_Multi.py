

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import CroppedBenignImages
from MIAS_2_Folders import CroppedMalignantImages
from MIAS_2_Folders import CroppedNormalImages

from MIAS_2_Folders import NOCroppedBenignImages
from MIAS_2_Folders import NOCroppedMalignantImages
from MIAS_2_Folders import NOCroppedNormalImages

from MIAS_2_Folders import CLAHECroppedBenignImages
from MIAS_2_Folders import CLAHECroppedMalignantImages
from MIAS_2_Folders import CLAHECroppedNormalImages

from MIAS_2_Folders import HECroppedBenignImages
from MIAS_2_Folders import HECroppedMalignantImages
from MIAS_2_Folders import HECroppedNormalImages

from MIAS_2_Folders import UMCroppedBenignImages
from MIAS_2_Folders import UMCroppedMalignantImages
from MIAS_2_Folders import UMCroppedNormalImages

from MIAS_2_Folders import CSCroppedBenignImages
from MIAS_2_Folders import CSCroppedMalignantImages
from MIAS_2_Folders import CSCroppedNormalImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_2_Folders import MFCLAHECroppedBenignImages
from MIAS_2_Folders import MFCLAHECroppedMalignantImages
from MIAS_2_Folders import MFCLAHECroppedNormalImages

from MIAS_2_Folders import MFHECroppedBenignImages
from MIAS_2_Folders import MFHECroppedMalignantImages
from MIAS_2_Folders import MFHECroppedNormalImages

from MIAS_2_Folders import MFUMCroppedBenignImages
from MIAS_2_Folders import MFUMCroppedMalignantImages
from MIAS_2_Folders import MFUMCroppedNormalImages

from MIAS_2_Folders import MFCSCroppedBenignImages
from MIAS_2_Folders import MFCSCroppedMalignantImages
from MIAS_2_Folders import MFCSCroppedNormalImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_8_Preprocessing_13_CNN_Parameters import RAWTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import NOTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CLAHETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import HETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import UMTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CSTechnique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from MIAS_4_MIAS_Functions import TriclassPrinting
from MIAS_6_Data_Augmentation import DataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 14
Benign = 55
Malignant = 70

NNormal = 'Normal'
NBenign = 'Benign'
NMalignant = 'Malignant'

IN = 0
IB = 1
IM = 2

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_Normal, Labels_Normal = DataAugmentation(CroppedNormalImages, NNormal, Normal, IN)

Images_Benign, Labels_Benign = DataAugmentation(CroppedBenignImages, NBenign, Benign, IB)

Images_Malignant, Labels_Malignant = DataAugmentation(CroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NOImages_Normal, NOLabels_Normal = DataAugmentation(NOCroppedNormalImages, NNormal, Normal, IN)

NOImages_Benign, NOLabels_Benign = DataAugmentation(NOCroppedBenignImages, NBenign, Benign, IB)

NOImages_Malignant, NOLabels_Malignant = DataAugmentation(NOCroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CLAHEImages_Normal, CLAHELabels_Normal = DataAugmentation(CLAHECroppedNormalImages, NNormal, Normal, IN)

CLAHEImages_Benign, CLAHELabels_Benign = DataAugmentation(CLAHECroppedBenignImages, NBenign, Benign, IB)

CLAHEImages_Malignant, CLAHELabels_Malignant = DataAugmentation(CLAHECroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

HEImages_Normal, HELabels_Normal = DataAugmentation(HECroppedNormalImages, NNormal, Normal, IN)

HEImages_Benign, HELabels_Benign = DataAugmentation(HECroppedBenignImages, NBenign, Benign, IB)

HEImages_Malignant, HELabels_Malignant = DataAugmentation(HECroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

UMImages_Normal, UMLabels_Normal = DataAugmentation(UMCroppedNormalImages, NNormal, Normal, IN)

UMImages_Benign, UMLabels_Benign = DataAugmentation(UMCroppedBenignImages, NBenign, Benign, IB)

UMImages_Malignant, UMLabels_Malignant = DataAugmentation(UMCroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CSImages_Normal, CSLabels_Normal = DataAugmentation(CSCroppedNormalImages, NNormal, Normal, IN)

CSImages_Benign, CSLabels_Benign = DataAugmentation(CSCroppedBenignImages, NBenign, Benign, IB)

CSImages_Malignant, CSLabels_Malignant = DataAugmentation(CSCroppedMalignantImages, NMalignant, Malignant, IM)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

TriclassPrinting(Images_Normal, Images_Benign, Images_Malignant, RAWTechnique)
TriclassPrinting(NOImages_Normal, NOImages_Benign, NOImages_Malignant, NOTechnique)
TriclassPrinting(CLAHEImages_Normal, CLAHEImages_Benign, CLAHEImages_Malignant, CLAHETechnique)
TriclassPrinting(HEImages_Normal, HEImages_Benign, HEImages_Malignant, HETechnique)
TriclassPrinting(UMImages_Normal, UMImages_Benign, UMImages_Malignant, UMTechnique)
TriclassPrinting(CSImages_Normal, CSImages_Benign, CSImages_Malignant, CSTechnique)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
