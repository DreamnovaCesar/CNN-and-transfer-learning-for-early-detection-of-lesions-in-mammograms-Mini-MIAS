import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from MIAS_2_Folders import MultiDataModels
from MIAS_2_Folders import MultiDataModelsEsp

from MIAS_8_Preprocessing_13_CNN_Parameters import labels_Triclass

from MIAS_8_Preprocessing_13_CNN_Parameters import RAWTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import NOTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CLAHETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import HETechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import UMTechnique
from MIAS_8_Preprocessing_13_CNN_Parameters import CSTechnique

from MIAS_8_Preprocessing_13_CNN_Parameters import Valid_split
from MIAS_8_Preprocessing_13_CNN_Parameters import Epochs

from MIAS_8_Preprocessing_13_CNN_Parameters import XsizeResized
from MIAS_8_Preprocessing_13_CNN_Parameters import YsizeResized

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Images_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Labels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Images_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Labels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Images_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import Labels_Malignant

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOImages_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOLabels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOImages_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import NOLabels_Malignant

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHEImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHELabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHEImages_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHELabels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHEImages_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CLAHELabels_Malignant

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HEImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HELabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HEImages_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HELabels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HEImages_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import HELabels_Malignant

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMImages_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMLabels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMImages_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import UMLabels_Malignant

from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSImages_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSLabels_Benign
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSImages_Malignant
from MIAS_8_Preprocessing_12_DataAugmentation_Multi import CSLabels_Malignant

from MIAS_7_1_CNN_Architectures import PreTrainedModels, ResNet152_PreTrained
from MIAS_7_1_CNN_Architectures import MobileNetV3Small_Pretrained
from MIAS_7_1_CNN_Architectures import MobileNetV3Large_Pretrained
from MIAS_7_1_CNN_Architectures import MobileNet_Pretrained

from MIAS_7_1_CNN_Architectures import ResNet50_PreTrained
from MIAS_7_1_CNN_Architectures import ResNet50V2_PreTrained

from MIAS_7_1_CNN_Architectures import Xception_Pretrained

from MIAS_7_1_CNN_Architectures import VGG16_PreTrained
from MIAS_7_1_CNN_Architectures import VGG19_PreTrained

from MIAS_7_1_CNN_Architectures import InceptionV3_PreTrained

from MIAS_7_1_CNN_Architectures import DenseNet121_PreTrained

from MIAS_7_1_CNN_Architectures import CustomCNNAlexNet12_Model

from MIAS_7_1_CNN_Architectures import CustomCNNAlexNet12_Model

from MIAS_4_MIAS_Functions import ConfigurationModels
from MIAS_4_MIAS_Functions import UpdateCSV

df = pd.read_csv("D:\MIAS\MIAS VS\MultiDataCSV\Multiclass_DataFrame_MIAS_Data.csv")
path = "D:\MIAS\MIAS VS\MultiDataCSV\Multiclass_DataFrame_MIAS_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]

ModelTest = CustomCNNAlexNet12_Model

ModelValues =   [ModelTest, RAWTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]
ModelValues1 =  [ModelTest, NOTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, NOImages_Normal, NOLabels_Normal, NOImages_Benign, NOLabels_Benign, NOImages_Malignant, NOLabels_Malignant]
ModelValues2 =  [ModelTest, CLAHETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Normal, CLAHELabels_Normal, CLAHEImages_Benign, CLAHELabels_Benign, CLAHEImages_Malignant, CLAHELabels_Malignant]
ModelValues3 =  [ModelTest, HETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Normal, HELabels_Normal, HEImages_Benign, HELabels_Benign, HEImages_Malignant, HELabels_Malignant]
ModelValues4 =  [ModelTest, UMTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Normal, UMLabels_Normal, UMImages_Benign, UMLabels_Benign, UMImages_Malignant, UMLabels_Malignant]
ModelValues5 =  [ModelTest, CSTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Normal, CSLabels_Normal, CSImages_Benign, CSLabels_Benign, CSImages_Malignant, CSLabels_Malignant]

#ModelValues = [MobileNetV3Small_Pretrained, CLAHETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
"""
Score = ConfigurationModels(MainKeys, ModelValues, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 54)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues1, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 55)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues2, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 56)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues3, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 57)
"""
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues4, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 58)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues5, MultiDataModels, MultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 60) 

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

