import os
from re import I
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from MIAS_2_Folders import DataModels
from MIAS_2_Folders import DataModelsEsp

from MIAS_8_Preprocessing_13_CNN_Parameters import labels_Biclass

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

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import Images_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import Images_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import Labels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import Labels_Tumor

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import NOImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import NOImages_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import NOLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import NOLabels_Tumor

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CLAHEImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CLAHEImages_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CLAHELabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CLAHELabels_Tumor

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import HEImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import HEImages_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import HELabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import HELabels_Tumor

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import UMImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import UMImages_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import UMLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import UMLabels_Tumor

from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CSImages_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CSImages_Tumor
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CSLabels_Normal
from MIAS_8_Preprocessing_12_DataAugmentation_Bi import CSLabels_Tumor

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

from MIAS_4_MIAS_Functions import ConfigurationModels
from MIAS_4_MIAS_Functions import UpdateCSV

df = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\Biclass_DataFrame_MIAS_Data.csv")
path = "D:\MIAS\MIAS VS\DataCSV\Biclass_DataFrame_MIAS_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs', 'Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc"]

ModelTest = CustomCNNAlexNet12_Model

ModelValues =   [ModelTest, RAWTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Tumor, Labels_Tumor]
ModelValues1 =  [ModelTest, NOTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, NOImages_Normal, NOLabels_Normal, NOImages_Tumor, NOLabels_Tumor]
ModelValues2 =  [ModelTest, CLAHETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Normal, CLAHELabels_Normal, CLAHEImages_Tumor, CLAHELabels_Tumor]
ModelValues3 =  [ModelTest, HETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Normal, HELabels_Normal, HEImages_Tumor, HELabels_Tumor]
ModelValues4 =  [ModelTest, UMTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Normal, UMLabels_Normal, UMImages_Tumor, UMLabels_Tumor]
ModelValues5 =  [ModelTest, CSTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Normal, CSLabels_Normal, CSImages_Tumor, CSLabels_Tumor]

#ModelValues = [MobileNetV3Small_Pretrained, CLAHETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 54)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues1, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 55)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues2, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 56)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues3, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 57)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues4, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 58)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Score = ConfigurationModels(MainKeys, ModelValues5, DataModels, DataModelsEsp)

UpdateCSV(Score, df, column_names, path, 59) 

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########