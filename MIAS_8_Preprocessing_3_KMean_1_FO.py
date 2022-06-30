import os
import pandas as pd

from MIAS_2_Folders import CroppedNormalImages
from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import DataModels

from MIAS_4_MIAS_Functions import TexturesFeatureFirstOrder
from MIAS_4_MIAS_Functions import KmeansFunction
from MIAS_4_MIAS_Functions import KmeansGraph

Data_FO_N, X_Data_FO_N, Filename_FO_N = TexturesFeatureFirstOrder(CroppedNormalImages, 0)

pd.set_option('display.max_rows', Data_FO_N.shape[0] + 1)

dst = 'FO_CROPPED_IMAGE' + '.csv'
dstPath = os.path.join(DataCSV, dst)

Data_FO_N.to_csv(dstPath)

kmeans_FO_N, y_kmeans_FO_N, N_clusters_FO_N = KmeansFunction(X_Data_FO_N, 5)

print(y_kmeans_FO_N)

KmeansGraph(DataModels, 'First Order', X_Data_FO_N, kmeans_FO_N, y_kmeans_FO_N, N_clusters_FO_N)

DataFrame_FO_N = pd.DataFrame({'y_kmeans':y_kmeans_FO_N, 'REFNUM':Filename_FO_N})
pd.set_option('display.max_rows', DataFrame_FO_N.shape[0] + 1)

print(DataFrame_FO_N)

pd.set_option('display.max_rows', DataFrame_FO_N.shape[0] + 1)

dst = 'GLCM_CROPPED_IMAGE_DATAFRAME' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_FO_N.to_csv(dstPath)

print(DataFrame_FO_N['y_kmeans'].value_counts())