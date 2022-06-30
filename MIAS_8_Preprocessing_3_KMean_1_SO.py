import os
import pandas as pd

from MIAS_2_Folders import CroppedTumorImages
from MIAS_2_Folders import DataCSV
from MIAS_2_Folders import DataModels

from MIAS_4_MIAS_Functions import TexturesFeatureGLCM
from MIAS_4_MIAS_Functions import KmeansFunction
from MIAS_4_MIAS_Functions import KmeansGraph

Data_SO_T, X_Data_SO_T, Filename_SO_T = TexturesFeatureGLCM(CroppedTumorImages, 0)

pd.set_option('display.max_rows', Data_SO_T.shape[0] + 1)

dst = 'GLCM_CROPPED_IMAGE' + '.csv'
dstPath = os.path.join(DataCSV, dst)

Data_SO_T.to_csv(dstPath)

kmeans_SO_T, y_kmeans_SO_T, N_clusters_SO_T = KmeansFunction(X_Data_SO_T, 2)

print(y_kmeans_SO_T)

KmeansGraph(DataModels, 'GLCM', X_Data_SO_T, kmeans_SO_T, y_kmeans_SO_T, N_clusters_SO_T)

DataFrame_SO_T = pd.DataFrame({'y_kmeans':y_kmeans_SO_T, 'REFNUM':Filename_SO_T})
pd.set_option('display.max_rows', DataFrame_SO_T.shape[0] + 1)
print(DataFrame_SO_T)

pd.set_option('display.max_rows', DataFrame_SO_T.shape[0] + 1)

dst = 'GLCM_CROPPED_IMAGE_DATAFRAME' + '.csv'
dstPath = os.path.join(DataCSV, dst)

DataFrame_SO_T.to_csv(dstPath)

print(DataFrame_SO_T['y_kmeans'].value_counts())