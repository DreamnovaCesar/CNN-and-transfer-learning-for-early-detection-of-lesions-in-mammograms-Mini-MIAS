import pandas as pd

from MIAS_4_MIAS_Functions import TexturesFeatureGLCM
from MIAS_4_MIAS_Functions import KmeansFunction
from MIAS_4_MIAS_Functions import KmeansGraph

from MIAS_2_Folders import CroppedMalignantImages

Data_SO_M, X_Data_SO_M, Filename_SO_M = TexturesFeatureGLCM(CroppedMalignantImages)

print(Data_SO_M)

kmeans_SO_M, y_kmeans_SO_M, N_clusters_SO_M = KmeansFunction(X_Data_SO_M, 2)

print(y_kmeans_SO_M)

KmeansGraph(X_Data_SO_M, kmeans_SO_M, y_kmeans_SO_M, N_clusters_SO_M)

DataFrame_SO_M = pd.DataFrame({'y_kmeans':y_kmeans_SO_M, 'REFNUM':Filename_SO_M})
pd.set_option('display.max_rows', DataFrame_SO_M.shape[0] + 1)
print(DataFrame_SO_M)

print(DataFrame_SO_M['y_kmeans'].value_counts())