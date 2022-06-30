import pandas as pd

from MIAS_4_MIAS_Functions import TexturesFeatureGLCM
from MIAS_4_MIAS_Functions import KmeansFunction
from MIAS_4_MIAS_Functions import KmeansGraph

from MIAS_2_Folders import CroppedBenignImages

Data_SO_B, X_Data_SO_B, Filename_SO_B = TexturesFeatureGLCM(CroppedBenignImages)

print(Data_SO_B)

kmeans_SO_B, y_kmeans_SO_B, N_clusters_SO_B = KmeansFunction(X_Data_SO_B, 2)

print(y_kmeans_SO_B)

KmeansGraph(X_Data_SO_B, kmeans_SO_B, y_kmeans_SO_B, N_clusters_SO_B)

DataFrame_SO_B = pd.DataFrame({'y_kmeans':y_kmeans_SO_B, 'REFNUM':Filename_SO_B})
pd.set_option('display.max_rows', DataFrame_SO_B.shape[0] + 1)
print(DataFrame_SO_B)

print(DataFrame_SO_B['y_kmeans'].value_counts())