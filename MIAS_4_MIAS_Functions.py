import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glrlm import GLRLM

from skimage.feature import graycomatrix, graycoprops

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from MIAS_3_General_Functions import ShowSort
from MIAS_7_1_CNN_Architectures import PreTrainedModels

# Resize Image

def Resize_Images(Folder_Path, XsizeResized, YsizeResized, interpolation, Extension):

  Images = [] 

  os.chdir(Folder_Path)

  print(os.getcwd())
  print("\n")

  sorted_files, images = ShowSort(Folder_Path)
  count = 1

  for File in sorted_files:

    filename, extension  = os.path.splitext(File)

    if File.endswith(Extension): # Read png files

      try:
        print(f"Working with {count} of {images} normal images")
        count += 1

        Path_File = os.path.join(Folder_Path, File)
        Imagen = cv2.imread(Path_File)

        dsize = (XsizeResized, YsizeResized)
        Resized_Imagen = cv2.resize(Imagen, dsize, interpolation = interpolation)

        print(Imagen.shape, ' -------- ', Resized_Imagen.shape)

        dst = filename + Extension
        dstPath_N = os.path.join(Folder_Path, dst)

        cv2.imwrite(dstPath_N, Resized_Imagen)
        Images.append(Resized_Imagen)
        
      except OSError:
        print('Cannot convert %s ❌' % File)

  print("\n")
  print(f"COMPLETE {count} of {images} RESIZED ✅")

  return Images

# Transform Pgm to Png

class changeExtension:
  
  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)

    if self.folder == None:
      raise ValueError("Folder does not exist")

    elif self.newfolder == None:
      raise ValueError("Destination Folder does not exist")

    elif self.extension == None:
      raise ValueError("Extension does not exist")

    elif self.newextension == None:
      raise ValueError("New extension does not exist")

  def ChangeExtension(self):

    os.chdir(self.folder)

    print(os.getcwd())

    sorted_files, images = ShowSort(self.folder)
    count = 1
  
    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
            filename, extension  = os.path.splitext(File)
            print(f"Working with {count} of {images} {self.extension} images, {filename} ------- {self.newextension} ✅")
            count += 1
            
            Path_File = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_File)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            dst_name = filename + self.newextension
            dstPath_name = os.path.join(self.newfolder, dst_name)

            cv2.imwrite(dstPath_name, Imagen)
            #FilenamesREFNUM.append(filename)

        except OSError:
            print('Cannot convert %s ❌' % File)

    print("\n")
    print(f"COMPLETE {count} of {images} TRANSFORMED ✅")

# Transform CSV MIAS

def TransformedCSV(df_M):

  df_M.iloc[:, 3].values
  LE = LabelEncoder()
  df_M.iloc[:, 3] = LE.fit_transform(df_M.iloc[:, 3])

  df_M['X'] = df_M['X'].fillna(0)
  df_M['Y'] = df_M['Y'].fillna(0)
  df_M['RADIUS'] = df_M['RADIUS'].fillna(0)

  df_M["X"].replace({"*NOTE": 0}, inplace = True)
  df_M["Y"].replace({"3*": 0}, inplace = True)

  df_M['X'] = df_M['X'].astype(int)
  df_M['Y'] = df_M['Y'].astype(int)
  df_M['SEVERITY'] = df_M['SEVERITY'].astype(int)
  df_M['RADIUS'] = df_M['RADIUS'].astype(int)

  return df_M

# Cropped Images Mias

def CroppedImages(Folder_Path, New_Folder_Path_Benign, New_Folder_Path_Malignant, New_Folder_Path_Both, New_Folder_Path_Normal, df, cropped, MeanX, MeanY):

  Images = []

  os.chdir(Folder_Path)

  Refnum = 0
  Severity = 3
  Xcolumn = 4
  Ycolumn = 5
  #Radius = 6

  Benign = 0
  Malignant = 1
  Normal = 2

  Index = 1

  png = ".png"    # png.

  sorted_files, images = ShowSort(Folder_Path)
  count = 1
  k = 0

  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)

    print("******************************************")
    print(df.iloc[Index - 1, 0])
    print(filename)
    print("******************************************")

    if df.iloc[Index - 1, Refnum] == filename:
      if df.iloc[Index - 1, Severity] == Benign:
        if df.iloc[Index - 1, Xcolumn] > 0  or df.iloc[Index - 1, Ycolumn] > 0:

            try:

                print(f"Working with {count} of {images} {extension} Benign images, {filename}")
                print(df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                count += 1

                Path_File = os.path.join(Folder_Path, File)
                Imagen = cv2.imread(Path_File)
          
                Distance = cropped # Perimetro de X y Y de la imagen.
                CD = Distance / 2 # Centro de la imagen.
                YA = Imagen.shape[0] # YAltura.

                Xsize = df.iloc[Index - 1, Xcolumn]
                Ysize = df.iloc[Index - 1, Ycolumn]
                
                XDL = Xsize - CD
                XDM = Xsize + CD

                YDL = YA - Ysize - CD
                YDM = YA - Ysize + CD

                # Cropping an image
                Cropped_Image_Benig = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                print(Imagen.shape, " ----------> ", Cropped_Image_Benig.shape)

                # print(Cropped_Image_Benig.shape)
                # Display cropped image
                # cv2_imshow(cropped_image)

                dst_name = filename + '_Benign_cropped' + png

                dstPath_name = os.path.join(New_Folder_Path_Benign, dst_name)
                cv2.imwrite(dstPath_name, Cropped_Image_Benig)

                dstPath_name = os.path.join(New_Folder_Path_Both, dst_name)
                cv2.imwrite(dstPath_name, Cropped_Image_Benig)

                Images.append(Cropped_Image_Benig)

            except OSError:
                    print('Cannot convert %s' % File)

      elif df.iloc[Index - 1, Severity] == Malignant:
        if df.iloc[Index - 1, Xcolumn] > 0  or df.iloc[Index - 1, Ycolumn] > 0:

            try:

                print(f"Working with {count} of {images} {extension} Malignant images, {filename}")
                print(df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                count += 1

                Path_File = os.path.join(Folder_Path, File)
                Imagen = cv2.imread(Path_File)

                Distance = cropped # Perimetro de X y Y de la imagen.
                CD = Distance / 2 # Centro de la imagen.
                YA = Imagen.shape[0] # YAltura.

                Xsize = df.iloc[Index - 1, Xcolumn]
                Ysize = df.iloc[Index - 1, Ycolumn]

                XDL = Xsize - CD
                XDM = Xsize + CD

                YDL = YA - Ysize - CD
                YDM = YA - Ysize + CD

                # Cropping an image
                Cropped_Image_Malig = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                print(Imagen.shape, " ----------> ", Cropped_Image_Malig.shape)

                # print(Cropped_Image_Malig.shape)
                # Display cropped image
                # cv2_imshow(cropped_image)
            
                dst_name = filename + '_Malignant_cropped' + png

                dstPath_name = os.path.join(New_Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name, Cropped_Image_Malig)

                dstPath_name = os.path.join(New_Folder_Path_Both, dst_name)
                cv2.imwrite(dstPath_name, Cropped_Image_Malig)

                Images.append(Cropped_Image_Malig)


            except OSError:
                    print('Cannot convert %s' % File)
      
      elif df.iloc[Index - 1, Severity] == Normal:
        if df.iloc[Index - 1, Xcolumn] == 0  or df.iloc[Index - 1, Ycolumn] == 0:

            try:

                print(f"Working with {count} of {images} {extension} Normal images, {filename}")
                print(df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                count += 1

                Path_File = os.path.join(Folder_Path, File)
                Imagen = cv2.imread(Path_File)

                Distance = cropped # Perimetro de X y Y de la imagen.
                CD = Distance / 2 # Centro de la imagen.
                YA = Imagen.shape[0] # YAltura.

                Xsize = MeanX
                Ysize = MeanY

                XDL = Xsize - CD
                XDM = Xsize + CD

                YDL = YA - Ysize - CD
                YDM = YA - Ysize + CD

                # Cropping an image
                Cropped_Image_Normal = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                print(Imagen.shape, " ----------> ", Cropped_Image_Normal.shape)

                # print(Cropped_Image_Malig.shape)
                # Display cropped image
                # cv2_imshow(cropped_image)
            
                dst_name = filename + '_Normal_cropped' + png

                dstPath_name = os.path.join(New_Folder_Path_Normal, dst_name)
                cv2.imwrite(dstPath_name, Cropped_Image_Normal)

                Images.append(Cropped_Image_Normal)


            except OSError:
                    print('Cannot convert %s' % File)

      Index += 1
      k += 1    

    else:

      Index += 1
      k += 1

  return Images

# Extraction features

class FeatureExtraction():

  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    #self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.label = kwargs.get('label', None)

    if self.folder == None:
      raise ValueError("Folder does not exist")

    elif self.newfolder == None:
      raise ValueError("Destination Folder does not exist")

  def TexturesFeatureFirstOrder(self):

    Mean = []
    Var = []
    Skew = []
    Kurtosis = []
    Energy = []
    Entropy = []
    Labels = []

    #Images = [] # Png Images
    Filename = [] 

    os.chdir(self.folder)

    sorted_files, images = ShowSort(self.folder)
    count = 1

    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {self.extension} ✅")
          count += 1

          Path_File = os.path.join(self.folder, File)
          Imagen = cv2.imread(Path_File)
          
          #mean = np.mean(Imagen)
          #std = np.std(Imagen)
          #entropy = shannon_entropy(Imagen)
          #kurtosis_ = kurtosis(Imagen, axis = None)
          #skew_ = skew(Imagen, axis = None)
          #labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
          #"FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
          #"FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
          #"FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
          #"FOS_90Percentile","FOS_HistogramWidth"]

          Features, Labels_ = fos(Imagen, None)

          Mean.append(Features[0])
          Var.append(Features[1])
          Skew.append(Features[4])
          Kurtosis.append(Features[5])
          Energy.append(Features[6])
          Entropy.append(Features[7])
          Labels.append(self.label)

          Filename.append(filename)
          #Extensions.append(extension)

          #print(len(Mean))
          #print(len(Var))
          #print(len(Skew))
          #print(len(Kurtosis))
          #print(len(Energy))
          #print(len(Entropy))
          #print(len(Labels))
          #print(len(Filename))

        except OSError:
          print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'REFNUM':Filename, 'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values
    Y = Dataset.iloc[:, 0].values

    return Dataset, X, Y

  def TexturesFeatureGLRLM(self):

    SRE = []  # Short Run Emphasis
    LRE  = [] # Long Run Emphasis
    GLU = []  # Grey Level Uniformity
    RLU = []  # Run Length Uniformity
    RPC = []  # Run Percentage
    Labels = []
    #Images = [] # Png Images
    Filename = [] 

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    count = 1

    for File in sorted_files:
      if File.endswith(png): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {png} ✅")
          count += 1

          Path_File = os.path.join(Folder_Path, File)
          Imagen = cv2.imread(Path_File)
          Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

          app = GLRLM()
          glrlm = app.get_features(Imagen, 8)
          print(glrlm.Features)

          SRE.append(glrlm.Features[0])
          LRE.append(glrlm.Features[1])
          GLU.append(glrlm.Features[2])
          RLU.append(glrlm.Features[3])
          RPC.append(glrlm.Features[4])
          Labels.append(Label)

          Filename.append(filename)
          #Extensions.append(extension)

        except OSError:
          print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'REFNUM':Filename, 'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5]].values
    Y = Dataset.iloc[:, -1].values

    return Dataset, X, Y

  def TexturesFeatureGLCM(Folder_Path, Label):

    Dissimilarity = []
    Correlation = []
    Homogeneity = []
    Energy = []
    Contrast = []
    ASM = []
    Labels = []
    #Images = [] # Png Images
    Filename = [] 

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    count = 1

    for File in sorted_files:
      if File.endswith(png): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {png} ✅")
          count += 1

          Path_File = os.path.join(Folder_Path, File)
          Imagen = cv2.imread(Path_File)
          Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

          GLCM = graycomatrix(Imagen, [1], [0, np.pi/4, np.pi/2, 3 * np.pi/4])
          Energy.append(graycoprops(GLCM, 'energy')[0, 0])
          Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
          Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
          Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
          Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
          ASM.append(graycoprops(GLCM, 'ASM')[0, 0])
          Labels.append(Label)

          Filename.append(filename)
          #Extensions.append(extension)

        except OSError:
          print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'REFNUM':Filename, 'Energy':Energy, 'Correlation':Correlation, 'Homogeneity':Homogeneity, 'Dissimilarity':Dissimilarity, 'Contrast':Contrast, 'ASM':ASM, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values
    Y = Dataset.iloc[:, 0].values

    return Dataset, X, Y

# K-means function for MIAS

def KmeansFunction(X_Data, N_clusters):

    """
	  Using the elbow method and get k-means clusters.

    Parameters:
    argument1 (List): Data that will be cluster
    argument2 (int): How many cluster will be use

    Returns:
	  model:Returning kmeans model
    list:Returning kmeans y axis
    int:Returning number of clusters
   	"""

    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X_Data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 10), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters = N_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X_Data)

    return kmeans, y_kmeans, N_clusters

def KmeansGraph(Folder_Path, Name, X_Data, kmeans, y_kmeans, N_clusters):

  """
	Graph Kmeans

  Parameters:
  argument1 (List): Data that will be cluster
  argument2 (model): Model K-means already trained
  argument3 (List): prediction from kmeans y axis
  argument4 (int): Clusters' number to graph

  Returns:
	void
  """

  Colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple']

  for i in range(N_clusters):

     if  N_clusters <= 10:

        plt.scatter(X_Data[y_kmeans == i, 0], X_Data[y_kmeans == i, 1], s = 100, c = Colors[i], label = 'Cluster ' + str(i))


  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

  plt.title('Clusters')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend()

  dst = 'Kmeans_Graph' + str(Name) + '.png'
  dstPath = os.path.join(Folder_Path, dst)

  plt.savefig(dstPath)

  plt.show()

# Remove Data from K-means function

def RemoveDataMias(Folder_Path, df, ClusterRemoved):

  """
	Remove the cluster chosen from dataframe

  Parameters:
  argument1 (Folder): Folder's path
  argument2 (dataframe): dataframe that will be used to remove data
  argument3 (int): the cluster's number that will be remove

  Returns:
	dataframe:Returning dataframe already modified
  """

  #Images = [] # Png Images
  Filename = [] 
  DataRemove = []
  Data = 0

  KmeansValue = 0
  Refnum = 1

  png = ".png"  # png.

  os.chdir(Folder_Path)

  sorted_files, images = ShowSort(Folder_Path)
  count = 1
  Index = 1

  for File in sorted_files:
    filename, extension  = os.path.splitext(File)
    if df.iloc[Index - 1, Refnum] == filename: # Read png files

      print(filename)
      print(df.iloc[Index - 1, Refnum])

      if df.iloc[Index - 1, KmeansValue] == ClusterRemoved:

        try:
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {png} ✅")
          count += 1

          Path_File = os.path.join(Folder_Path, File)
          #print(Path_File)
          os.remove(Path_File)
          print(df.iloc[Index - 1, Refnum], ' removed ❌')
          DataRemove.append(count)
          Data += 0
          #df = df.drop(df.index[count])

        except OSError:
          print('Cannot convert %s ❌' % File)

      elif df.iloc[Index - 1, KmeansValue] != ClusterRemoved:
        
        Filename.append(filename)

      Index += 1

    elif df.iloc[Index - 1, Refnum] != filename:
      
      print(filename)
      print(df.iloc[Index - 1, Refnum])
      print('Files are not equal')
      break

    else:
  
      Index += 1

    for i in range(Data):

      df = df.drop(df.index[DataRemove[i]])

  #Dataset = pd.DataFrame({'y_kmeans':df_u.iloc[Index - 1, REFNUM], 'REFNUM':df_u.iloc[Index - 1, KmeansValue]})
  #X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

  return df

# Mean Value for cropping

def MeanValue(df_M, column):

    """
	  Obtaining the mean value of the mammograms

    Parameters:
    argument1 (dataframe): dataframe that will be use to acquire the values
    argument2 (int): the column number to get the mean value

    Returns:
	  float:Returning the mean value
    """

    Data = []

    for i in range(df_M.shape[0]):
        if df_M.iloc[i - 1, column] > 0:
            Data.append(df_M.iloc[i - 1, column])

    Mean = int(np.mean(Data))

    print(Data)
    print(Mean)

    return Mean

# Pre processing for cropping MIAS

def CroppedImagesPreprocessing(CSV_Path):

    col_list = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
    df = pd.read_csv(CSV_Path, usecols = col_list)

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)

    df_T = TransformedCSV(df)

    pd.set_option('display.max_rows', df_T.shape[0] + 1)
    print(df_T)

    return df_T

# Texture features.

#Tamura

def coarseness(image, kmax):

	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2, k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return fcrs

def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image),4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon

def directionality(image):
	image = np.array(image, dtype = 'int64')
	h = image.shape[0]
	w = image.shape[1]
	convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	deltaH = np.zeros([h,w])
	deltaV = np.zeros([h,w])
	theta = np.zeros([h,w])

	# calc for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = image[0][wi+1] - image[0][wi]
		deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = image[hi][1] - image[hi][0]
		deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

	# calc for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = image[1][wi] - image[0][wi]
		deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = image[hi+1][0] - image[hi][0]
		deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calc the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0;
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

	n = 16
	t = 12
	cnt = 0
	hd = np.zeros(n)
	dlen = deltaG_vec.shape[0]
	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
	hd = hd / np.mean(hd)
	hd_max_index = np.argmax(hd)
	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]
	return fdir

def linelikeness(image, sita, dist):
	pass

def regularity(image, filter):
	pass

def roughness(fcrs, fcon):
	return fcrs + fcon

# First Order features

def fos(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Returns
    -------
    features : numpy ndarray
        1)Mean, 2)Variance, 3)Median (50-Percentile), 4)Mode, 
        5)Skewness, 6)Kurtosis, 7)Energy, 8)Entropy, 
        9)Minimal Gray Level, 10)Maximal Gray Level, 
        11)Coefficient of Variation, 12,13,14,15)10,25,75,90-
        Percentile, 16)Histogram width
    labels : list
        Labels of features.
    '''
    if mask is None:
        mask = np.ones(f.shape)
    
    # 1) Labels
    labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
              "FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
              "FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
              "FOS_90Percentile","FOS_HistogramWidth"]
    
    # 2) Parameters
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255
    Ng = (level_max - level_min) + 1
    bins = Ng
    
    # 3) Calculate Histogram H inside ROI
    f_ravel = f.ravel() 
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)] 
    H = np.histogram(roi, bins = bins, range = [level_min, level_max], density = True)[0]
    
    # 4) Calculate Features
    features = np.zeros(16, np.double)  
    i = np.arange(0, bins)
    features[0] = np.dot(i, H)
    features[1] = sum(np.multiply((( i- features[0]) ** 2), H))
    features[2] = np.percentile(roi, 50) 
    features[3] = np.argmax(H)
    features[4] = sum(np.multiply(((i-features[0]) ** 3), H)) / (np.sqrt(features[1]) ** 3)
    features[5] = sum(np.multiply(((i-features[0]) ** 4), H)) / (np.sqrt(features[1]) ** 4)
    features[6] = sum(np.multiply(H, H))
    features[7] = -sum(np.multiply(H, np.log(H + 1e-16)))
    features[8] = min(roi)
    features[9] = max(roi)
    features[10] = np.sqrt(features[2]) / features[0]
    features[11] = np.percentile(roi, 10) 
    features[12] = np.percentile(roi, 25)  
    features[13] = np.percentile(roi, 75) 
    features[14] = np.percentile(roi, 90) 
    features[15] = features[14] - features[11]
    
    return features, labels

# First Order features Images

def TexturesFeatureFirstOrderImage(Images, Label):

  Fof = 'First Order Features'

  Mean = []
  Var = []
  Skew = []
  Kurtosis = []
  Energy = []
  Entropy = []
  Labels = []

  count = 1

  for File in range(len(Images)):

      try:

          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          Images[File] = cv2.cvtColor(Images[File], cv2.COLOR_BGR2GRAY)

          Features, Labels_ = fos(Images[File], None)

          Mean.append(Features[0])
          Var.append(Features[1])
          Skew.append(Features[4])
          Kurtosis.append(Features[5])
          Energy.append(Features[6])
          Entropy.append(Features[7])
          Labels.append(Label)

      except OSError:
          print('Cannot convert %s ❌' % File)

  Dataset = pd.DataFrame({'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

  X = Dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Fof

# GLRLM features Images

def TexturesFeatureGLRLMImage(Images, Label):

  Glrlm = 'Gray-Level Run Length Matrix'

  SRE = []  # Short Run Emphasis
  LRE  = [] # Long Run Emphasis
  GLU = []  # Grey Level Uniformity
  RLU = []  # Run Length Uniformity
  RPC = []  # Run Percentage
  Labels = []

  count = 1

  for File in range(len(Images)):

      try:
          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          app = GLRLM()
          glrlm = app.get_features(Images[File], 8)

          SRE.append(glrlm.Features[0])
          LRE.append(glrlm.Features[1])
          GLU.append(glrlm.Features[2])
          RLU.append(glrlm.Features[3])
          RPC.append(glrlm.Features[4])
          Labels.append(Label)

      except OSError:
          print('Cannot convert %s ❌' % File)

  Dataset = pd.DataFrame({'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

  X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Glrlm

# GLCM features Images

def TexturesFeatureGLCMImage(Images, Label):

  Glcm = 'Gray-Level Co-Occurance Matrix'

  Dataset = pd.DataFrame()
  
  Dissimilarity = []
  Correlation = []
  Homogeneity = []
  Energy = []
  Contrast = []

  Dissimilarity2 = []
  Correlation2 = []
  Homogeneity2 = []
  Energy2 = []
  Contrast2 = []

  Dissimilarity3 = []
  Correlation3 = []
  Homogeneity3 = []
  Energy3 = []
  Contrast3 = []

  Dissimilarity4 = []
  Correlation4 = []
  Homogeneity4 = []
  Energy4 = []
  Contrast4 = []

  #Entropy = []
  #ASM = []
  Labels = []
  Labels2 = []
  Labels3 = []
  Labels4 = []

  count = 1

  for File in range(len(Images)):

      try:
          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          Images[File] = cv2.cvtColor(Images[File], cv2.COLOR_BGR2GRAY)

          GLCM = graycomatrix(Images[File], [1], [0])
          Energy.append(graycoprops(GLCM, 'energy')[0, 0])
          Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
          Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
          Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
          Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
         
          GLCM2 = graycomatrix(Images[File], [1], [np.pi/4])
          Energy2.append(graycoprops(GLCM2, 'energy')[0, 0])
          Correlation2.append(graycoprops(GLCM2, 'correlation')[0, 0])
          Homogeneity2.append(graycoprops(GLCM2, 'homogeneity')[0, 0])
          Dissimilarity2.append(graycoprops(GLCM2, 'dissimilarity')[0, 0])
          Contrast2.append(graycoprops(GLCM2, 'contrast')[0, 0])

          GLCM3 = graycomatrix(Images[File], [5], [np.pi/2])
          Energy3.append(graycoprops(GLCM3, 'energy')[0, 0])
          Correlation3.append(graycoprops(GLCM3, 'correlation')[0, 0])
          Homogeneity3.append(graycoprops(GLCM3, 'homogeneity')[0, 0])
          Dissimilarity3.append(graycoprops(GLCM3, 'dissimilarity')[0, 0])
          Contrast3.append(graycoprops(GLCM3, 'contrast')[0, 0])

          GLCM4 = graycomatrix(Images[File], [5], [3 * np.pi/4])
          Energy4.append(graycoprops(GLCM4, 'energy')[0, 0])
          Correlation4.append(graycoprops(GLCM4, 'correlation')[0, 0])
          Homogeneity4.append(graycoprops(GLCM4, 'homogeneity')[0, 0])
          Dissimilarity4.append(graycoprops(GLCM4, 'dissimilarity')[0, 0])
          Contrast4.append(graycoprops(GLCM4, 'contrast')[0, 0])
         
          Labels.append(Label)
          # np.pi/4
          # np.pi/2
          # 3*np.pi/4

      except OSError:
          print('Cannot convert %s ❌' % File)
  

  Dataset = pd.DataFrame({'Energy':Energy,  'Homogeneity':Homogeneity,  'Contrast':Contrast,  'Correlation':Correlation,
                          'Energy2':Energy, 'Homogeneity2':Homogeneity, 'Contrast2':Contrast, 'Correlation2':Correlation, 
                          'Energy3':Energy, 'Homogeneity3':Homogeneity, 'Contrast3':Contrast, 'Correlation3':Correlation, 
                          'Energy4':Energy, 'Homogeneity4':Homogeneity, 'Contrast4':Contrast, 'Correlation4':Correlation, 'Labels3':Labels})


  #'Energy':Energy
  #'Homogeneity':Homogeneity
  #'Correlation':Correlation
  #'Contrast':Contrast
  #'Dissimilarity':Dissimilarity

  X = Dataset.iloc[:, [0, 1]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Glcm

# CNN Configuration

def ConfigurationModels(MainKeys, Arguments, Folder_Save, Folder_Save_Esp):

    TotalImage = []
    TotalLabel = []

    ClassSize = (len(Arguments[2]))
    Images = 7
    Labels = 8

    if len(Arguments) == len(MainKeys):
        
        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            #print('Total:', len(TotalImage))
        
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            #print('Total:', len(TotalLabel))

            Images += 2
            Labels += 2

        #TotalImage = [*list(DicAruments.values())[Images], *list(DicAruments.values())[Images + 2]]
        
    elif len(Arguments) > len(MainKeys):

        TotalArguments = len(Arguments) - len(MainKeys)

        for i in range(TotalArguments // 2):

            MainKeys.append('Images ' + str(i + 3))
            MainKeys.append('Labels ' + str(i + 3))

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            Images += 2
            Labels += 2

    elif len(Arguments) < len(MainKeys):

        raise ValueError('No se puede xDD')

    #print(DicAruments)

    def printDict(DicAruments):

        for i in range(7):
            print(list(DicAruments.items())[i])

    printDict(DicAruments)

    print(len(TotalImage))
    print(len(TotalLabel))

    X_train, X_test, y_train, y_test = train_test_split(np.array(TotalImage), np.array(TotalLabel), test_size = 0.20, random_state = 42)

    Score = PreTrainedModels(Arguments[0], Arguments[1], Arguments[2], Arguments[3], Arguments[4], ClassSize, Arguments[5], Arguments[6], X_train, y_train, X_test, y_test, Folder_Save, Folder_Save_Esp)
    #Score = PreTrainedModels(ModelPreTrained, technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test)
    return Score

# Update CSV changing value

def UpdateCSV(Score, df, column_names, path, row):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Tumor images.
    argument3 (str): Technique used

    Returns:
	  void
   	"""
     
    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
  
    df.to_csv(path, index = False)
  
    print(df)

# Biclass printing data augmentation

def BiclassPrinting(ImagesNormal, ImagesTumor, Technique):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Tumor images.
    argument3 (str): Technique used

    Returns:
	  void
   	"""

    print("\n")
    print(Technique + ' Normal images: ' + str(len(ImagesNormal)))
    print(Technique + ' Tumor images: ' + str(len(ImagesTumor)))

# Triclass printing data augmentation

def TriclassPrinting(ImagesNormal, ImagesBenign, ImagesMalignant, Technique):

  """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Benign images.
    argument3 (list): The number of Malignant images.
    argument4 (str): Technique used

    Returns:
	  void
   	"""

  print("\n")
  print(Technique + ' Normal images: ' + str(len(ImagesNormal)))
  print(Technique + ' Normal images: ' + str(len(ImagesBenign)))
  print(Technique + ' Normal images: ' + str(len(ImagesMalignant)))

# Barchar models

def BarCharModels(Parameters):

    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""
    
    Path = Parameters[0]
    Title = Parameters[1]
    XLabel = Parameters[2]
    Data = Parameters[3]
    Reverse = Parameters[4]
    Folder_Save = Parameters[5]
    Num_classes = Parameters[6]

    if Num_classes == 2:
        LabelClassName = 'Biclass_'
    elif Num_classes > 2:
        LabelClassName = 'Multiclass_'

    YFast = []
    Yslow = []

    XFast = []
    Xslow = []

    XFastest = []
    YFastest = []

    XSlowest = []
    YSlowest = []

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")
  
    df = pd.DataFrame(Path)
  
    X = list(df.iloc[:, 0])
    Y = list(df.iloc[:, Data])

    plt.figure(figsize = (22, 24))

    if Reverse == True:

        for index, (k, i) in enumerate(zip(X, Y)):
            if i < np.mean(Y):
                XFast.append(k)
                YFast.append(i)
            elif i >= np.mean(Y):
                Xslow.append(k)
                Yslow.append(i)

        for index, (k, i) in enumerate(zip(XFast, YFast)):
            if i == np.min(YFast):
                XFastest.append(k)
                YFastest.append(i)
                print(XFastest)
                print(YFastest)

        for index, (k, i) in enumerate(zip(Xslow, Yslow)):
            if i == np.max(Yslow):
                XSlowest.append(k)
                YSlowest.append(i)

    elif Reverse == False:

        for index, (k, i) in enumerate(zip(X, Y)):
            if i < np.mean(Y):
                Xslow.append(k)
                Yslow.append(i)
            elif i >= np.mean(Y):
                XFast.append(k)
                YFast.append(i)

        for index, (k, i) in enumerate(zip(XFast, YFast)):
            if i == np.max(YFast):
                XFastest.append(k)
                YFastest.append(i)
                print(XFastest)
                print(YFastest)

        for index, (k, i) in enumerate(zip(Xslow, Yslow)):
            if i == np.min(Yslow):
                XSlowest.append(k)
                YSlowest.append(i)

                
# Plot the data using bar() method
    plt.barh(Xslow, Yslow, label = "Bad", color = 'gray')
    plt.barh(XSlowest, YSlowest, label = "Worse", color = 'black')
    plt.barh(XFast, YFast, label = "Better", color = 'lightcoral')
    plt.barh(XFastest, YFastest, label = "Best", color = 'red')

    #print(Xslow)
    #print(Yslow)

    #print(XFast)
    #print(YFast)
    #Postion = len(Y) - len(Yslow)

    for index, value in enumerate(YSlowest):
        plt.text(0, 68,
            'Worse value: ' + str(value) + ' -------> ' + str(XSlowest[0]), fontweight = 'bold', fontsize = 25)

    for index, value in enumerate(YFastest):
        plt.text(0, 70,
            'Best value: ' + str(value) + ' -------> ' + str(XFastest[0]), fontweight = 'bold', fontsize = 25)

    plt.legend(fontsize = 25)

    plt.title(Title, fontsize = 40)
    plt.xlabel(XLabel, fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.ylabel("Models", fontsize = 25)
    plt.yticks(fontsize = 15)
    plt.grid(color = 'gray', linestyle = '-', linewidth = 0.2)

    dst = LabelClassName + Title + '.png'
    dstPath = os.path.join(Folder_Save, dst)

    #Show the plot
    #plt.show()

    plt.savefig(dstPath)