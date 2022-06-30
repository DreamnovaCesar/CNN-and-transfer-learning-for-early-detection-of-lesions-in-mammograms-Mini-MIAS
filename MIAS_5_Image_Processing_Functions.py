import os
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import normalized_mutual_information as nmi

from skimage import img_as_ubyte
from skimage import io
from skimage import filters

from skimage.exposure import equalize_adapthist
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity

from skimage.filters import unsharp_mask

from MIAS_3_General_Functions import Removeallfiles
from MIAS_3_General_Functions import ShowSort

# Normalization

def NormalizeData(data):  

      """
	    Normalize the images' value from 0-255 to 0-1

      Parameters:
      argument1 (Int): Image chosen.

      Returns:
	    float:Returning normalized value

   	  """

      return (data - np.min(data)) / (np.max(data) - np.min(data))

def TransformationImageIntFloat(File, Folder_Path, filename, Images, Count, Ext): 

    """
	  Normalize mutiple images.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Str): filename of the image.
    argument4 (Int): The number of the images.
    argument5 (Int): Count of each interation.
    argument6 (Str): Extension chosen.

    Returns:
	  int:Returning normalize image
    int:Returning normalize image filename

   	"""

    print(f"Working with {Count} of {Images} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    Normalization_Imagen = NormalizeData(Imagen)

    FilenamesREFNUM = filename

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(Path_File, dst)
    
    io.imsave(dstPath, Normalization_Imagen)

    Image = Normalization_Imagen
    Refnum = FilenamesREFNUM

    return Image, Refnum

def Intfloat(Folder_Path):

    """
	  Normalize mutiple images, general function.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	  Void

    """

    #Removeallfiles(New_Folder_Path)

    png = ".png"

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files

        Image, Refnum = TransformationImageIntFloat(File, Folder_Path, filename, images, Count, png) 
        Count += 1

# Normalization

def TransformationImageNormalization(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Xsize, Ysize, Ext): 

    """
	  Transform each image using normalization and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (Str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument6 (Str): Severity of each image.
    argument7 (Int): Division for median filter.
    argument8 (Str): Extension chosen.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = cv2.imread(Path_File)

    Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

    Norm_img = np.zeros((Xsize, Ysize))
    Normalization_Imagen = cv2.normalize(Imagen, Norm_img, 0, 255, cv2.NORM_MINMAX)

    #Normalization_Imagen = NormalizeData(Normalization_Imagen)

    Mae = mae(Imagen, Normalization_Imagen)
    Mse = mse(Imagen, Normalization_Imagen)

    Ssim = ssim(Imagen, Normalization_Imagen)
    Psnr = psnr(Imagen, Normalization_Imagen)

    Nrmse = nrmse(Imagen, Normalization_Imagen)
    Nmi = nmi(Imagen, Normalization_Imagen)

    R2s = r2s(Imagen, Normalization_Imagen)

    FilenamesREFNUM = filename + '_Normalization'

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(New_Folder_Path, dst)
    
    cv2.imwrite(dstPath, Normalization_Imagen)

    Image = Normalization_Imagen
    Refnum = FilenamesREFNUM

    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum

def Normalization(Folder_Path, New_Folder_Path, Severity, Xsize, Ysize, Label):

  """
	Get the values from median filter images and save them into a dataframe.

  Parameters:
  argument1 (Folder): Folder chosen.
  argument2 (Folder): Folder destination for new images.
  argument3 (Str): Severity of each image.
  argument4 (Int): The number of the images.
  argument5 (Int): Division for median filter.
  argument6 (Str): Label for each image.

  Returns:
	int:Returning dataframe with all data.
    
  """

  Removeallfiles(New_Folder_Path)

  Images = [] # Mammograms (normal and abnormal).
  Labels = []
  Filename_ALL = []

  Mae_ALL = [] # MSE normal.
  Mse_ALL = [] # PSNR normal.

  Ssim_ALL = [] # MSE normal.
  Psnr_ALL = [] # PSNR normal.

  Nrmse_ALL = [] # MSE normal.
  Nmi_ALL = [] # PSNR normal.

  R2s_ALL = [] # PSNR normal.

  png = ".png"    # png.

  # Normals Images.

  os.chdir(Folder_Path)

  sorted_files, images = ShowSort(Folder_Path)
  Count = 1

  for File in sorted_files:

    filename, extension  = os.path.splitext(File)

    if File.endswith(png): # Read png files

      Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageNormalization(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, Xsize, Ysize, png) 
      Count += 1

      Images.append(Image)
      Labels.append(Label)
      Filename_ALL.append(Refnum)

      Mae_ALL.append(Mae)
      Mse_ALL.append(Mse)

      Ssim_ALL.append(Ssim)
      Psnr_ALL.append(Psnr)

      Nrmse_ALL.append(Nrmse)
      Nmi_ALL.append(Nmi)

      R2s_ALL.append(R2s)

  Dataset = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

  return Images, Dataset

# Median filter

def TransformationImageMedianFilter(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Division, Ext): 

    """
	  Transform each image using median filter and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (Str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument6 (Str): Severity of each image.
    argument7 (Int): Division for median filter.
    argument8 (Str): Extension chosen.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    #Image_Median_Filter = cv2.medianBlur(Imagen, Division)
    Image_Median_Filter = filters.median(Imagen, np.ones((Division, Division)))

    Mae = mae(Imagen, Image_Median_Filter)
    Mse = mse(Imagen, Image_Median_Filter)

    Ssim = ssim(Imagen, Image_Median_Filter)
    Psnr = psnr(Imagen, Image_Median_Filter)

    Nrmse = nrmse(Imagen, Image_Median_Filter)
    Nmi = nmi(Imagen, Image_Median_Filter)

    R2s = r2s(Imagen, Image_Median_Filter)

    FilenamesREFNUM = filename + '_Median_Filter'

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(New_Folder_Path, dst)
    io.imsave(dstPath, Image_Median_Filter)

    Image = Image_Median_Filter
    Refnum = FilenamesREFNUM

    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum

def MedianFilterNoise(Folder_Path, New_Folder_Path, Severity, Division, Label):

    """
	  Get the values from median filter images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (Str): Severity of each image.
    argument4 (Int): The number of the images.
    argument5 (Int): Division for median filter.
    argument6 (Str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    # Remove images of the folder chosen

    Removeallfiles(New_Folder_Path)
    
    # General lists

    Images = [] 
    Labels = []
    Filename_ALL = []

    # Statistic lists

    Mae_ALL = [] 
    Mse_ALL = [] 

    Ssim_ALL = [] 
    Psnr_ALL = [] 

    Nrmse_ALL = [] 
    Nmi_ALL = [] 

    R2s_ALL = [] 

    # Extension used

    png = ".png"    

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    # For each file sorted.

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files
        
        # Using median filter function

        Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageMedianFilter(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, Division, png)
        Count += 1

        # Saving values in the lists

        Images.append(Image)
        Labels.append(Label)
        Filename_ALL.append(Refnum)
 
        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)

        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)

        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)

        R2s_ALL.append(R2s)

    # Final dataframe

    DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

# CLAHE

def TransformationImageCLAHE(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Clip_limit, Ext): 

    """
	  Transform each image using CLAHE and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (Str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument7 (Str): Severity of each image.
    argument8 (float): clip limit value use to change CLAHE images.
    argument9 (Str): Extension chosen.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
    #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    #CLAHE_Imagen = CLAHE.apply(Imagen)
    CLAHE_Imagen = equalize_adapthist(Imagen, clip_limit = Clip_limit)

    Imagen = img_as_ubyte(Imagen)
    CLAHE_Imagen = img_as_ubyte(CLAHE_Imagen)

    Mae = mae(Imagen, CLAHE_Imagen)
    Mse = mse(Imagen, CLAHE_Imagen)

    Ssim = ssim(Imagen, CLAHE_Imagen)
    Psnr = psnr(Imagen, CLAHE_Imagen)

    Nrmse = nrmse(Imagen, CLAHE_Imagen)
    Nmi = nmi(Imagen, CLAHE_Imagen)

    R2s = r2s(Imagen, CLAHE_Imagen)

    FilenamesREFNUM = filename + '_CLAHE'
    print(FilenamesREFNUM)
    dst = str(FilenamesREFNUM) + Ext

    dstPath = os.path.join(New_Folder_Path, dst)
    io.imsave(dstPath, CLAHE_Imagen)

    Image = CLAHE_Imagen
    Refnum = FilenamesREFNUM

    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum
  
def CLAHE_Technique(Folder_Path, New_Folder_Path, Severity, Clip_limit, Label):

    """
	  Get the values from CLAHE images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (Str): Severity of each image.
    argument4 (Float): clip limit value use to change CLAHE images.
    argument5 (Str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    # Remove images of the folder chosen

    Removeallfiles(New_Folder_Path)
    
    # General lists

    Images = [] 
    Labels = []
    Filename_ALL = []

    # Statistic lists

    Mae_ALL = [] 
    Mse_ALL = [] 

    Ssim_ALL = [] 
    Psnr_ALL = [] 

    Nrmse_ALL = [] 
    Nmi_ALL = [] 

    R2s_ALL = [] 

    # Extension used

    png = ".png"    

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    # For each file sorted.

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png):
        
        # Using CLAHE function

        Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageCLAHE(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, Clip_limit, png)
        Count += 1

        # Saving values in the lists

        Images.append(Image)
        Labels.append(Label)
        Filename_ALL.append(Refnum)

        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)

        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)

        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)

        R2s_ALL.append(R2s)

    # Final dataframe

    DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

# Histogram equalization

def TransformationImageHE(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Ext): 

    """
	  Transform each image using histogram equalization and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (Str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument7 (Str): Severity of each image.
    argument8 (Str): Extension chosen.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    HE_Imagen = equalize_hist(Imagen)

    Imagen = img_as_ubyte(Imagen)
    HE_Imagen = img_as_ubyte(HE_Imagen)

    Mae = mae(Imagen, HE_Imagen)
    Mse = mse(Imagen, HE_Imagen)

    Ssim = ssim(Imagen, HE_Imagen)
    Psnr = psnr(Imagen, HE_Imagen)

    Nrmse = nrmse(Imagen, HE_Imagen)
    Nmi = nmi(Imagen, HE_Imagen)

    R2s = r2s(Imagen, HE_Imagen)

    FilenamesREFNUM = filename + '_HE'

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(New_Folder_Path, dst)
    io.imsave(dstPath, img_as_ubyte(HE_Imagen))

    Image = HE_Imagen
    Refnum = FilenamesREFNUM
    #Ssim, Psnr
    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum

def HE_Technique(Folder_Path, New_Folder_Path, Severity, Label):

    """
	  Get the values from histogram equalization images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (Str): Severity of each image.
    argument4 (Str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    # Remove images of the folder chosen

    Removeallfiles(New_Folder_Path)
    
    # General lists

    Images = [] 
    Labels = []
    Filename_ALL = []

    # Statistic lists

    Mae_ALL = [] 
    Mse_ALL = [] 

    Ssim_ALL = [] 
    Psnr_ALL = [] 

    Nrmse_ALL = [] 
    Nmi_ALL = [] 

    R2s_ALL = [] 

    # Extension used

    png = ".png"    

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    # For each file sorted.

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files

        # Using CLAHE function

        Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageHE(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, png)
        Count += 1

        # Saving values in the lists

        Images.append(Image)
        Labels.append(Label)
        Filename_ALL.append(Refnum)

        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)

        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)

        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)

        R2s_ALL.append(R2s)

    # Final dataframe

    DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame

# Unsharp mask

def TransformationImageUM(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Ext, Radius, Amount): 

    """
	  Transform each image using CLAHE and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument7 (str): Severity of each image.
    argument8 (str): Extension chosen.
    argument9 (float): Radius value use to change Unsharp mask images.
    argument10 (float): Amount value use to change Unsharp mask images.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    UM_Imagen = unsharp_mask(Imagen, radius = Radius, amount = Amount)

    Imagen = img_as_ubyte(Imagen)
    UM_Imagen = img_as_ubyte(UM_Imagen)

    Mae = mae(Imagen, UM_Imagen)
    Mse = mse(Imagen, UM_Imagen)

    Ssim = ssim(Imagen, UM_Imagen)
    Psnr = psnr(Imagen, UM_Imagen)

    Nrmse = nrmse(Imagen, UM_Imagen)
    Nmi = nmi(Imagen, UM_Imagen)

    R2s = r2s(Imagen, UM_Imagen)

    FilenamesREFNUM = filename + '_UM'

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(New_Folder_Path, dst)

    io.imsave(dstPath, img_as_ubyte(UM_Imagen))

    Image = UM_Imagen
    Refnum = FilenamesREFNUM

    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum

def UM_Technique(Folder_Path, New_Folder_Path, Severity, Radius, Amount, Label):

    """
	  Get the values from unsharp masking images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (str): Severity of each image.
    argument4 (float): Radius value use to change Unsharp mask images.
    argument5 (float): Amount value use to change Unsharp mask images.
    argument6 (str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    # Remove images of the folder chosen

    Removeallfiles(New_Folder_Path)
    
    # General lists

    Images = [] 
    Labels = []
    Filename_ALL = []

    # Statistic lists

    Mae_ALL = [] 
    Mse_ALL = [] 

    Ssim_ALL = [] 
    Psnr_ALL = [] 

    Nrmse_ALL = [] 
    Nmi_ALL = [] 

    R2s_ALL = [] 

    # Extension used

    png = ".png"    

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    # For each file sorted.

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): 
        
        # Using unsharp masking function

        Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageUM(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, png, Radius, Amount)
        Count += 1

        # Saving values in the lists

        Images.append(Image)
        Labels.append(Label)
        Filename_ALL.append(Refnum)

        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)

        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)

        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)

        R2s_ALL.append(R2s)

    # Final dataframe

    Dataframe = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe

# Contrast Stretching.

def TransformationImageCS(File, Folder_Path, New_Folder_Path, filename, Images, Count, Severity, Ext): 

    """
	  Transform each image using constrast streching and save them into a folder.

    Parameters:
    argument1 (Int): File chosen.
    argument2 (Folder): Folder chosen.
    argument3 (Folder): Folder destination for new images.
    argument4 (str): filename of the image.
    argument5 (Int): The number of the images.
    argument6 (Int): Count of each interation.
    argument7 (str): Severity of each image.
    argument8 (str): Extension chosen.

    Returns:
	  int:Returning median filter image
    float:Returning MAE value
    float:Returning MSE value
    float:Returning SSIM value
    float:Returning PSNR value
    float:Returning NRMSE value
    float:Returning NMI value
    float:Returning R2S value
    float:Returning Image's name

   	"""

    print(f"Working with {Count} of {Images} {Severity} images ✅")
    print(f"Working with {filename} ✅")

    Path_File = os.path.join(Folder_Path, File)
    Imagen = io.imread(Path_File, as_gray = True)

    p2, p98 = np.percentile(Imagen, (2, 98))
    CS_Imagen = rescale_intensity(Imagen, in_range = (p2, p98))

    Imagen = img_as_ubyte(Imagen)
    CS_Imagen = img_as_ubyte(CS_Imagen)

    Mae = mae(Imagen, CS_Imagen)
    Mse = mse(Imagen, CS_Imagen)

    Ssim = ssim(Imagen, CS_Imagen)
    Psnr = psnr(Imagen, CS_Imagen)

    Nrmse = nrmse(Imagen, CS_Imagen)
    Nmi = nmi(Imagen, CS_Imagen)

    R2s = r2s(Imagen, CS_Imagen)

    FilenamesREFNUM = filename + '_CS'

    dst = FilenamesREFNUM + Ext

    dstPath = os.path.join(New_Folder_Path, dst)

    io.imsave(dstPath, img_as_ubyte(CS_Imagen))

    Image = CS_Imagen
    Refnum = FilenamesREFNUM

    return Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum

def CS_Technique(Folder_Path, New_Folder_Path, Severity, Label):

    """
	  Get the values from constrast streching images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (str): Severity of each image.
    argument6 (str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    # Remove images of the folder chosen

    Removeallfiles(New_Folder_Path)
    
    # General lists

    Images = [] 
    Labels = []
    Filename_ALL = []

    # Statistic lists

    Mae_ALL = [] 
    Mse_ALL = [] 

    Ssim_ALL = [] 
    Psnr_ALL = [] 

    Nrmse_ALL = [] 
    Nmi_ALL = [] 

    R2s_ALL = [] 

    # Extension used

    png = ".png"    

    os.chdir(Folder_Path)

    sorted_files, images = ShowSort(Folder_Path)
    Count = 1

    # For each file sorted.

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files
        
        # Using unsharp masking function

        Image, Mae, Mse, Ssim, Psnr, Nrmse, Nmi, R2s, Refnum = TransformationImageCS(File, Folder_Path, New_Folder_Path, filename, images, Count, Severity, png)
        Count += 1

        # Saving values in the lists

        Images.append(Image)
        Labels.append(Label)
        Filename_ALL.append(Refnum)

        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)

        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)

        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)

        R2s_ALL.append(R2s)

    # Final dataframe

    DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return DataFrame