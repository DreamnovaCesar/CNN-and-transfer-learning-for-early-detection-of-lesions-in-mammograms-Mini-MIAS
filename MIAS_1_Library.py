import cv2
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
import albumentations as A
import matplotlib.pyplot as plt

from math import log10
from math import sqrt

from PIL import Image

from scipy import ndimage
from scipy import misc

import skimage.measure

from skimage import data, img_as_float
from skimage import exposure
from skimage.metrics import structural_similarity as ssim

from skimage.filters import unsharp_mask

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis, skew
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from skimage.feature import graycomatrix, graycoprops

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall

from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Ftrl
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from tensorflow.keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model

#from keras_tuner import HyperParameter, HyperParameters
#from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband