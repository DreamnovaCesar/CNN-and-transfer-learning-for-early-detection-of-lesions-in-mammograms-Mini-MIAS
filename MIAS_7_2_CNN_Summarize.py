# summarize the models

from keras.utils.vis_utils import plot_model

from MIAS_7_1_CNN_Architectures import ResNet50_PreTrained
from MIAS_7_1_CNN_Architectures import ResNet50V2_PreTrained
from MIAS_7_1_CNN_Architectures import ResNet152_PreTrained
from MIAS_7_1_CNN_Architectures import ResNet152V2_PreTrained

from MIAS_7_1_CNN_Architectures import MobileNet_Pretrained
from MIAS_7_1_CNN_Architectures import MobileNetV3Small_Pretrained
from MIAS_7_1_CNN_Architectures import MobileNetV3Large_Pretrained

from MIAS_7_1_CNN_Architectures import Xception_Pretrained

from MIAS_7_1_CNN_Architectures import VGG16_PreTrained
from MIAS_7_1_CNN_Architectures import VGG19_PreTrained

from MIAS_7_1_CNN_Architectures import InceptionV3_PreTrained

from MIAS_7_1_CNN_Architectures import DenseNet121_PreTrained
from MIAS_7_1_CNN_Architectures import DenseNet201_PreTrained

from MIAS_7_1_CNN_Architectures import CustomCNNAlexNet12_Model

from MIAS_8_Preprocessing_5_Resize import XsizeResized
from MIAS_8_Preprocessing_5_Resize import YsizeResized

ResNet50Model = ResNet50_PreTrained(XsizeResized, YsizeResized)
ResNet50V2Model = ResNet50V2_PreTrained(XsizeResized, YsizeResized)
ResNet152Model = ResNet152_PreTrained(XsizeResized, YsizeResized)
ResNet152V2Model = ResNet152V2_PreTrained(XsizeResized, YsizeResized)

MobileNetModel = MobileNet_Pretrained(XsizeResized, YsizeResized)
MobileNetV3SmallModel = MobileNetV3Small_Pretrained(XsizeResized, YsizeResized)
MobileNetV3LargeModel = MobileNetV3Large_Pretrained(XsizeResized, YsizeResized)

XceptionModel = Xception_Pretrained(XsizeResized, YsizeResized)

VGG16Model = VGG16_PreTrained(XsizeResized, YsizeResized)
VGG19Model = VGG19_PreTrained(XsizeResized, YsizeResized)

InceptionV3Model = InceptionV3_PreTrained(XsizeResized, YsizeResized)

DenseNet121Model = DenseNet121_PreTrained(XsizeResized, YsizeResized)
DenseNet201Model = DenseNet201_PreTrained(XsizeResized, YsizeResized)

CustomCNNAlexNet12_Model = CustomCNNAlexNet12_Model()

plot_model(ResNet50Model, to_file = 'ResNet50Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(ResNet50V2Model, to_file = 'ResNet50V2Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(ResNet152Model, to_file = 'ResNet152Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(ResNet152V2Model, to_file = 'ResNet152V2Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(MobileNetV3SmallModel, to_file = 'MobileNetV3SmallModel_plot.png', show_shapes = True, show_layer_names = True)

plot_model(MobileNetV3LargeModel, to_file = 'MobileNetV3LargeModel_plot.png', show_shapes = True, show_layer_names = True)

plot_model(XceptionModel, to_file = 'XceptionModel_plot.png', show_shapes = True, show_layer_names = True)

plot_model(VGG16Model, to_file = 'VGG16Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(ResNet50Model, to_file = 'ResNet50Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(ResNet50Model, to_file = 'ResNet50Model_plot.png', show_shapes = True, show_layer_names = True)

plot_model(CustomCNNAlexNet12_Model, to_file = 'CustomCNNAlexNet12_Model_plot.png', show_shapes = True, show_layer_names = True)
