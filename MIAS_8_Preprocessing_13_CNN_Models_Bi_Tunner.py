import keras_tuner as kt
from tensorflow import keras

from MIAS_7_1_CNN_Architectures import CustomCNNAlexNet12Tunner_Model

hp = kt.HyperParameters()

model = CustomCNNAlexNet12Tunner_Model(224, 224, 2, hp)

tuner = kt.Hyperband(model,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')

