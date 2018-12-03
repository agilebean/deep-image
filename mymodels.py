import os
from keras.models import load_model
from time import time

# get cwd from this file's location
current_dir = os.path.dirname(os.path.abspath(__file__))

ticTotal = time()

def load_keras_model(model_label):
    
    tic = time()
    model = load_model(os.path.join(current_dir, 'models', model_label + '.h5'))
    toc = time()
    print("%s model loaded in %.2fs" % (model_label, toc-tic))
    return model

model_labels = [ 'VGG16',
                 'VGG19'#,
                 #'ResNet50',
                 #'InceptionV3',
                 #'Xception',
                 #'MobileNet',
                 #'MobileNetV2',
                 #'NASNetMobile',
                 #'DenseNet201',
                 #'InceptionResNetV2'
               ]

models = [ load_keras_model(model_label) for model_label in model_labels ]


tocTotal = time()
print("*** All models loaded in %.2fs ***" % (tocTotal-ticTotal))
