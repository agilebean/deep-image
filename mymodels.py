import os
from keras.models import load_model
from time import time

# get cwd from this file's location
current_dir = os.path.dirname(os.path.abspath(__file__))

ticTotal = time()

def load_keras_model(model_label):
    
    tic = time()
    model = load_model(os.path.join(current_dir, 'models', model_label, '.h5'))
    toc = time()
    print("%s model loaded in %.2fs" % (model_label, toc-tic))
    return model

ticTotal = time()

model_mobilenet = load_model(os.path.join(current_dir, 'models/MobileNet.h5'))
toc = time()
#model_mobilenet = MobileNet(weights = 'imagenet', include_top = True)
print("MobileNet model loaded in %.2fs" % (toc-tic))


tic = time()
model_resnet50 = load_model(os.path.join(current_dir, 'models/ResNet50.h5'))
toc = time()
#model_resnet50 = ResNet50(weights = 'caffe', include_top = True)
print("ResNet50 model loaded in %.2fs" % (toc-tic))


tic = time()
model_inception_v3 = load_model(os.path.join(current_dir, 'models/InceptionV3.h5'))
toc = time()
#model_inception_v3 = InceptionV3(weights = 'imagenet', include_top = True)
print("InceptionV3 model loaded in %.2fs" % (toc-tic))


tic = time()
model_xception = load_model(os.path.join(current_dir, 'models/Xception.h5'))
toc = time()
#model_inception_v3 = InceptionV3(weights = 'imagenet', include_top = True)
print("Xception model loaded in %.2fs" % (toc-tic))


tic = time()
model_vgg16 = load_model(os.path.join(current_dir, 'models/VGG16.h5'))
toc = time()
print("VGG16 model loaded in %.2fs" % (toc-tic))

tic = time()
model_vgg19 = load_model(os.path.join(current_dir, 'models/VGG19.h5'))
toc = time()
print("VGG19 model loaded in %.2fs" % (toc-tic))


tocTotal = time()
print("*** All models loaded in %.2fs ***" % (tocTotal-ticTotal))
