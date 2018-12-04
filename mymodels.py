from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications import mobilenet_v2
from keras.applications import vgg16, vgg19, resnet50, inception_v3, xception, mobilenet, nasnet, densenet, inception_resnet_v2
from keras.models import load_model, Model

import os
from time import time

# get cwd from this file's location
current_dir = os.path.dirname(os.path.abspath(__file__))

Tic = time()

model_data =    {
                 'VGG16': {'image_size' : 224,'top1_acc' : 71.5, 'top5_acc' : 90.1,
                           'model_preprocess_input': vgg16.preprocess_input,
                           'model_decode_predictions': vgg16.decode_predictions
                           },
                  'VGG19': {'image_size' : 224, 'top1_acc' : 72.7, 'top5_acc' : 91.0,
                            'model_preprocess_input': vgg19.preprocess_input,
                            'model_decode_predictions': vgg19.decode_predictions
                            },
                  'ResNet50': {'image_size' : 224, 'top1_acc' : 75.9, 'top5_acc' : 92.9,
                               'model_preprocess_input': resnet50.preprocess_input,
                               'model_decode_predictions': resnet50.decode_predictions
                               },
                  'InceptionV3': {'image_size' : 299, 'top1_acc' : 78.8, 'top5_acc' : 94.4,
                                  'model_preprocess_input': inception_v3.preprocess_input,
                                  'model_decode_predictions': inception_v3.decode_predictions
                                  },
                  'Xception': {'image_size' : 299, 'top1_acc' : 79.0, 'top5_acc' : 94.5,
                               'model_preprocess_input': xception.preprocess_input,
                               'model_decode_predictions': xception.decode_predictions
                               },
                  # MobileNet Source: https://arxiv.org/pdf/1704.04861.pdf
                  'MobileNet': {'image_size' : 224, 'top1_acc' : 70.6, 'top5_acc' : 87.1,
                                'model_preprocess_input': mobilenet.preprocess_input,
                                'model_decode_predictions': mobilenet.decode_predictions
                                },
                  # MobileNetV2 Source: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
                  'MobileNetV2': {'image_size' : 224, 'top1_acc' : 71.8, 'top5_acc' : 91.0,
                                  'model_preprocess_input': mobilenet_v2.preprocess_input,
                                  'model_decode_predictions': mobilenet_v2.decode_predictions
                                  },
                  # NasNetMobile Source: https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet
                  'NASNetMobile': {'image_size' : 224, 'top1_acc' : 74.0, 'top5_acc' : 91.6,
                                   'model_preprocess_input': nasnet.preprocess_input,
                                   'model_decode_predictions': nasnet.decode_predictions
                                   },
                  'DenseNet201': {'image_size' : 224, 'top1_acc' : 77.0, 'top5_acc' : 93.3,
                                  'model_preprocess_input': densenet.preprocess_input,
                                  'model_decode_predictions': densenet.decode_predictions
                                  },
                  'InceptionResNetV2': {'image_size' : 299, 'top1_acc' : 80.1, 'top5_acc' : 95.1,
                                        'model_preprocess_input': inception_resnet_v2.preprocess_input,
                                        'model_decode_predictions': inception_resnet_v2.decode_predictions
                                        }
                 }

def load_keras_model(model_label):
    tic = time()
    model = load_model(os.path.join(current_dir, 'models', model_label + '.h5'))
    toc = time()
    print("%s model loaded in %.2fs" % (model_label, toc-tic))

    model._make_predict_function()
    print("%s model _make_predict_function() called" % (model_label))

    assert isinstance(model, Model)

    return model


# Append model_instance into model_data dict of dicts:
def append_model_instances(models_dict, model_data):
    # for each model_label and its model_instance
    for model_label, model_instance in models_dict.items():
        # if the same model_label is in model_data dict
        if model_label in model_data:
            print('key: '+ model_label)
            print('value: ' + str(model_instance))
            # insert the model_instance into the model_data dict by model_data key
            model_data[model_label].update(model_instance = model_instance)
    return (model_data)

print('*** Loading Keras models...')
models = { model_label : load_keras_model(model_label) for model_label in model_data }

model_data = append_model_instances(models, model_data)


print(model_data.keys())
# print(model_data.values())

Toc = time()
print('****************************************************')
print("*** All models loaded in %.2fs ***" % (Toc-Tic))

