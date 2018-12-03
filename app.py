from flask import Flask, render_template, request

import keras
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions


from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
# ERROR Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys

# get wsgi server to replace flask app.run()
from gevent.pywsgi import WSGIServer
# file processing
from werkzeug.utils import secure_filename

app = Flask(__name__)


model_mobilenet = load_model('models/MobileNet.h5')
print("MobileNet model loaded.")

model_resnet50 = load_model('models/ResNet50.h5')
print("ResNet50 model loaded.")

model_inception_v3 = load_model('models/InceptionV3.h5')
print("InceptionV3 model loaded.")

def predict_dog(image_input, model) -> str:
    if (model.name == 'mobilenet_1.00_224'):
        model_preprocess_input = keras.applications.mobilenet.preprocess_input
        model_decode_predictions = keras.applications.mobilenet.decode_predictions
    elif (model.name == 'resnet50'):
        model_preprocess_input = keras.applications.resnet50.preprocess_input
        model_decode_predictions = keras.applications.resnet50.decode_predictions
    elif (model.name == 'inception_v3'):
        model_preprocess_input = keras.applications.inception_v3.preprocess_input
        model_decode_predictions = keras.applications.inception_v3.decode_predictions
    
    # read image input
    img = image.img_to_array(image_input)

    # reshape data for the model
    img = np.expand_dims(img, axis=0)

    # prepare the image for the VGG model
    img = model_preprocess_input(img)

    # predict the probability across all output classes
    predictions = model.predict(img)

    # convert the probabilities to class labels
    label = model_decode_predictions(predictions, top=1)

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    # print the classification
    classification = '%s predicts: %s (%.2f%%)' % (model.name, label[1], label[2]*100)
    
    #sys.stdout.write(classification)
    
    return (classification)

    # show dog image
    #plt.figure()
    #plt.imshow(image)

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path

@app.route('/', methods = ['GET'])
def show_index():
    return render_template('index.html')

@app.route('/predictResNet50', methods = ['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        img_224 = image.load_img(file_path, target_size = (224,224))
        img_299 = image.load_img(file_path, target_size = (299,299))
        
        result = predict_dog(img_224, model_mobilenet)
        result += '\n'
        result += predict_dog(img_224, model_resnet50)
        result += '\n'
        result += predict_dog(img_299, model_inception_v3)
        
        return result
    return None

def predictInceptionV3():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        img = image.load_img(file_path, target_size = (299,299))

        result = predict_dog(img, model)

        return result
    return None

def predictMobileNet():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        img = image.load_img(file_path, target_size = (224,224))

        result = predict_dog(img, model)

        return result
    return None

if (__name__ == '__main__'):
    print('* Loading Keras models and starting Flask server...')
    print('.....')
    # app.run()
    web_server = WSGIServer(('', 5000), app)
    web_server.serve_forever()



