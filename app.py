from keras import backend
import tensorflow as tf

session = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True
    ))

backend.set_session(session)

import pdb
from time import time
from flask import Flask, render_template, request

from keras.applications import resnet50, inception_v3, mobilenet, xception, vgg16, vgg19
from keras.preprocessing import image

import numpy as np
import os
# ERROR Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file processing
from werkzeug.utils import secure_filename

# import global variables (model_mobilenet, model_resnet50, model_inception_v3)
import mymodels

# create Flask app
app = Flask(__name__)

def predict_image(image_input, model) -> str:
    tic = time()
    print("predict_image() entered")

    if (model.name == 'mobilenet_1.00_224'):
        #model.name = 'mobilenet'
        model_preprocess_input = mobilenet.preprocess_input
        model_decode_predictions = mobilenet.decode_predictions
    elif (model.name == 'resnet50'):
        model_preprocess_input = resnet50.preprocess_input
        model_decode_predictions = resnet50.decode_predictions
    elif (model.name == 'inception_v3'):
        model_preprocess_input = inception_v3.preprocess_input
        model_decode_predictions = inception_v3.decode_predictions
    elif (model.name == 'xception'):
        model_preprocess_input = xception.preprocess_input
        model_decode_predictions = xception.decode_predictions
    elif (model.name == 'vgg16'):
        model_preprocess_input = vgg16.preprocess_input
        model_decode_predictions = vgg16.decode_predictions
    elif (model.name == 'vgg19'):
        model_preprocess_input = vgg19.preprocess_input
        model_decode_predictions = vgg19.decode_predictions

    # read image input
    img = image.img_to_array(image_input)
    # reshape data for the model
    img = np.expand_dims(img, axis=0)
    # prepare the image for the models

    if (model.name == 'resnet50'):
        img = model_preprocess_input(img, mode='caffe')
    else:
        img = model_preprocess_input(img)

    print("image preprocessed")
    #pdb.set_trace()

    # predict the probability across all output classes
    predictions = model.predict(img)

    print("image predicted")
    #pdb.set_trace()

    # convert the probabilities to class labels
    label = model_decode_predictions(predictions, top=1)

    print("predictions decoded")

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    #classification = '%s predicts: %s (%.2f%%)' % (model.name, label[1], label[2]*100)

    toc = time()

    processing_time = toc-tic
    print("Image classified by %s in >> %.2f s" % (model.name, processing_time))

    classification = ( model.name, processing_time, label[2]*100, label[1] )
    print("image classified >> " + str(classification))

    return (classification)


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
    print('entered homepage')
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def generate_predictions():
    print('entered predict image')

    def row_prediction(image_input, model):
        model_name, ptime, confidence, label = predict_image(image_input, model)
        return '<tr><td> %s </td><td> %.2fs </td><td> %.2f%% </td><td> %s </td></tr>' % (model_name, ptime, confidence, label)

    if request.method == 'POST':
        tic = time()
        file_path = get_file_path_and_save(request)
        img_224 = image.load_img(file_path, target_size = (224,224))
        img_299 = image.load_img(file_path, target_size = (299,299))

        result = '<table style="width:70%">'
        result += '<caption>Image Classifications for %s </caption>' % (os.path.basename(file_path))
        result += '<tr><th>Machine Learning Model</th><th>Time</th><th>Confidence</th><th>Classification</th></tr>'


        result += row_prediction(img_224, mymodels.model_vgg16)
        
        result += row_prediction(img_224, mymodels.model_vgg19)

        result += row_prediction(img_224, mymodels.model_resnet50)

        result += row_prediction(img_299, mymodels.model_inception_v3)

        result += row_prediction(img_299, mymodels.model_xception)
        
        result += row_prediction(img_224, mymodels.model_mobilenet)
        

        result += '</table>'

        toc = time()
        print("Total time for all predictions >> %.2f s" % (toc-tic) )

        return result
    return None

if (__name__ == '__main__'):
    print('* Loading Keras models and starting Flask server...')
    print('.....')
    # get wsgi server to replace flask app.run()
    from gevent.pywsgi import WSGIServer
    web_server = WSGIServer(('', 5000), app)
    web_server.serve_forever()

    print('Success! Server available at http://127.0.0.1:5000')


