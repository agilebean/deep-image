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

from keras.applications import vgg16, vgg19, resnet50, inception_v3, mobilenet, xception, densenet, inception_resnet_v2,  nasnet
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

def get_model_data(model):
    print('***** model name ' + model.name)
    if (model.name == 'vgg16'):
        model_preprocess_input = vgg16.preprocess_input
        model_decode_predictions = vgg16.decode_predictions
        image_size = 224
        top1_acc = '71.5'
        top5_acc = '90.1'
    elif (model.name == 'vgg19'):
        model_preprocess_input = vgg19.preprocess_input
        model_decode_predictions = vgg19.decode_predictions
        image_size = 224
        top1_acc = '72.7'
        top5_acc = '91.0'
    elif (model.name == 'resnet50'):
        model_preprocess_input = resnet50.preprocess_input
        model_decode_predictions = resnet50.decode_predictions
        image_size = 224
        top1_acc = '75.9'
        top5_acc = '92.9'
    elif (model.name == 'inception_v3'):
        model_preprocess_input = inception_v3.preprocess_input
        model_decode_predictions = inception_v3.decode_predictions
        image_size = 299
        top1_acc = '78.8'
        top5_acc = '94.4'
    elif (model.name == 'xception'):
        model_preprocess_input = xception.preprocess_input
        model_decode_predictions = xception.decode_predictions
        image_size = 299
        top1_acc = '79.0'
        top5_acc = '94.5'
    elif (model.name == 'mobilenet_1.00_224'):
        #model.name = 'mobilenet'
        model_preprocess_input = mobilenet.preprocess_input
        model_decode_predictions = mobilenet.decode_predictions
        image_size = 224
        top1_acc = '70.6' # Source: https://arxiv.org/pdf/1704.04861.pdf
        top5_acc = '87.1'
    elif (model.name == 'densenet201'):
        model_preprocess_input = densenet.preprocess_input
        model_decode_predictions = densenet.decode_predictions
        image_size = 224
        top1_acc = '77.0'
        top5_acc = '93.3'
    elif (model.name == 'NASNet'):
        model_preprocess_input = inception_resnet_v2.preprocess_input
        model_decode_predictions = inception_resnet_v2.decode_predictions
        image_size = 224
        top1_acc = '74.0' # Source: https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet
        top5_acc = '91.6' # Source: dito
    elif (model.name == 'inception_resnet_v2'):
        model_preprocess_input = inception_resnet_v2.preprocess_input
        model_decode_predictions = inception_resnet_v2.decode_predictions
        image_size = 299
        top1_acc = '80.1' # 80.4
        top5_acc = '95.1' # 95.3
    elif (model.name == 'mobilenetv2_1.00_224'):
        model_preprocess_input = inception_resnet_v2.preprocess_input
        model_decode_predictions = inception_resnet_v2.decode_predictions
        image_size = 224
        top1_acc = '71.8' #Source: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
        top5_acc = '91.0' # Source: dito
    
    return (top1_acc, top5_acc, model_preprocess_input, model_decode_predictions, image_size)

def predict_image(image_path, model) -> str:
    tic = time()
    model_name = model.name
    print("*** entered predict_image() by " + model_name)
    
    ( top1_acc, top5_acc,
    model_preprocess_input, 
    model_decode_predictions, 
    image_size ) = get_model_data(model)

    image_input = image.load_img(image_path, target_size = (image_size, image_size))

    # read image input
    img = image.img_to_array(image_input)
    # reshape data for the model
    img = np.expand_dims(img, axis=0)
    # prepare the image for the models

    if (model_name == 'resnet50'):
        img = model_preprocess_input(img, mode='caffe')
    else:
        img = model_preprocess_input(img)
    print("image preprocessed")

    # predict the probability across all output classes
    predictions = model.predict(img)
    print("image predicted")

    # convert the probabilities to class labels
    label = model_decode_predictions(predictions, top=1)
    print("predictions decoded")

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    toc = time()
    processing_time = toc-tic
    print("Image classified by %s in >> %.2f s" % (model_name, processing_time))

    classification = ( model_name, top1_acc, top5_acc, processing_time, label[2]*100, label[1] )
    print("image classified >> " + str(classification))

    return classification


def upload_image(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(image_path)
    return image_path


@app.route('/', methods = ['GET'])
def show_index():
    print('entered homepage')
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def generate_predictions():
    print('entered predict image')

    if request.method == 'POST':
        tic = time()
        image_path = upload_image(request)

        result = '<table style="width:70%">'
        result += '<caption>Image Classifications for %s </caption>' % (os.path.basename(image_path))
        result += '<tr><th>Machine Learning Model</th><th>Top1-Acc</th><th>Top5-Acc</th><th>Time</th><th>Confidence</th><th>Classification</th></tr>'

        content_list = [ predict_image(image_path, model) for model in mymodels.models ]
        
        html_content_list = [ '<tr><td> %s </td><td>%s%%</td><td>%s%%</td><td> %.2fs </td><td> %.2f%% </td><td> %s </td></tr>' % (model_name, top1_acc, top5_acc, ptime, confidence, label)
            for (model_name, top1_acc, top5_acc, ptime, confidence, label) in content_list ]
        
        result += '\n'.join(html_content_list)

        result += '</table>'

        toc = time()
        print("Total time for all predictions >> %.2f s" % (toc-tic) )

        return result
    return None

if (__name__ == '__main__'):
    print('* Loading Keras models and starting Flask server...')
    print('.....')
    # get wsgi server to replace flask app.run()
    #from gevent.pywsgi import WSGIServer
    #web_server = WSGIServer(('', 5000), app)
    #web_server.serve_forever()
    app.run(host='0.0.0.0')

    print('Success! Server available at http://127.0.0.1:5000')


