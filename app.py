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


def upload_image(request):
    # Get the file from post request
    file = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
    file.save(image_path)

    return image_path


@app.route('/', methods = ['GET'])
def show_index():
    print('*** entered homepage')
    return render_template('index.html')


def predict_image(image_path, model_label, model_dict) -> str:
    tic = time()
    print("*** entered predict_image() for " + model_label)

    # Get model-specific metadata and functions from model dictionary
    image_size = model_dict['image_size']
    top1_acc = model_dict['top1_acc']
    top5_acc = model_dict['top5_acc']
    model_preprocess_input = model_dict['model_preprocess_input']
    model_decode_predictions = model_dict['model_decode_predictions']

    model = model_dict['model_instance']
    model_name = model.name

    image_input = image.load_img(image_path, target_size=(image_size, image_size))

    # read image input
    img = image.img_to_array(image_input)

    # reshape data for the model
    img = np.expand_dims(img, axis=0)

    # prepare the image for the models
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

    # get classification confidence (%)
    confidence = '%.2f' % (label[2] * 100)

    toc = time()
    processing_time = '%.2fs' % (toc-tic)
    print("Image classified by %s in >> %s" % (model_name, processing_time))

    classification = (model_name, top1_acc, top5_acc, processing_time, confidence, label[1])
    print("image classified >> " + str(classification))

    return classification


@app.route('/predict', methods = ['GET', 'POST'])
def generate_classifications_output():
    print('*entered generate_classifications_output()')

    if request.method == 'POST':
        tic = time()
        image_path = upload_image(request)

        # Create html table for all classifications
        output = '<table style="width:70%">'
        output += '<caption>Image Classifications for %s </caption>' % (os.path.basename(image_path))
        output += '<tr><th>Machine Learning Model</th><th>Top1-Acc</th><th>Top5-Acc</th><th>Time</th><th>Confidence</th><th>Classification</th></tr>'

        # feed predict_image(model_name, top1_acc, top5_acc, ptime, confidence, label) directly into html
        html_content_list = [
            '<tr><td> %s </td><td>%s%%</td><td>%s%%</td><td> %ss </td><td> %s%% </td><td> %s </td></tr>'
            % predict_image(image_path, key, values)
            for key, values in mymodels.model_data.items() ]

        output += '\n'.join(html_content_list)
        output += '</table>'

        toc = time()
        print("Total time for all predictions >> %.2f s" % (toc-tic) )

        return output
    return None

if (__name__ == '__main__'):

    print('*** Starting WSGI Server...')
    print('****************************************************')
    print('*** Server is available at http://127.0.0.1:5000')
    print('****************************************************')

    # Get wsgi server to replace flask app.run()
    from gevent.pywsgi import WSGIServer
    web_server = WSGIServer(('', 5000), app)
    web_server.serve_forever()



