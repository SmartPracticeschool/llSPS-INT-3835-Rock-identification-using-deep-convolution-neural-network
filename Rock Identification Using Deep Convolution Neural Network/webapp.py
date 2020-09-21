# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:09:42 2020

@author: lavke
"""


from __future__ import division , print_function
import sys
import os 
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model
from keras import backend
from tensorflow.keras import backend


import tensorflow as tf
global graph
graph = tf.get_default_graph()
from skimage.transform import resize
from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'cnn_model.h5'
model = load_model(MODEL_PATH)

@app.route('/',methods = ["GET"] )
def index():
    return render_template("base.html")

@app.route('/predict',methods = ["GET","POST"])
def upload():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        print("current path: ",basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        f.save(file_path)
        print("joined path: ",file_path)
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
       
        with graph.as_default():
            prediction_class = model.predict_classes(x)
            print("prediction",prediction_class)
        
        i=prediction_class.flatten()
        index = ['igneous','metamorphic','sedimentary']
        if(str(index[i[0]])=="igneous"):
            text="It is an " + str(index[i[0]] +" rock")
        elif (str(index[i[0]])=="metamorphic"):
            text ="It is a " + str(index[i[0]] +" rock")
        elif(str(index[i[0]])=="sedimentary"):
            text="It is a " + str(index[i[0]] +" rock")
    return text 
if __name__ == "__main__":
    app.run(debug = False ,threaded= False)
