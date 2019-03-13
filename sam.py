from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask import jsonify
from collections import Counter

import argparse
import json
import numpy as np
import tensorflow as tf
import cv2
import math
import os
import shutil

app = Flask(__name__)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    f = request.files['file']
    f.save(secure_filename(f.filename))
  graph = load_graph(model_file)

  flag=0
  mylist=list()
  resultt = []
  name1 = ""

  if(f.filename.endswith(".mp4")):
    name = f.filename.split('.')
    name1 = name[0]
    os.mkdir(name1)
    cap = cv2.VideoCapture(f.filename)
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    frameRate = math.floor(frameRate)*20
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (math.floor(frameRate)!=0 and frameId % math.floor(frameRate) == 0):
            cv2.imwrite(name1+"/frame%d.jpg" % frameId, frame)
            mylist.append(name1+"/frame%d.jpg" % frameId)
            flag +=1
    cap.release()
    flag-=1
  else:
    mylist.append(f.filename)

  htag=[]
  htag1=[]

  while flag>=0:
    file_name = mylist[flag]
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    mainset=["birthday","food","party","vacation","selfi"]

    with open("datasetscore.txt") as f:
      dsetscore = f.readlines()
    dsetscore = [x.strip().split(',') for x in dsetscore]

    with open("dataset.txt") as f:
      dset = f.readlines()
    dset = [x.strip().split(',') for x in dset]

    dseti=[[0 for i in range(len(dset[j]))] for j in range(len(dset))]

    with open("hashtag.txt") as f:
      tag = f.readlines()
    tag = [x.strip().split(',') for x in tag]

    print("FRAME :",flag+1)
    for i in top_k:
      #print(labels[i],results[i])
      if results[i]>=0.01:
        for j in range(len(dset)):
          for k in range(len(dset[j])):
            if labels[i]==dset[j][k]:
              dseti[j][k]=int(dsetscore[j][k])
    hashtag=[]
    hashtag1=[]
    summ=[]
    print(dseti)
    for i in range(len(dseti)):
      if sum(dseti[i])/100 > 0.3:
        summ.append(sum(dseti[i]))
        hashtag1.append(tag[i])
        hashtag.append(mainset[i])

    hashtag1=[x for _,x in sorted(zip(summ,hashtag1),reverse=True)]
    hashtag=[x for _,x in sorted(zip(summ,hashtag),reverse=True)]

    #print(hashtag)
    #print(hashtag1)

    htag+=hashtag
    htag1+=hashtag1

    flag-=1

  if name1:
    shutil.rmtree(name1)

  htag1=[tuple(i) for i in htag1]

  htag=Counter(htag)
  htag1=Counter(htag1)

  sorted(htag, key=htag.get, reverse=True)
  sorted(htag1, key=htag1.get, reverse=True)

  htag=list(htag)
  htag1=[list(i) for i in htag1]

  print(htag)
  return render_template("result.html",result = htag,result1 = htag1)

if __name__ == '__main__':
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  app.run(debug = True)