import face_model
import argparse
import cv2
import sys
import numpy as np
import mxnet
import easydict
import json

parser = argparse.ArgumentParser(description='face model test')
# general

parser.add_argument('--image_size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/', help='path to load model.')
parser.add_argument('--ga_model', default='../models/model-r100-ii/', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')


args = easydict.EasyDict(
{
    "image_size": '112,112',
    "model": '../models/model-r100-ii/model, 0',
    "ga_model": '../models/gamodel-r50/model,0',
    "gpu": 0,
    "det": 0,
    "flip": 0,
    "threshold": 1.24
})
def val(input_image):
    model = face_model.FaceModel(args)
    img = model.get_input(input_image)
    f2 = model.get_feature(input_image)
    features = {}
    for i in range(len(f2)):
        features['feature'+str(i)] = str(f2[i])
    app_json = json.dumps(features)
    #print(app_json)
    return app_json

if __name__ == "__main__":
	val(sys.argv[1])
