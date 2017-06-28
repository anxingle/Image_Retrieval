# coding: utf-8
#python search.py -i dataset/train/ukbench00000.jpg
import argparse  
import cv2
import imutils
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

# from pylab import *
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
# from matplotlib import pylab
# matplotlib.use('Qt4Agg')
from PIL import Image
# from rootsift import RootSIFT

# Get the path of the training set
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required = "True")
    args = vars(parser.parse_args())
    return args

def extract(path):
    
    # 加载特征库， 数据集路径， 词频， 单词量， 聚类中心 
    im_features, image_paths, idf, numWords, centroid = joblib.load("../data/bof.pkl")
    
    # opencv特征提取器 
    fea_det = cv2.xfeatures2d.SIFT_create()

    # 列表： 路径，特征 
    des_list = []
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kpts, des = fea_det.detectAndCompute(gray, None)

    des_list.append((image_path, des))

    # 仅含有特征 
    descriptors = des_list[0][1]

    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors, centroid)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features*idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)
    return im, image_paths, rank_ID

def draw_result(im, image_paths, rank_ID):
    plt.figure()
    plt.gray()
    plt.subplot(5,4,1)
    plt.imshow(im[:,:,::-1])
    plt.axis('off')
    for i, ID in enumerate(rank_ID[0][0:16]):
	    img = Image.open(image_paths[ID])
	    plt.gray()
	    plt.subplot(5,4,i+5)
	    plt.imshow(img)
	    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    args = args_parser()
    # 得到检索图片的路径 
    image_path = args["image"]
    img, train_paths, rank = extract(image_path)
    draw_result(img, train_paths, rank)
