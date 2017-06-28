# coding: utf-8
"""
python findFeatures.py -t dataset/train/
"""
import argparse as ap
import os
import numpy as np
import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import math

# Get the path of the training set
def args_parser():
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--training_set", default = "../data/train_data/", help="Path to Training Set", required="True")
    parser.add_argument("-n", "--numwords", default = "1000" ,help= "words of BOW!")
    parser.add_argument("-o", "--output", default = "../data/")
    args = vars(parser.parse_args())
    return args


def extract_sift_features(train_path):
    '''
    根据输入args中 trainingSet 和 numwords获取sift特征
    '''
    training_names = os.listdir(train_path)

    # 图片列表
    image_paths = []
    for training_name in training_names:
        image_path = os.path.join(train_path, training_name)
        image_paths += [image_path]

    # 获取SIFT特征提取器
    fea_det = cv2.xfeatures2d.SIFT_create()

    # list: 元素为 图片-> 特征
    des_list = []
    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        print("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))
        kp, des = fea_det.detectAndCompute(gray, None)
        des_list.append((image_path, des))

    # 整个特征列表
    valid_des = np.ones([len(des_list)])
    descriptors = des_list[0][1]
    print(descriptors.shape)
    start = 0
    for image_path, descriptor in des_list[1:]:
        start += 1
        try:
            print(image_path, "  ", descriptor.shape)
            descriptors = np.vstack((descriptors, descriptor))
        except Exception as e:
            valid_des[start] = 0
            print("Error:  \n  ".encode('utf-8'), image_path)
    
    return image_paths, descriptors, des_list, valid_des

def bow_trainer(image_paths, output, descriptors, des_list, num_words, valid_des):
    # K-means 聚类 
    print("Start k-means: %d words, %d key points" %(num_words, descriptors.shape[0]))
    centroids, variance = kmeans(descriptors, num_words, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), num_words), "float32")
    for i in range(len(image_paths)):
        if valid_des[i] == 1:
            words, distance = vq(des_list[i][1], centroids)
            # words, distance = vq(descriptors[i], centroids)
            for w in words:
                im_features[i][w] += 1

    # TF-IDF 表示 
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    joblib.dump((im_features, image_paths, idf, num_words, centroids), "../data/bof.pkl", compress=3)

if __name__ == "__main__":
    args = args_parser()
    train_path = args["training_set"]
    num_words = int(args["numwords"])
    output = args["output"]
    image_paths, descriptors, des_list, valid_des = extract_sift_features(train_path)
    bow_trainer(image_paths, output, descriptors, des_list, num_words, valid_des)