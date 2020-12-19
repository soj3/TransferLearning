import os
import numpy as np
import cv2

def ext_keypoints_descriptors(data, extractor):
    image_list = []
    for root, dirs, files in os.walk("/Users/Chris Toomey/PycharmProjects/self_taught_clustering/Object_Data/" +
                                     data, topdown=False):
        for name in files:
            image_list.append(cv2.imread(os.path.join(root, name)))

    keypoints = []
    descriptors = []
    for image in image_list:
        keypoints_temp, descriptors_temp = features(image, extractor)
        keypoints.append(keypoints_temp)
        descriptors.append(descriptors_temp)
    return keypoints, descriptors


def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


def save_data(targ_descriptors, aux_descriptors, feature_descriptors):
    file = open('targ_image_descriptor.txt', 'w+')
    for image_targ_desc in targ_descriptors:
        np.savetxt(file, image_targ_desc)
    file.close

    file = open('aux_image_descriptor.txt', 'w+')
    for image_aux_desc in aux_descriptors:
      np.savetxt(file, image_aux_desc)
    file.close
    file = open('feature_desciptor.txt', 'w+')
    for feature in range(feature_descriptors):
        np.savetxt(file, feature)
    file.close


def get_saved_data():
    file = open('targ_image_descriptor.txt', 'r')
    targ_image_descriptors = file.readlines()
    file.close()

    file = open('aux_image_descriptor.txt', 'r')
    aux_image_descriptors = file.readlines()
    file.close()

    file = open('feature_desciptor.txt', 'r')
    feature_descriptors = file.readlines()
    file.close()