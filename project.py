# @inproceedings{gallagher_cvpr_09_groups,
# author = {A. Gallagher and T. Chen},
# title = {Understanding Images of Groups of People},
# booktitle = {Proc. CVPR},
# year = {2009},
# } http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html

# A. Gallagher, T. Chen, “Understanding Groups of Images of People,” IEEE Conference on Computer Vision and Pattern Recognition, 2009.

import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import cv2
import sys
import os, glob
from matplotlib import pyplot as plt

def getTotalFaces(filename):
    fileData = "../Fam2a/PersonData.txt"
    # print(filename)
    with open(fileData) as f:
        while True:
            line = f.readline()
            if not line:
                break
            #print(line)
            if (filename in line):
                # print(line)
                line = f.readline()
                c = 0
                # print(line)
                while ("\t" in line):
                    c = c + 1
                    line = f.readline()
                    # print(line)
                return c
    return 0


path = './train_data/'
faceDetectionPath = './FaceDetection/'
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
s = 0
s1 = 0

for filename in glob.glob(os.path.join(path, '*.jpg')):
    # read and convert image
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    totalFaces = getTotalFaces(filename.split("/")[2])
    print("Found {0} faces! Total Faces {1}, Name: {2}".format(len(faces), totalFaces, filename.split("/")[2]))
    s = s + totalFaces
    s1 = s1 + len(faces)

    # show face detections
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imwrite(faceDetectionPath+filename.split("/")[2], image)
    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)

os.system('python3 u2net_test1.py')

os.system('python3 u2net_test.py')

fig = plt.figure(figsize=(15, 8))
rows = 2
columns = 4
images1 = []
images2 = []
images3 = []
images4 = []


noFaceDetection = "./Fam2a/u2net_results/"
afterFaceDetection = "./FaceDetection/u2net_results/"
original = "./Fam2a/"
face = "./FaceDetection/"

dir_list1 = os.listdir(afterFaceDetection)
dir_list2 = os.listdir(noFaceDetection)

dir_list = [c for c in dir_list1 if c in dir_list2]

print(dir_list1, dir_list2, dir_list)

for name in dir_list:

    images1.append(cv2.imread(original + name))

    # Read First Image
    images2.append(cv2.imread(noFaceDetection + name))
    
    # Read Second Image
    images3.append(cv2.imread(afterFaceDetection + name))

    images4.append(cv2.imread(face + name))


fig.add_subplot(rows, columns, 1)
plt.imshow(images1[0])
plt.axis('off')
plt.title("Original")

fig.add_subplot(rows, columns, 2)
plt.imshow(images4[0])
plt.axis('off')
plt.title("Original")

fig.add_subplot(rows, columns, 3)
plt.imshow(images2[0])
number_of_white_pix = np.sum(images2[0] > 0)
plt.axis('off')
plt.title("No Face Detection, \nwith white pixels: " + str(number_of_white_pix))

fig.add_subplot(rows, columns, 4)
plt.imshow(images3[0])
number_of_white_pix = np.sum(images3[0] > 0)
plt.axis('off')
plt.title("Face Detection, \nwith white pixels: " + str(number_of_white_pix))



fig.add_subplot(rows, columns, 5)
plt.imshow(images1[1])
plt.axis('off')
plt.title("Original")

fig.add_subplot(rows, columns, 6)
plt.imshow(images4[1])
plt.axis('off')
plt.title("Original")

fig.add_subplot(rows, columns, 7)
plt.imshow(images2[1])
number_of_white_pix = np.sum(images2[1] > 0)
plt.axis('off')
plt.title("No Face Detection, \nwith white pixels: " + str(number_of_white_pix))

fig.add_subplot(rows, columns, 8)
plt.imshow(images3[1])
number_of_white_pix = np.sum(images3[1] > 0)
plt.axis('off')
plt.title("Face Detection, \nwith white pixels: " + str(number_of_white_pix))
