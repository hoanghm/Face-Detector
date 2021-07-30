import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

link_dataset = 'C:/Users/hoang/Downloads/CelebA'
image_folder = link_dataset + '/' + 'img_align_celeba/img_align_celeba'
image_folder_save = link_dataset + '/' + 'img_align_celeba/cropped_celeba'
os.chdir(link_dataset)

# =============================================================================
# CV2 Face detector
# =============================================================================

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def detect_face(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    x,y,w,h = faces[0]
    cropped = img[y:y+h,x:x+w]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped

img = cv2.imread(image_folder + '/000001.jpg')
cropped = detect_face(img)


# =============================================================================
# Cropping and Saving Images
# =============================================================================

checkpoint = 0
i = 0
for f_name in os.listdir(image_folder):
    if i > checkpoint:
        checkpoint += 1000
        print("checking image {}".format(checkpoint))
    i += 1
    if f_name.endswith(".jpg"):
        img = cv2.imread(image_folder + '/' + f_name)
        cropped_img = detect_face(img)
        if cropped_img is not None:
            plt.imsave(image_folder_save + '/' + f_name, cropped_img)


