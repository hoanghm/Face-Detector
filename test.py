import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras

project_folder = 'C:/Users/hoang/Downloads/CelebA'
models_folder = project_folder + '/trained_models'
image_folder = project_folder + '/' + 'img_align_celeba/cropped_celeba'

GENDER_MODEL_LINK = models_folder + '/' + 'Male_trained'
YOUNG_MODEL_LINK = models_folder + '/' + 'Young_trained'
CHUBBY_MODEL_LINK = models_folder + '/' + 'Chubby_trained'
SMILING_MODEL_LINK = models_folder + '/' + 'Smiling_trained'
EYEGLASSES_MODEL_LINK = models_folder + '/' + 'Eyeglasses_trained'


gender_model = keras.models.load_model(GENDER_MODEL_LINK)
young_model = keras.models.load_model(YOUNG_MODEL_LINK)
chubby_model = keras.models.load_model(CHUBBY_MODEL_LINK)
smiling_model = keras.models.load_model(SMILING_MODEL_LINK)
eyeglasses_model = keras.models.load_model(EYEGLASSES_MODEL_LINK)  


models = [gender_model, young_model, chubby_model, smiling_model, eyeglasses_model]
models_names = ['male', 'young', 'chubby', 'smiling', 'eyeglasses']


for model, name in zip(models, models_names):
    model_json = model.to_json()
    with open(models_folder + "/" + name + "_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(models_folder + "/" + name + "_model.h5")


# load json and create model
json_file = open(models_folder + "/gender_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(models_folder + "/gender_model.h5")
print("Loaded model from disk")

# # compile
loaded_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
