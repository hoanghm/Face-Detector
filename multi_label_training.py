import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, Model, optimizers, models, callbacks, activations


link_dataset = 'C:/Users/hoang/Downloads/CelebA'
image_folder = link_dataset + '/' + 'img_align_celeba/cropped_celeba'
os.chdir(link_dataset)

attribs_df = pd.read_csv('list_attr_cropped.csv')

ATTRIBS = ['Eyeglasses', 'Male']

df = attribs_df[['image_id'] + ATTRIBS]



# convert image_id to absolute path, e.g '000001.jpg' -> 'C:/.../000001.jpg'
df.loc[:,'image_id'] = image_folder + '/' + df['image_id'].values

# convert -1 to 0
for col in df.columns[1:]:
    df.loc[:,col] = np.where(df[col].values>0, 1, 0)
    


# =============================================================================
# Data Flow
# =============================================================================

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
IMG_ROWS, IMG_COLS = 128, 128
num_samples = df.shape[0]

indices = np.random.permutation(num_samples)
train_cutoff = int(num_samples*0.8)
val_cutoff = train_cutoff + int(num_samples*0.1)

train_df = df.iloc[indices[:train_cutoff]]
val_df =  df.iloc[indices[train_cutoff:val_cutoff]]
test_df = df.iloc[indices[val_cutoff:]]

datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(train_df, 
                                              directiory= None,     # image_id must have absolute paths for some reason
                                              x_col = 'image_id',
                                              y_col = ATTRIBS,
                                              batch_size=TRAIN_BATCH_SIZE,
                                              seed=42,
                                              class_mode = 'raw',   # numpy array of values in y_col columns
                                              target_size=(IMG_ROWS,IMG_COLS)
                                              )
val_generator = datagen.flow_from_dataframe(val_df, 
                                              directiory= None,     # image_id must have absolute paths for some reason
                                              x_col = 'image_id',
                                              y_col = ATTRIBS,
                                              batch_size=VAL_BATCH_SIZE,
                                              seed=42,
                                              class_mode = 'raw',
                                              target_size=(IMG_ROWS,IMG_COLS)
                                              )
test_generator = datagen.flow_from_dataframe(test_df, 
                                              directiory= None,     # image_id must have absolute paths for some reason
                                              x_col = 'image_id',
                                              y_col = ATTRIBS,
                                              batch_size=1,
                                              seed=42,
                                              class_mode = 'raw',
                                              target_size=(IMG_ROWS,IMG_COLS)
                                              )

# =============================================================================
# MobilNet Transfer Learning
# =============================================================================

num_classes = len(ATTRIBS)
mobilnet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_ROWS, IMG_COLS,3))

# freeze the lower layers
for layer in mobilnet_model.layers:
    layer.trainable = False 

# add new top layers (layers that make prediction) starting from the Max Pooling layer -> Dense
def addTopModelResnNet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = layers.GlobalAveragePooling2D()(top_model)
    top_model = layers.Dense(1024, activation='relu')(top_model)
    top_model = layers.Dense(1024, activation='relu')(top_model)
    top_model = layers.Dense(512, activation='relu')(top_model)
    top_model = layers.Dense(num_classes, activation='sigmoid')(top_model)
    
    return top_model

fc_head = addTopModelResnNet(mobilnet_model, num_classes)
model = Model(inputs=mobilnet_model.input, outputs=fc_head)
# print(model.summary())

earlystopping_cb = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
history = model.fit_generator(generator=train_generator, 
                              steps_per_epoch=train_df.shape[0]//TRAIN_BATCH_SIZE, 
                              validation_data = val_generator,
                              epochs = 30,
                              callbacks = [earlystopping_cb]
                              )


