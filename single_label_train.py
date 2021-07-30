import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, Model, optimizers, models, callbacks, activations


link_dataset = 'C:/Users/hoang/Downloads/CelebA'
image_folder = link_dataset + '/' + 'img_align_celeba/cropped_celeba'
ATTRIBS = ['Wearing_Hat']
SAVE_MODEL = True

os.chdir(link_dataset)
attribs_df = pd.read_csv(link_dataset + '/list_attr_cropped.csv')
df = attribs_df.loc[:,['image_id'] + ATTRIBS]

# convert image_id to absolute path, e.g '000001.jpg' -> 'C:/.../000001.jpg'
df.loc[:,'image_id'] = image_folder + '/' + df['image_id'].values

# convert -1 to 0
for col in df.columns[1:]:
    df.loc[:,col] = np.where(df[col].values>0, 1, 0)

# separate positive labels and negative labels
pos_df = df.loc[df[ATTRIBS[0]] == 1]
neg_df = df.loc[df[ATTRIBS[0]] == 0]    

# create new df with equal ratio of positive and negative classes
min_size = min(pos_df.shape[0], neg_df.shape[0])
pos_df = pos_df.iloc[:min_size]
neg_df = neg_df.iloc[:min_size]
df = pd.concat([pos_df,neg_df], axis=0)


    
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
# MobilNetv1 Transfer Learning
# =============================================================================

num_classes = 2
mobilnet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_ROWS, IMG_COLS,3))

# freeze the lower layers
for layer in mobilnet_model.layers:
    layer.trainable = False 

# add new top layers (layers that make prediction) starting from the Max Pooling layer -> Dense
def addTopModelMobilNetV1(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = layers.GlobalAveragePooling2D()(top_model)
    top_model = layers.Dense(1024, activation='relu')(top_model)
    top_model = layers.Dense(1024, activation='relu')(top_model)
    top_model = layers.Dense(512, activation='relu')(top_model)
    top_model = layers.Dense(1, activation='sigmoid')(top_model)
    
    return top_model

fc_head = addTopModelMobilNetV1(mobilnet_model, num_classes)
model = Model(inputs=mobilnet_model.input, outputs=fc_head)


# =============================================================================
# Start Training
# =============================================================================

earlystopping_cb = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
history = model.fit_generator(generator=train_generator, 
                              steps_per_epoch=train_df.shape[0]//TRAIN_BATCH_SIZE, 
                              validation_data = val_generator,
                              epochs = 10,
                              callbacks = [earlystopping_cb]
                              )


# =============================================================================
# See Results 
# =============================================================================

def predict_image(img_link=None):
    if img_link is None:
        img_link = test_df.iloc[np.random.randint(0,test_df.shape[0])].values[0]
    # img_name = img_link[-10:]
    img = cv2.imread(img_link)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
    img = img/255.
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
    truelabel = test_df.loc[test_df['image_id'] == img_link][ATTRIBS].values[0][0]
    plt.imshow(img)
    print("True label:", 'Positive' if truelabel > 0.5 else 'Negative')
    print("Predicted label:", 'Positive' if prediction > 0.5 else 'Negative', '({})'.format(prediction))
    
    
# =============================================================================
# Save Model
# =============================================================================
if SAVE_MODEL:
    model_name = ATTRIBS[0].lower()
    models_folder = link_dataset + '/trained_models' + '/' + model_name + '_trained'  
    model_json = model.to_json()
    with open(models_folder + "/" + model_name + "_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(models_folder + "/" + model_name + "_model.h5")
    print("Model saved at", models_folder)

