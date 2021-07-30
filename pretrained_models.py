from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, Model, optimizers, models, callbacks, activations




# =============================================================================
# MobilNetv1 Transfer Learning
# =============================================================================
def getMobilNetV1(IMG_ROWS, IMG_COLS):
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
    
    return model


# =============================================================================
# MobilNetv2 Transfer Learning
# =============================================================================

def getMobilNetV2(IMG_ROWS, IMG_COLS):
    mobilnetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_ROWS, IMG_COLS,3))
    
    # freeze the lower layers
    for layer in mobilnetv2_model.layers:
        layer.trainable = False 
    
    # add new top layers (layers that make prediction) starting from the Max Pooling layer -> Dense
    def addTopModelMobilNetV2(bottom_model, num_classes):
        top_model = bottom_model.output
        top_model = layers.GlobalAveragePooling2D()(top_model)
        top_model = layers.Dense(1, activation='sigmoid')(top_model)
        
        return top_model
    
    fc_head = addTopModelMobilNetV2(mobilnetv2_model, num_classes)
    model = Model(inputs=mobilnetv2_model.input, outputs=fc_head)
    
    return model


# =============================================================================
# ResNet50
# =============================================================================
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_means = X_train.mean(axis=0)
# X_stds = X_train.std(axis=0)
# X_train = (X_train-X_means)/X_stds
# X_val = (X_val-X_means)/X_stds
# X_test = (X_test-X_means)/X_stds 

resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_ROWS, IMG_COLS,3), classes=4)

for layer in resnet_model.layers:
    layer.trainable = False

def addTopModel(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = layers.GlobalAveragePooling2D()(top_model)
    top_model = layers.Dense(1, activation="sigmoid")(top_model)
    return top_model

fc_head = addTopModel(resnet_model, num_classes)
model = Model(inputs=resnet_model.input, outputs=fc_head)