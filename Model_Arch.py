import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy, sparse_categorical_crossentropy
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2, Xception

def conv1x1(input_tensor, filters, kernel_size=(1,1), strides=(1,1), padding="valid", bn_axis=3):
    # 1x1 Concolution Layer
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer='he_normal',
        use_bias=False,)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    
    return x

def conv3x3(input_tensor, filters, kernel_size=(3,3), strides=(1,1), padding="valid", bn_axis=3):
    # 3x3 Convolution Layer
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer='he_normal',
        use_bias=False,)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    
    return x


## Basic blocks of model arch - Residual Blocks
def residual_block(input_tensor, filters, strides=(2,2)):
    # Residual Block with Skip_Connection
    filters_1, filters_2, filters_3, filters_4, filters_5 = filters
    
    # layer 1 - 4
    x = conv1x1(input_tensor, filters_1, strides=(2,2), padding="valid")
    x = Activation('relu')(x)
    x = conv3x3(x, filters_2, padding="same")
    x = Activation('relu')(x)
    x = conv1x1(x, filters_3, padding="valid")
    x = Activation('relu')(x)
    x = conv3x3(x, filters_4, padding="same")
    x = Activation('relu')(x)
    
    # Layer 5 & Skip Connection
    x = conv1x1(x, filters_5, padding="valid")
    skip_con = conv1x1(input_tensor, filters_5, strides=strides, padding="valid")
    
    x = Add()([x,skip_con])
    x = Activation('relu')(x)
    
    return x

# Residual Block without Skip_Connection
def identity_block(input_tensor, filters, strides=(2,2)):
    
    filters_1, filters_2, filters_3 = filters
    
    # layer 1
    x = conv1x1(input_tensor, filters_1, padding="valid")
    x = Activation('relu')(x)
    x = conv3x3(x, filters_2, padding="same")
    x = Activation('relu')(x)
    x = conv1x1(x, filters_3, padding="valid")
    
    x = Add()([x,input_tensor])
    x = Activation('relu')(x)
    
    return x


class BuildModel():
    """
        'ostroarthnet','resnet50', 'inceptionv3', 'mobilenet', 'xception'
    """

    def __init__(self, learning_rate=1e-3, input_shape=(224, 224,3), no_of_classes=5):
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.no_of_classes = no_of_classes
        self.model = None
        self.hist = None
        
    def OsteoArthNet(self,pooling=None):
        img_input = Input(shape=self.input_shape)
    
        # ConvLayer with (7,7) Kernel
        x = ZeroPadding2D(padding=(3, 3))(img_input)
        x = Conv2D(64, (7, 7),
                strides=(2, 2),
                padding='valid',
                kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(1,1))(x)
        
        # Block 1
        x = residual_block(input_tensor=x, filters=(64, 64, 64, 64, 256))
        x = identity_block(input_tensor=x, filters=(64, 64, 256))
        x = identity_block(input_tensor=x, filters=(64, 64, 256))
        
        # Block 2
        x = residual_block(input_tensor=x, filters=(256, 256, 256, 256, 512))
        x = identity_block(input_tensor=x, filters=(256, 256, 512))
        x = identity_block(input_tensor=x, filters=(256, 256, 512))
        
        # Block 3
        x = residual_block(input_tensor=x, filters=(512, 512, 512, 512, 1024))
        x = identity_block(input_tensor=x, filters=(512, 512, 1024))
        x = identity_block(input_tensor=x, filters=(512, 512, 1024))
        x = identity_block(input_tensor=x, filters=(512, 512, 1024))
        
        
        if pooling == 'max':
            x = GlobalMaxPool2D()(x)
        elif pooling == "avg" or pooling == None:
            x = GlobalAveragePooling2D()(x)
        
        x = Dense(self.no_of_classes, activation="softmax")(x)
        
        self.model = Model(img_input, x)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,decay=0.0001),
                           metrics=["acc"],
                           loss= tf.keras.losses.categorical_crossentropy)

        return self.model


#     def generate_model(self, base_model_name):
#         # Build desired model
#         if base_model_name == 'resnet50':
#             base_model = ResNet50(include_top=False,
#                                   weights="imagenet", input_shape=self.input_shape)
#         elif base_model_name == 'inceptionv3':
#             base_model = InceptionV3(include_top=False,
#                                      weights="imagenet", input_shape=self.input_shape)
#         elif base_model_name == 'mobilenet':
#             base_model = MobileNetV2(include_top=False,
#                                      weights="imagenet", input_shape=self.input_shape)
#         elif base_model_name == 'xception':
#             base_model = Xception(include_top=False,
#                                   weights="imagenet", input_shape=self.input_shape)

#         # for i in range(len(base_model.layers)):
#         #     base_model.layers[i].trainable = False

#         # Add Top Layers of model
#         x = GlobalAveragePooling2D()(base_model.output)
#         x = Flatten()(x)
#         x = Dense(128, activation='relu')(x)
#         x = Dropout(0.5)(x)
#         output = Dense(self.no_of_classes, activation='softmax')(x)

#         self.model = Model(base_model.input, output)

#         # Configure Model and Compile
#         self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,decay=0.0001),
#                  metrics=["acc"],
#                  loss= tf.keras.losses.categorical_crossentropy)

#         return self.model

    def train_model(self, train_data, validation_data, epochs=10, callbacks=None):
        self.hist = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
        )

        return self.hist

    def model_info(self):
        return self.model.summary()