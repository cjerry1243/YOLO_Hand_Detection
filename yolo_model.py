from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, \
    Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet


def FullYolo(input_shape):
    input_image = Input(shape=input_shape)

    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 4
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 5
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 6
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 7
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 9
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 10
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 11
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 12
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 13
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    ### change order
    x = MaxPooling2D(pool_size=(2, 2))(x)
    skip_connection = x

    # Layer 14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 17
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 18
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Layer 21
    # skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    # skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    # skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    # skip_connection = Lambda(space_to_depth_x2)(skip_connection)
    skip_connection = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
        skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    x = concatenate([skip_connection, x])
    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    model = Model(input_image, x)
    model.summary()
    return model


def TinyYolo(input_shape):
    input_image = Input(shape=input_shape)
    # Layer 1
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 2 - 5
    for i in range(0, 4):
        x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(i + 2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # Layer 6
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    # Layer 7 - 8
    for i in range(0, 2):
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(i + 7))(x)
        x = LeakyReLU(alpha=0.1)(x)

    model = Model(input_image, x)
    model.summary()
    return model


def Vgg16_Yolo(input_shape=(240, 320, 3)):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg16.summary()
    img_input = Input(shape=input_shape)
    yolo = vgg16(img_input)

    yolo = Conv2D(128, [1, 1])(yolo)
    yolo = BatchNormalization(name='norm_1')(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)

    yolo = Flatten()(yolo)
    yolo = Dense(1024)(yolo)
    yolo = BatchNormalization()(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)
    yolo = Dense(576)(yolo)
    yolo = Reshape((6, 8, 12))(yolo)
    model = Model(img_input, yolo)
    model.summary()
    return model


def Xception_Yolo(input_shape=(240, 320, 3)):
    xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    xception.summary()
    img_input = Input(shape=input_shape)
    yolo = xception(img_input)

    yolo = Conv2D(128, [1, 1])(yolo)
    yolo = BatchNormalization(name='norm_1')(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)

    yolo = Flatten()(yolo)
    yolo = Dense(1024)(yolo)
    yolo = BatchNormalization()(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)
    yolo = Dense(576)(yolo)
    yolo = Reshape((6, 8, 12))(yolo)
    model = Model(img_input, yolo)
    model.summary()
    return model


def MobileNet_Yolo(input_shape=(240, 320, 3)):
    mobile = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    mobile.summary()
    img_input = Input(shape=input_shape)
    yolo = mobile(img_input)

    yolo = Conv2D(128, [1, 1])(yolo)
    yolo = BatchNormalization(name='norm_1')(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)

    yolo = Flatten()(yolo)
    yolo = Dense(1024)(yolo)
    yolo = BatchNormalization()(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)
    yolo = Dense(576)(yolo)
    yolo = Reshape((6, 8, 12))(yolo)
    model = Model(img_input, yolo)
    model.summary()
    return model



def ResNet_Yolo(input_shape=(240, 320, 3), train_resnet=True):
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # resnet.summary()

    resnet.trainable = train_resnet
    img_input = Input(shape=input_shape)
    yolo = resnet(img_input)

    yolo = Conv2D(128, [1, 1])(yolo)
    yolo = BatchNormalization(name='norm_1')(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)
    yolo = Dropout(0.2)(yolo)

    yolo = Flatten()(yolo)
    yolo = Dense(1024)(yolo)
    yolo = BatchNormalization()(yolo)
    yolo = LeakyReLU(alpha=0.1)(yolo)
    yolo = Dropout(0.2)(yolo)

    yolo = Dense(576)(yolo)
    yolo = Reshape((6, 8, 12))(yolo)
    model = Model(img_input, yolo)
    # model.summary()
    return model


def build_model(model_name='VGG', input_shape=(240, 320, 3)):
    model = None

    if model_name == 'VGG':
        model = Vgg16_Yolo(input_shape)

    elif model_name == 'Xception':
        model = Xception_Yolo(input_shape)

    elif model_name == 'Mobile':
        model = MobileNet_Yolo(input_shape)

    return model


if __name__ == '__main__':
    # FullYolo(input_shape=(240, 320, 3))
    # TinyYolo(input_shape=(240, 320, 3))
    # Vgg16_Yolo()
    # Xception_Yolo()
    # MobileNet_Yolo()
    # ResNet_Yolo(train_resnet=False).summary()

    pass
