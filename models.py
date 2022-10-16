from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.layers as L



def layer_type1(x_inp, filters, kernel_size=(3, 3)):
    """_summary_

    Args:
        x_inp (_type_): _description_
        filters (_type_): _description_
        kernel_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        _type_: _description_
    """
    x = L.Conv2D(filters, kernel_size, padding="same")(x_inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    return x


def layer_type2(x_inp, filters, kernel_size=(3, 3)):
    """_summary_

    Args:
        x_inp (_type_): _description_
        filters (_type_): _description_
        kernel_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        _type_: _description_
    """
    x = layer_type1(x_inp, filters, kernel_size)
    x = layer_type1(x, filters, kernel_size)

    x = L.Add()([x, x_inp])

    return x


def layer_type3(x_inp, filters, kernel_size=(3, 3)):
    """_summary_

    Args:
        x_inp (_type_): _description_
        filters (_type_): _description_
        kernel_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        _type_: _description_
    """
    x = layer_type1(x_inp, filters, kernel_size)
    x = layer_type1(x, filters,kernel_size)
    x = L.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x_res = L.Conv2D(filters, kernel_size=(2,2), strides=(2, 2))(x_inp)
    x_res = L.BatchNormalization()(x_res)

    x = L.Add()([x, x_res])

    return x


def layer_type4(x_inp, filters, kernel_size=(3, 3)):
    """_summary_

    Args:
        x_inp (_type_): _description_
        filters (_type_): _description_
        kernel_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        _type_: _description_
    """
    x = layer_type1(x_inp, filters, kernel_size)
    x = layer_type1(x, filters, kernel_size)

    x = L.Conv2DTranspose(filters, kernel_size, padding='same', strides=(2, 2))(x)

    x = layer_type1(x, filters, kernel_size)
    x = layer_type1(x, filters, kernel_size)

    return x


def layer_type5(x_inp, filters, kernel_size=(3, 3)):
    """_summary_

    Args:
        x_inp (_type_): _description_
        filters (_type_): _description_
        kernel_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        _type_: _description_
    """
    x = L.Conv2D(filters, kernel_size, activation='sigmoid', padding="same")(x_inp)

    return x


def make_model1(input_shape1=(360, 340, 3), input_shape2=(360, 340, 1)):
    """_summary_

    Args:
        input_shape1 (tuple, optional): _description_. Defaults to (360, 340, 3).
        input_shape2 (tuple, optional): _description_. Defaults to (360, 340, 1).

    Returns:
        _type_: _description_
    """
    input1 = L.Input(shape=input_shape1)
    input2 = L.Input(shape=input_shape2)

    xb01 = layer_type1(input2, filters=8)
    xb02 = layer_type2(xb01, filters=8)
    xb03 = layer_type3(xb02, filters=16)

    xb04 = layer_type1(xb03, filters=16)
    xb05 = layer_type2(xb04, filters=16)
    xb06 = layer_type3(xb05, filters=32)

    xb07 = layer_type1(xb06, filters=32)
    xb08 = layer_type1(xb07, filters=32)
    xb09 = layer_type2(xb08, filters=32)
    xb10 = layer_type2(xb09, filters=32)

    xb11 = layer_type4(xb10, filters=16)
    xb12c = layer_type1(xb11, filters=16)
    xb12 = L.concatenate([xb12c, xb04], axis=-1)
    xb13 = layer_type2(xb12, filters=32)

    xb14 = layer_type4(xb13, filters=32)
    xb15c = layer_type1(xb14, filters=32)
    xb15 = L.concatenate([xb15c, xb01], axis=-1)
    xb16 = layer_type2(xb15, filters=40)

    x01 = layer_type1(input1, filters=8)
    x02c = layer_type2(x01, filters=8)
    x02 = L.concatenate([x02c, xb02], axis=-1)
    x03 = layer_type1(x02, filters=8)
    x04 = layer_type3(x03, filters=16)

    x05 = layer_type1(x04, filters=16)
    x06c = layer_type2(x05, filters=16)
    x06 = L.concatenate([x06c, xb05], axis=-1)
    x07 = layer_type1(x06, filters=16)
    x08 = layer_type3(x07, filters=32)

    x09 = layer_type1(x08, filters=32)
    x10 = layer_type1(x09, filters=32)
    x11 = layer_type2(x10, filters=32)
    x12c = layer_type2(x11, filters=32)
    x12 = L.concatenate([x12c, xb10], axis=-1)
    x13 = layer_type1(x12, filters=32)

    x14 = layer_type4(x13, filters=16)
    x15 = layer_type1(x14, filters=16)
    x16c = layer_type2(x15, filters=16)
    x16 = L.concatenate([x16c, xb13], axis=-1)
    x17c = layer_type1(x16, filters=16)
    x17 = L.concatenate([x07, x17c], axis=-1)
    x18 = layer_type1(x17, filters=32)

    x19 = layer_type4(x18, filters=8)
    x20c = layer_type1(x19, filters=8)
    x20 = L.concatenate([x03, x20c], axis=-1)
    x21 = layer_type1(x20, filters=8)
    x22c = layer_type2(x21, filters=8)
    x22 = L.concatenate([x22c, xb16], axis=-1)
    x23 = layer_type1(x22, filters=8)
    xout= layer_type5(x23, filters=3)

    model = Model(inputs=[input1, input2], outputs=xout)

    return model



if __name__ == '__main__':

    model = make_model1()
    model.summary()
