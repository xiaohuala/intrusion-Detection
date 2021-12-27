import keras


def tp(y_true, y_predict):
    y_true = keras.backend.argmax(y_true)
    y_predict = keras.backend.argmax(y_predict)
    y_predict = keras.backend.clip(y_predict, 0, 1)
    y_predict = keras.backend.round(y_predict)
    y = y_true * y_predict
    y = keras.backend.cast(y, keras.backend.floatx())

    return keras.backend.sum(y)


def fp(y_true, y_predict):
    y_true = keras.backend.argmax(y_true)
    y_predict = keras.backend.argmax(y_predict)
    y_predict = keras.backend.clip(y_predict, 0, 1)
    y_predict = keras.backend.round(y_predict)
    y_true = keras.backend.ones_like(y_true) - y_true
    y = y_true * y_predict
    y = keras.backend.cast(y, keras.backend.floatx())

    return keras.backend.sum(y)


def tn(y_true, y_predict):
    y_true = keras.backend.argmax(y_true)
    y_predict = keras.backend.argmax(y_predict)
    y_predict = keras.backend.clip(y_predict, 0, 1)
    y_predict = keras.backend.round(y_predict)
    y_predict = keras.backend.ones_like(y_predict) - y_predict
    y_true = keras.backend.ones_like(y_true) - y_true
    y = y_true * y_predict
    y = keras.backend.cast(y, keras.backend.floatx())

    return keras.backend.sum(y)


def fn(y_true, y_predict):
    y_true = keras.backend.argmax(y_true)
    y_predict = keras.backend.argmax(y_predict)
    y_predict = keras.backend.clip(y_predict, 0, 1)
    y_predict = keras.backend.round(y_predict)
    y_predict = keras.backend.ones_like(y_predict) - y_predict
    y = y_true * y_predict
    y = keras.backend.cast(y, keras.backend.floatx())

    return keras.backend.sum(y)


def accuracy(y_true, y_predict):
    _tp = tp(y_true, y_predict)
    _fp = fp(y_true, y_predict)
    _tn = tn(y_true, y_predict)
    _fn = fn(y_true, y_predict)

    return (_tp + _tn) / (_tp + _fp + _tn + _fn + keras.backend.epsilon())


def precision(y_true, y_predict):
    _tp = tp(y_true, y_predict)
    _fp = fp(y_true, y_predict)

    return _tp / (_tp + _fp + keras.backend.epsilon())


def recall(y_true, y_predict):
    _tp = tp(y_true, y_predict)
    _fn = fn(y_true, y_predict)

    return _tp / (_tp + _fn + keras.backend.epsilon())


def fp_rate(y_true, y_predict):
    _fp = fp(y_true, y_predict)
    _tn = tn(y_true, y_predict)

    return _fp / (_fp + _tn + keras.backend.epsilon())


def fn_rate(y_true, y_predict):
    _fn = fn(y_true, y_predict)
    _tp = tp(y_true, y_predict)

    return _fn / (_fn + _tp + keras.backend.epsilon())
