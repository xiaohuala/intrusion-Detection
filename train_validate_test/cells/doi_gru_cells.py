import keras
import tensorflow


class GRUCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = hidden_state_size
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 3), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        x_t = inputs

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t]


class DoIGRUCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, input_size)
        super(DoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 5), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        x_tm1 = states[1]

        x_t = inputs

        d_x_t = x_t - x_tm1

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, x_t]


class DDoIGRUCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, input_size, input_size)
        super(DDoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 7), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size
        self.dd_Wxr = self.input_weight[:, self.hidden_state_size * 5: self.hidden_state_size * 6]  # 维度：input_size * hidden_state_size
        self.dd_Wxu = self.input_weight[:, self.hidden_state_size * 6: self.hidden_state_size * 7]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        x_tm1 = states[1]

        x_tm2 = states[2]

        x_t = inputs

        d_x_t = x_t - x_tm1

        dd_x_t = x_t - 2 * x_tm1 + x_tm2

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(dd_x_t, self.dd_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(dd_x_t, self.dd_Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, x_t, x_tm1]


class DDDoIGRUCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, input_size, input_size, input_size)
        super(DDDoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 9), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size
        self.dd_Wxr = self.input_weight[:, self.hidden_state_size * 5: self.hidden_state_size * 6]  # 维度：input_size * hidden_state_size
        self.dd_Wxu = self.input_weight[:, self.hidden_state_size * 6: self.hidden_state_size * 7]  # 维度：input_size * hidden_state_size
        self.ddd_Wxr = self.input_weight[:, self.hidden_state_size * 7: self.hidden_state_size * 8]  # 维度：input_size * hidden_state_size
        self.ddd_Wxu = self.input_weight[:, self.hidden_state_size * 8: self.hidden_state_size * 9]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        x_tm1 = states[1]

        x_tm2 = states[2]

        x_tm3 = states[3]

        x_t = inputs

        d_x_t = x_t - x_tm1

        dd_x_t = x_t - 2 * x_tm1 + x_tm2

        ddd_x_t = x_t - 3 * x_tm1 + 3 * x_tm2 - x_tm3

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(dd_x_t, self.dd_Wxr) + keras.backend.dot(ddd_x_t, self.ddd_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(dd_x_t, self.dd_Wxu) + keras.backend.dot(ddd_x_t, self.ddd_Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, x_t, x_tm1, x_tm2]


class AnotherGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(AnotherGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(input_shape[-1], self.units * 3), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.units * 0: self.units * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.units * 1: self.units * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.units * 2: self.units * 3]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.units, self.units * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.units * 0: self.units * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.units * 1: self.units * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.units * 2: self.units * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.units * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.units * 0: self.units * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.units * 1: self.units * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.units * 2: self.units * 3]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]
        x_t = inputs

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t]


class AnotherDoIGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        super(AnotherDoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(input_shape[-1], self.units * 5), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.units * 0: self.units * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.units * 1: self.units * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.units * 2: self.units * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.units * 3: self.units * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.units * 4: self.units * 5]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.units, self.units * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.units * 0: self.units * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.units * 1: self.units * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.units * 2: self.units * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.units * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.units * 0: self.units * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.units * 1: self.units * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.units * 2: self.units * 3]  # 维度：1 * hidden_state_size

        def constant_initializer(*args, **kwargs):
            eye = keras.backend.eye(size=input_shape[-1])  # 维度：input_size * input_size
            zeros = keras.backend.zeros(shape=(self.units - input_shape[-1], input_shape[-1]))  # hidden_state_size - input_size * input_size

            return keras.backend.concatenate(tensors=[eye, zeros], axis=0)  # hidden_state_size * input_size

        self.constant = self.add_weight(shape=(self.units, input_shape[-1]), initializer=constant_initializer, name="constant", trainable=False)

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]
        x_tm1 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[1], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_t = inputs
        d_x_t = x_t - x_tm1

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, tensorflow.einsum("bi,hi->bc", x_t, self.constant)]


class AnotherDDoIGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)
        super(AnotherDDoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(input_shape[-1], self.units * 7), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.units * 0: self.units * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.units * 1: self.units * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.units * 2: self.units * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.units * 3: self.units * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.units * 4: self.units * 5]  # 维度：input_size * hidden_state_size
        self.dd_Wxr = self.input_weight[:, self.units * 5: self.units * 6]  # 维度：input_size * hidden_state_size
        self.dd_Wxu = self.input_weight[:, self.units * 6: self.units * 7]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.units, self.units * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.units * 0: self.units * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.units * 1: self.units * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.units * 2: self.units * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.units * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.units * 0: self.units * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.units * 1: self.units * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.units * 2: self.units * 3]  # 维度：1 * hidden_state_size

        def constant_initializer(*args, **kwargs):
            eye = keras.backend.eye(size=input_shape[-1])  # 维度：input_size * input_size
            zeros = keras.backend.zeros(shape=(self.units - input_shape[-1], input_shape[-1]))  # hidden_state_size - input_size * input_size

            return keras.backend.concatenate(tensors=[eye, zeros], axis=0)  # hidden_state_size * input_size

        self.constant = self.add_weight(shape=(self.units, input_shape[-1]), initializer=constant_initializer, name="constant", trainable=False)

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]
        x_tm1 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[1], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_tm2 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[2], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_t = inputs
        d_x_t = x_t - x_tm1
        dd_x_t = x_t - 2 * x_tm1 + x_tm2

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(dd_x_t, self.dd_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(dd_x_t, self.dd_Wxu) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, tensorflow.einsum("bi,hi->bc", x_t, self.constant), tensorflow.einsum("bi,hi->bc", x_tm1, self.constant)]


class AnotherDDDoIGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units, units)
        super(AnotherDDDoIGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(input_shape[-1], self.units * 9), initializer="glorot_uniform", name="input_weight")
        self.Wxr = self.input_weight[:, self.units * 0: self.units * 1]  # 维度：input_size * hidden_state_size
        self.Wxu = self.input_weight[:, self.units * 1: self.units * 2]  # 维度：input_size * hidden_state_size
        self.Wxh = self.input_weight[:, self.units * 2: self.units * 3]  # 维度：input_size * hidden_state_size
        self.d_Wxr = self.input_weight[:, self.units * 3: self.units * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxu = self.input_weight[:, self.units * 4: self.units * 5]  # 维度：input_size * hidden_state_size
        self.dd_Wxr = self.input_weight[:, self.units * 5: self.units * 6]  # 维度：input_size * hidden_state_size
        self.dd_Wxu = self.input_weight[:, self.units * 6: self.units * 7]  # 维度：input_size * hidden_state_size
        self.ddd_Wxr = self.input_weight[:, self.units * 7: self.units * 8]  # 维度：input_size * hidden_state_size
        self.ddd_Wxu = self.input_weight[:, self.units * 8: self.units * 9]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.units, self.units * 3), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whr = self.hidden_state_weight[:, self.units * 0: self.units * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whu = self.hidden_state_weight[:, self.units * 1: self.units * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whh = self.hidden_state_weight[:, self.units * 2: self.units * 3]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.units * 3,), initializer="zeros", name="bias")
        self.br = self.bias[self.units * 0: self.units * 1]  # 维度：1 * hidden_state_size
        self.bu = self.bias[self.units * 1: self.units * 2]  # 维度：1 * hidden_state_size
        self.bh = self.bias[self.units * 2: self.units * 3]  # 维度：1 * hidden_state_size

        def constant_initializer(*args, **kwargs):
            eye = keras.backend.eye(size=input_shape[-1])  # 维度：input_size * input_size
            zeros = keras.backend.zeros(shape=(self.units - input_shape[-1], input_shape[-1]))  # hidden_state_size - input_size * input_size

            return keras.backend.concatenate(tensors=[eye, zeros], axis=0)  # hidden_state_size * input_size

        self.constant = self.add_weight(shape=(self.units, input_shape[-1]), initializer=constant_initializer, name="constant", trainable=False)

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]
        x_tm1 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[1], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_tm2 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[2], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_tm3 = keras.backend.sum(x=tensorflow.einsum("bc,ih->bih", states[3], keras.backend.transpose(self.constant)), axis=-1, keepdims=False)
        x_t = inputs
        d_x_t = x_t - x_tm1
        dd_x_t = x_t - 2 * x_tm1 + x_tm2
        ddd_x_t = x_t - 2 * x_tm1 + 3 * x_tm2 - x_tm3

        r_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxr) + keras.backend.dot(d_x_t, self.d_Wxr) + keras.backend.dot(dd_x_t, self.dd_Wxr) + keras.backend.dot(ddd_x_t, self.ddd_Wxr) + keras.backend.dot(h_tm1, self.Whr) + self.br)

        u_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxu) + keras.backend.dot(d_x_t, self.d_Wxu) + keras.backend.dot(dd_x_t, self.dd_Wxu) + keras.backend.dot(ddd_x_t, self.ddd_Wxr) + keras.backend.dot(h_tm1, self.Whu) + self.bu)

        candidate_h_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxh) + keras.backend.dot(r_t * h_tm1, self.Whh) + self.bh)

        h_t = u_t * h_tm1 + (1 - u_t) * candidate_h_t

        return h_t, [h_t, tensorflow.einsum("bi,hi->bc", x_t, self.constant), tensorflow.einsum("bi,hi->bc", x_tm1, self.constant), tensorflow.einsum("bi,hi->bc", x_tm2, self.constant)]
