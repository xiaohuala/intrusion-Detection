import keras


class LSTMCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, hidden_state_size)
        super(LSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 4), initializer="glorot_uniform", name="input_weight")
        self.Wxf = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxi = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxc = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.Wxo = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 4), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whf = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whi = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whc = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size
        self.Who = self.hidden_state_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 4,), initializer="zeros", name="bias")
        self.bf = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bi = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bc = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size
        self.bo = self.bias[self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        c_tm1 = states[1]

        x_t = inputs

        f_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxf) + keras.backend.dot(h_tm1, self.Whf) + self.bf)

        i_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxi) + keras.backend.dot(h_tm1, self.Whi) + self.bi)

        candidate_c_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxc) + keras.backend.dot(h_tm1, self.Whc) + self.bc)

        c_t = f_t * c_tm1 + i_t * candidate_c_t

        o_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxo) + keras.backend.dot(h_tm1, self.Who) + self.bo)

        h_t = o_t * keras.backend.tanh(c_t)

        return h_t, [h_t, c_t]


class DoILSTMCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, hidden_state_size, input_size)
        super(DoILSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 7), initializer="glorot_uniform", name="input_weight")
        self.Wxf = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxi = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxc = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.Wxo = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxf = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size
        self.d_Wxi = self.input_weight[:, self.hidden_state_size * 5: self.hidden_state_size * 6]  # 维度：input_size * hidden_state_size
        self.d_Wxo = self.input_weight[:, self.hidden_state_size * 6: self.hidden_state_size * 7]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 4), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whf = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whi = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whc = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size
        self.Who = self.hidden_state_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 4,), initializer="zeros", name="bias")
        self.bf = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bi = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bc = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size
        self.bo = self.bias[self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        c_tm1 = states[1]

        x_tm1 = states[2]

        x_t = inputs

        d_x_t = x_t - x_tm1

        f_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxf) + keras.backend.dot(d_x_t, self.d_Wxf) + keras.backend.dot(h_tm1, self.Whf) + self.bf)

        i_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxi) + keras.backend.dot(d_x_t, self.d_Wxi) + keras.backend.dot(h_tm1, self.Whi) + self.bi)

        candidate_c_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxc) + keras.backend.dot(h_tm1, self.Whc) + self.bc)

        c_t = f_t * c_tm1 + i_t * candidate_c_t

        o_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxo) + keras.backend.dot(d_x_t, self.d_Wxo) + keras.backend.dot(h_tm1, self.Who) + self.bo)

        h_t = o_t * keras.backend.tanh(c_t)

        return h_t, [h_t, c_t, x_t]


class DDoILSTMCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, hidden_state_size, input_size, input_size)
        super(DDoILSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 10), initializer="glorot_uniform", name="input_weight")
        self.Wxf = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxi = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxc = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.Wxo = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxf = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size
        self.d_Wxi = self.input_weight[:, self.hidden_state_size * 5: self.hidden_state_size * 6]  # 维度：input_size * hidden_state_size
        self.d_Wxo = self.input_weight[:, self.hidden_state_size * 6: self.hidden_state_size * 7]  # 维度：input_size * hidden_state_size
        self.dd_Wxf = self.input_weight[:, self.hidden_state_size * 7: self.hidden_state_size * 8]  # 维度：input_size * hidden_state_size
        self.dd_Wxi = self.input_weight[:, self.hidden_state_size * 8: self.hidden_state_size * 9]  # 维度：input_size * hidden_state_size
        self.dd_Wxo = self.input_weight[:, self.hidden_state_size * 9: self.hidden_state_size * 10]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 4), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whf = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whi = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whc = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size
        self.Who = self.hidden_state_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 4,), initializer="zeros", name="bias")
        self.bf = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bi = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bc = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size
        self.bo = self.bias[self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        c_tm1 = states[1]

        x_tm1 = states[2]

        x_tm2 = states[3]

        x_t = inputs

        d_x_t = x_t - x_tm1

        dd_x_t = x_t - 2 * x_tm1 + x_tm2

        f_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxf) + keras.backend.dot(d_x_t, self.d_Wxf) + keras.backend.dot(dd_x_t, self.dd_Wxf) + keras.backend.dot(h_tm1, self.Whf) + self.bf)

        i_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxi) + keras.backend.dot(d_x_t, self.d_Wxi) + keras.backend.dot(dd_x_t, self.dd_Wxi) + keras.backend.dot(h_tm1, self.Whi) + self.bi)

        candidate_c_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxc) + keras.backend.dot(h_tm1, self.Whc) + self.bc)

        c_t = f_t * c_tm1 + i_t * candidate_c_t

        o_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxo) + keras.backend.dot(d_x_t, self.d_Wxo) + keras.backend.dot(dd_x_t, self.dd_Wxo) + keras.backend.dot(h_tm1, self.Who) + self.bo)

        h_t = o_t * keras.backend.tanh(c_t)

        return h_t, [h_t, c_t, x_t, x_tm1]


class DDDoILSTMCell(keras.layers.Layer):
    def __init__(self, input_size, hidden_state_size, **kwargs):
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, hidden_state_size, input_size, input_size, input_size)
        super(DDDoILSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_weight = self.add_weight(shape=(self.input_size, self.hidden_state_size * 13), initializer="glorot_uniform", name="input_weight")
        self.Wxf = self.input_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：input_size * hidden_state_size
        self.Wxi = self.input_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：input_size * hidden_state_size
        self.Wxc = self.input_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：input_size * hidden_state_size
        self.Wxo = self.input_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：input_size * hidden_state_size
        self.d_Wxf = self.input_weight[:, self.hidden_state_size * 4: self.hidden_state_size * 5]  # 维度：input_size * hidden_state_size
        self.d_Wxi = self.input_weight[:, self.hidden_state_size * 5: self.hidden_state_size * 6]  # 维度：input_size * hidden_state_size
        self.d_Wxo = self.input_weight[:, self.hidden_state_size * 6: self.hidden_state_size * 7]  # 维度：input_size * hidden_state_size
        self.dd_Wxf = self.input_weight[:, self.hidden_state_size * 7: self.hidden_state_size * 8]  # 维度：input_size * hidden_state_size
        self.dd_Wxi = self.input_weight[:, self.hidden_state_size * 8: self.hidden_state_size * 9]  # 维度：input_size * hidden_state_size
        self.dd_Wxo = self.input_weight[:, self.hidden_state_size * 9: self.hidden_state_size * 10]  # 维度：input_size * hidden_state_size
        self.ddd_Wxf = self.input_weight[:, self.hidden_state_size * 10: self.hidden_state_size * 11]  # 维度：input_size * hidden_state_size
        self.ddd_Wxi = self.input_weight[:, self.hidden_state_size * 11: self.hidden_state_size * 12]  # 维度：input_size * hidden_state_size
        self.ddd_Wxo = self.input_weight[:, self.hidden_state_size * 12: self.hidden_state_size * 13]  # 维度：input_size * hidden_state_size

        self.hidden_state_weight = self.add_weight(shape=(self.hidden_state_size, self.hidden_state_size * 4), initializer="glorot_uniform", name="hidden_state_weight")
        self.Whf = self.hidden_state_weight[:, self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：hidden_state_size * hidden_state_size
        self.Whi = self.hidden_state_weight[:, self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：hidden_state_size * hidden_state_size
        self.Whc = self.hidden_state_weight[:, self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：hidden_state_size * hidden_state_size
        self.Who = self.hidden_state_weight[:, self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：hidden_state_size * hidden_state_size

        self.bias = self.add_weight(shape=(self.hidden_state_size * 4,), initializer="zeros", name="bias")
        self.bf = self.bias[self.hidden_state_size * 0: self.hidden_state_size * 1]  # 维度：1 * hidden_state_size
        self.bi = self.bias[self.hidden_state_size * 1: self.hidden_state_size * 2]  # 维度：1 * hidden_state_size
        self.bc = self.bias[self.hidden_state_size * 2: self.hidden_state_size * 3]  # 维度：1 * hidden_state_size
        self.bo = self.bias[self.hidden_state_size * 3: self.hidden_state_size * 4]  # 维度：1 * hidden_state_size

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]

        c_tm1 = states[1]

        x_tm1 = states[2]

        x_tm2 = states[3]

        x_tm3 = states[4]

        x_t = inputs

        d_x_t = x_t - x_tm1

        dd_x_t = x_t - 2 * x_tm1 + x_tm2

        ddd_x_t = x_t - 3 * x_tm1 + 3 * x_tm2 - x_tm3

        f_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxf) + keras.backend.dot(d_x_t, self.d_Wxf) + keras.backend.dot(dd_x_t, self.dd_Wxf) + keras.backend.dot(ddd_x_t, self.ddd_Wxf) + keras.backend.dot(h_tm1, self.Whf) + self.bf)

        i_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxi) + keras.backend.dot(d_x_t, self.d_Wxi) + keras.backend.dot(dd_x_t, self.dd_Wxi) + keras.backend.dot(ddd_x_t, self.ddd_Wxi) + keras.backend.dot(h_tm1, self.Whi) + self.bi)

        candidate_c_t = keras.backend.tanh(keras.backend.dot(x_t, self.Wxc) + keras.backend.dot(h_tm1, self.Whc) + self.bc)

        c_t = f_t * c_tm1 + i_t * candidate_c_t

        o_t = keras.backend.sigmoid(keras.backend.dot(x_t, self.Wxo) + keras.backend.dot(d_x_t, self.d_Wxo) + keras.backend.dot(dd_x_t, self.dd_Wxo) + keras.backend.dot(ddd_x_t, self.ddd_Wxo) + keras.backend.dot(h_tm1, self.Who) + self.bo)

        h_t = o_t * keras.backend.tanh(c_t)

        return h_t, [h_t, c_t, x_t, x_tm1, x_tm2]
