import keras
import tensorflow


class Attention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.context = self.add_weight(shape=(input_shape[-1],),
                                       initializer="glorot_uniform",
                                       name="context")

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        def cosine_similarity(vec_a, vec_b):
            vec_a = tensorflow.nn.l2_normalize(x=vec_a, axis=-1)
            vec_b = tensorflow.nn.l2_normalize(x=vec_b, axis=-1)
            cos_sim = tensorflow.reduce_sum(tensorflow.multiply(vec_a, vec_b), axis=-1)

            return cos_sim

        m = cosine_similarity(inputs, self.context)

        s = keras.backend.softmax(m)

        z = keras.backend.sum(inputs * keras.backend.expand_dims(s, axis=-1), axis=1)

        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
