import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Concatenate, Reshape, RNN, Bidirectional, LSTM

from train_validate_test.layers.layers import Attention
from train_validate_test.cells.doi_gru_cells import DDDoIGRUCell
from train_validate_test.metrics.metrics import accuracy, precision, recall, fp_rate, fn_rate

# memory的输入特征的数量
memory_n_input = 62
# cpu的输入特征的数量
cpu_n_input = 5
# system_call的输入特征的数量
system_call_n_input = 5
# network_traffic的输入特征的数量
network_traffic_n_input = 15
# 时间步长的数量.
n_step = 60

# 循环层的隐藏单元的数量
rnn_layer_units = 128
# 全连接层的隐藏单元的数量
fully_connected_layer_units = 128
# 输出层的隐藏单元的数量
output_layer_units = 2

# epoch的数量
n_epoch = 200
# mini_batch的数量
n_mini_batch = 32
# 训练数据集的validation_rate比例划分成为验证数据集
n_validation_rate = 0.2

# 模型参数的存储路径
model_parameter_filepath = "./other_model_parameter/DDDoIGRUCell.hdf5"
# 训练数据集的训练指标的存储路径
train_metrics_filepath = "./other_train_metrics/DDDoIGRUCell.npy"
# 测试数据集的测试指标的存储路径
test_metrics_filepath = "./other_test_metrics/DDDoIGRUCell.npy"

# 训练、测试和验证的次数
n_time = 1


def load_data():
    train_data = np.load("../dataset/train_data.npy")
    train_labels = np.load("../dataset/train_labels.npy")

    test_data = np.load("../dataset/test_data.npy")
    test_labels = np.load("../dataset/test_labels.npy")

    return train_data, train_labels, test_data, test_labels


def split_data(data):
    memory_data = data[..., :62]
    cpu_data = data[..., 62:67]
    system_call_data = data[..., 67:72]
    network_traffic_data = data[..., 72:]

    return memory_data, cpu_data, system_call_data, network_traffic_data


def build_model(memory_rnn_cell, cpu_rnn_cell, system_call_rnn_cell, network_traffic_rnn_cell):
    memory_input_layer = Input(shape=(n_step, memory_n_input))
    memory_rnn_layer = RNN(cell=memory_rnn_cell, return_sequences=True)(memory_input_layer)
    memory_attention_layer = Attention()(memory_rnn_layer)

    cpu_input_layer = Input(shape=(n_step, cpu_n_input))
    cpu_rnn_layer = RNN(cell=cpu_rnn_cell)(cpu_input_layer)
    cpu_attention_layer = Attention()(cpu_rnn_layer)

    system_call_input_layer = Input(shape=(n_step, system_call_n_input))
    system_call_rnn_layer = RNN(cell=system_call_rnn_cell)(system_call_input_layer)
    system_call_attention_layer = Attention()(system_call_rnn_layer)

    network_traffic_input_layer = Input(shape=(n_step, network_traffic_n_input))
    network_traffic_rnn_layer = RNN(cell=network_traffic_rnn_cell)(network_traffic_input_layer)
    network_traffic_attention_layer = Attention()(network_traffic_rnn_layer)

    concatenate_layer = Concatenate(axis=-1)([memory_attention_layer, cpu_attention_layer, system_call_attention_layer, network_traffic_attention_layer])
    input_layer = Reshape(target_shape=(4, -1))(concatenate_layer)
    rnn_layer = Bidirectional(LSTM(units=rnn_layer_units))(input_layer)
    fully_connected_layer = Dense(units=fully_connected_layer_units, activation="relu")(rnn_layer)
    output_layer = Dense(units=output_layer_units, activation="softmax")(fully_connected_layer)
    model = Model(inputs=[memory_input_layer, cpu_input_layer, system_call_input_layer, network_traffic_input_layer], outputs=output_layer)

    return model


def train(model, train_data, train_labels):
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=[accuracy, precision, recall, fp_rate, fn_rate])

    tensor_board = TensorBoard(log_dir="./other_log/DDDoIGRUCell")  # tensorboard --logdir=./

    model_check_point = ModelCheckpoint(filepath=model_parameter_filepath,
                                        monitor="accuracy",
                                        mode="max",
                                        # save_best_only=True,
                                        save_weights_only=True,
                                        verbose=1)

    train_metrics = model.fit(x=train_data,
                              y=train_labels,
                              validation_split=n_validation_rate,
                              epochs=n_epoch,
                              batch_size=n_mini_batch,
                              callbacks=[tensor_board, model_check_point],
                              verbose=1)

    return train_metrics.history


def test(model, test_data, test_labels):
    model.load_weights(filepath=model_parameter_filepath)
    test_metrics = model.evaluate(x=test_data,
                                  y=test_labels,
                                  batch_size=n_mini_batch,
                                  verbose=1)

    return test_metrics


train_data, train_labels, test_data, test_labels = load_data()
memory_train_data, cpu_train_data, system_call_train_data, network_traffic_train_data = split_data(train_data)
memory_test_data, cpu_test_data, system_call_test_data, network_traffic_test_data = split_data(test_data)

memory_cell = DDDoIGRUCell(input_size=memory_n_input, hidden_state_size=rnn_layer_units)
cpu_cell = DDDoIGRUCell(input_size=cpu_n_input, hidden_state_size=rnn_layer_units)
system_call_cell = DDDoIGRUCell(input_size=system_call_n_input, hidden_state_size=rnn_layer_units)
network_traffic_cell = DDDoIGRUCell(input_size=network_traffic_n_input, hidden_state_size=rnn_layer_units)

model = build_model(memory_rnn_cell=memory_cell,
                    cpu_rnn_cell=cpu_cell,
                    system_call_rnn_cell=system_call_cell,
                    network_traffic_rnn_cell=network_traffic_cell)

test_metrics = {
    "test_loss": [],
    "test_accuracy": [],
    "test_precision": [],
    "test_recall": [],
    "test_fp_rate": [],
    "test_fn_rate": [],
}

for _ in range(n_time):
    train_metrics = train(model=model,
                          train_data=[memory_train_data, cpu_train_data, system_call_train_data, network_traffic_train_data],
                          train_labels=train_labels)
    np.save(train_metrics_filepath, train_metrics)

    test_metrics_tmp = test(model=model,
                            test_data=[memory_test_data, cpu_test_data, system_call_test_data, network_traffic_test_data],
                            test_labels=test_labels)
    test_metrics["test_loss"].append(test_metrics_tmp[0])
    test_metrics["test_accuracy"].append(test_metrics_tmp[1])
    test_metrics["test_precision"].append(test_metrics_tmp[2])
    test_metrics["test_recall"].append(test_metrics_tmp[3])
    test_metrics["test_fp_rate"].append(test_metrics_tmp[4])
    test_metrics["test_fn_rate"].append(test_metrics_tmp[5])
np.save(test_metrics_filepath, test_metrics)
