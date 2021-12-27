import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, RNN
from keras.callbacks import ModelCheckpoint, TensorBoard

from cells.doi_lstm_cells import DDDoILSTMCell
from metrics.metrics import accuracy, precision, recall, fp_rate, fn_rate

# 时间步长
n_step = 60
# 输入特征
n_input = 87

# 循环层的隐藏单元
rnn_layer_units = 128
# 全连接层的隐藏单元
fully_connected_layer_units = 128
# 输出层的隐藏单元
output_layer_units = 2

# epoch
n_epoch = 100
# mini_batch
n_mini_batch = 32
# 训练数据集的validation_rate比例划分成为验证数据集
n_validation_rate = 0.2

# 模型参数的存储路径
model_parameter_filepath = "./model_parameter/DDDoIGRUCell.hdf5"
# 训练数据集的训练指标的存储路径
train_metrics_filepath = "./train_metrics/DDDoIGRUCell.npy"
# 测试数据集的测试指标的存储路径
test_metrics_filepath = "./test_metrics/DDDoIGRUCell.npy"

# 运行n_time数
n_time = 1


def load_data():
    train_data = np.load("../dataset/train_data.npy")
    train_labels = np.load("../dataset/train_labels.npy")

    test_data = np.load("../dataset/test_data.npy")
    test_labels = np.load("../dataset/test_labels.npy")

    return train_data, train_labels, test_data, test_labels


def build_model(rnn_cell):
    input_layer = Input(shape=(n_step, n_input))
    rnn_layer = RNN(cell=rnn_cell)(input_layer)
    fully_connected_layer = Dense(units=fully_connected_layer_units, activation="relu")(rnn_layer)
    output_layer = Dense(units=output_layer_units, activation="softmax")(fully_connected_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def train(model, train_data, train_labels):
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=[accuracy, precision, recall, fp_rate, fn_rate])

    # tensor_board = TensorBoard(log_dir="./log/DDDoILSTMCell")  # tensorboard --logdir=./

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
                              # callbacks=[tensor_board, model_check_point],
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

cell = DDDoILSTMCell(input_size=n_input, hidden_state_size=rnn_layer_units)
model = build_model(rnn_cell=cell)

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
                          train_data=train_data,
                          train_labels=train_labels)
    np.save(train_metrics_filepath, train_metrics)

    test_metrics_tmp = test(model=model,
                            test_data=test_data,
                            test_labels=test_labels)
    test_metrics["test_loss"].append(test_metrics_tmp[0])
    test_metrics["test_accuracy"].append(test_metrics_tmp[1])
    test_metrics["test_precision"].append(test_metrics_tmp[2])
    test_metrics["test_recall"].append(test_metrics_tmp[3])
    test_metrics["test_fp_rate"].append(test_metrics_tmp[4])
    test_metrics["test_fn_rate"].append(test_metrics_tmp[5])
np.save(test_metrics_filepath, test_metrics)
