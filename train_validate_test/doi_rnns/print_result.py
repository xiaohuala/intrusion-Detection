import numpy as np

test_metrics_filepath = "./test_metrics/DDDoIGRUCell.npy"
test_metrics = np.load(test_metrics_filepath).item()
for key, values in test_metrics.items():
    print("DDDoIGRUCell_%s =" % key, values)


GRUCell_test_loss = [1.463128412406528]
GRUCell_test_accuracy = [0.8287292822070323]
GRUCell_test_precision = [0.7366923800160213]
GRUCell_test_recall = [0.7607325606363693]
GRUCell_test_fp_rate = [0.1299313611417844]
GRUCell_test_fn_rate = [0.2392674481726044]

DoIGRUCell_test_loss = [1.1088416698229246]
DoIGRUCell_test_accuracy = [0.8508287295013063]
DoIGRUCell_test_precision = [0.7966693857098153]
DoIGRUCell_test_recall = [0.7399055078543352]
DoIGRUCell_test_fp_rate = [0.093138563932086]
DoIGRUCell_test_fn_rate = [0.2600945062235574]

DDoIGRUCell_test_loss = [1.242570508250874]
DDoIGRUCell_test_accuracy = [0.8416206249435523]
DDoIGRUCell_test_precision = [0.7718059750990753]
DDoIGRUCell_test_recall = [0.7664415828650388]
DDoIGRUCell_test_fp_rate = [0.11769902986057555]
DDoIGRUCell_test_fn_rate = [0.2335584249834549]

DDDoIGRUCell_test_loss = [1.1016571844480314]
DDDoIGRUCell_test_accuracy = [0.8545119708633774]
DDDoIGRUCell_test_precision = [0.7837403228928371]
DDDoIGRUCell_test_recall = [0.7610022917417312]
DDDoIGRUCell_test_fp_rate = [0.09698601250929507]
DDDoIGRUCell_test_fn_rate = [0.23899771654583912]