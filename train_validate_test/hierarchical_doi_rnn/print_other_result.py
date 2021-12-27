import numpy as np

test_metrics_filepath = "./other_test_metrics/DDDoIGRUCell.npy"
test_metrics = np.load(test_metrics_filepath).item()
for key, values in test_metrics.items():
    print("DDDoIGRUCell_%s =" % key, values)

# print(sorted(FirstOrderDifference_GRUCell_test_fn_rate))
GRUCell_test_loss = [0.985755138524549]
GRUCell_test_accuracy = [0.8729281752588121]
GRUCell_test_precision = [0.8429460949221588]
GRUCell_test_recall = [0.768151129916667]
GRUCell_test_fp_rate = [0.07321120963241513]
GRUCell_test_fn_rate = [0.23184888108768956]

DoIGRUCell_test_loss = [0.976073705359717]
DoIGRUCell_test_accuracy = [0.8747697977510386]
DoIGRUCell_test_precision = [0.8369144032234206]
DoIGRUCell_test_recall = [0.7681604324124794]
DoIGRUCell_test_fp_rate = [0.07270632917490234]
DoIGRUCell_test_fn_rate = [0.23183957938770344]

DDoIGRUCell_test_loss = [0.9990216662651048]
DDoIGRUCell_test_accuracy = [0.8563535914895284]
DDoIGRUCell_test_precision = [0.7899965598877403]
DDoIGRUCell_test_recall = [0.7702438806082442]
DDoIGRUCell_test_fp_rate = [0.0996515671302381]
DDoIGRUCell_test_fn_rate = [0.22975613031378547]

DDDoIGRUCell_test_loss = [1.1429152097930346]
DDDoIGRUCell_test_accuracy = [0.8674033135998973]
DDDoIGRUCell_test_precision = [0.8041683731267905]
DDDoIGRUCell_test_recall = [0.7839500070935455]
DDDoIGRUCell_test_fp_rate = [0.09282169275630683]
DDDoIGRUCell_test_fn_rate = [0.21605000778017341]