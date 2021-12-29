import numpy as np
from mAP import cal_mAP


def readtxt(path):
    label = []
    with open(path, 'r') as f:
        x = f.readlines()
        for name in x:
            temp = int(name.strip().split()[1])
            label.append(temp)
    label = np.array(label)
    return label


root_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/dataset/AID_05/"
train_label = readtxt(root_path + "/train.txt")
test_label = readtxt(root_path + "/test.txt")
label_data = np.concatenate([train_label, test_label], axis=0)
code_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/AID_05/"
bits = [24, 32, 40, 48]

for i in bits:
    path = code_path + str(i) + "bits"
    traincodes = np.sign(np.load(path + "/traincodes.npy"))
    testcodes = np.sign(np.load(path + "/testcodes.npy"))
    data = np.concatenate([traincodes, testcodes], axis=0)
    label = label_data
    querybase = [testcodes, test_label]
    database = [data, label]
    score = cal_mAP(querybase, database, with_top=20)
    print("AID-05 bits {}, map {:.4f}".format(i, score))


root_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/dataset/AID_08/"
train_label = readtxt(root_path + "/train.txt")
test_label = readtxt(root_path + "/test.txt")
label_data = np.concatenate([train_label, test_label], axis=0)
code_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/AID_08/"
bits = [24, 32, 40, 48]

for i in bits:
    path = code_path + str(i) + "bits"
    traincodes = np.sign(np.load(path + "/traincodes.npy"))
    testcodes = np.sign(np.load(path + "/testcodes.npy"))
    data = np.concatenate([traincodes, testcodes], axis=0)
    label = label_data
    querybase = [testcodes, test_label]
    database = [data, label]
    score = cal_mAP(querybase, database, with_top=20)
    print("AID_08 bits {}, map {:.4f}".format(i, score))


root_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/dataset/AID_10/"
train_label = readtxt(root_path + "/train.txt")
test_label = readtxt(root_path + "/test.txt")
label_data = np.concatenate([train_label, test_label], axis=0)
code_path = "/home/admin1/PytorchProject/Meta-Hashing/Dynamic-Meta-hash-AID/AID_10/"
bits = [24, 32, 40, 48]

for i in bits:
    path = code_path + str(i) + "bits"
    traincodes = np.sign(np.load(path + "/traincodes.npy"))
    testcodes = np.sign(np.load(path + "/testcodes.npy"))
    data = np.concatenate([traincodes, testcodes], axis=0)
    label = label_data
    querybase = [testcodes, test_label]
    database = [data, label]
    score = cal_mAP(querybase, database, with_top=20)
    print("AID-10 bits {}, map {:.4f}".format(i, score))
