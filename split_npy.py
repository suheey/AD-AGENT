import numpy as np
for filename in ["MSL_test_label.npy","MSL_test.npy","MSL_train.npy"]:
    # 读取 .npy 文件
    data = np.load("./Time-Series-Library/dataset/MSL/"+filename)

    # 截取前 10%
    n = int(len(data) * 0.01)
    data_10_percent = data[:n]

    # 可选：保存截取的数据
    np.save("./data/"+filename, data_10_percent)
